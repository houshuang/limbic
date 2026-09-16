"""Codex CLI wrapper — structured + agentic generation via `codex exec`.

The Codex counterpart to `limbic.cerebellum.claude_cli`. Consolidates two
patterns that were copy-pasted across projects (alif's
`backend/app/services/codex_cli.py` and dragoman's `llm.py`):

- ``codex_json``     — locked-down, read-only, structured single-shot. Use when
                       you just want a model to return JSON for some text.
- ``codex_research`` — DELIBERATELY agentic: web search on + a writable
                       workspace with network egress, so Codex can search the
                       open web, fetch and read pages, follow leads, and write
                       files. Use for enrichment/discovery where a single prompt
                       isn't enough.

Both return parsed JSON when given a schema (via Codex's ``--output-schema`` +
``--output-last-message``), or raw text otherwise. Both strip ``CLAUDECODE`` so
nested invocation from inside a Claude Code session doesn't inherit parent state.

Codex is free under the user's ChatGPT subscription, so (unlike claude_cli)
these calls are not written to ``cost_log`` — there is no Codex cost adapter.

Usage::

    from limbic.cerebellum.codex_cli import codex_json, codex_research

    out = codex_json("Classify sentiment of: I love it", schema=SCHEMA)

    dossier = codex_research(
        "Research X. Web-search anything ambiguous. Write findings to out.json.",
        schema=SCHEMA, scratch_dir="/tmp/run",
    )
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = os.environ.get("LIMBIC_CODEX_MODEL", "gpt-5.5")
DEFAULT_REASONING = os.environ.get("LIMBIC_CODEX_REASONING", "medium")
QUOTA_COOLDOWN_S = int(os.environ.get("LIMBIC_CODEX_QUOTA_COOLDOWN_S", "21600"))
# Cap on captured stdout/stderr per call. An agentic run is otherwise free to
# stream until the parent runs out of memory.
OUTPUT_LIMIT = int(os.environ.get("LIMBIC_CODEX_OUTPUT_LIMIT", str(2 * 1024 * 1024)))

# Keys never passed through to the child. CLAUDECODE is stripped so a nested call
# from a Claude Code session doesn't inherit in-session state. (We deliberately
# leave OpenAI auth env alone so Codex uses whatever its own `codex auth`
# precedence dictates.)
_STRIPPED_ENV_KEYS = frozenset({"CLAUDECODE"})

_DISABLED_UNTIL = 0.0
_DISABLED_REASON = ""


class CodexCLIError(RuntimeError):
    """Raised when `codex exec` is missing, exits non-zero, times out, or returns malformed output."""


def is_available() -> bool:
    """Whether the `codex` binary is on PATH."""
    return shutil.which("codex") is not None


def _is_quota_error(message: str) -> bool:
    msg = (message or "").lower()
    return any(m in msg for m in (
        "out of extra usage", "usage limit", "rate limit", "too many requests", "quota",
    ))


def mark_unavailable_from_error(error: Exception | str) -> None:
    """Cool down on quota errors so a cron run stops hammering a depleted quota.

    Process-local — each new invocation gets one chance to discover recovery.
    """
    global _DISABLED_UNTIL, _DISABLED_REASON
    message = str(error)
    if not _is_quota_error(message):
        return
    _DISABLED_UNTIL = max(_DISABLED_UNTIL, time.time() + max(1, QUOTA_COOLDOWN_S))
    _DISABLED_REASON = message[:200]


def temporarily_disabled() -> bool:
    return time.time() < _DISABLED_UNTIL


def disabled_reason() -> str:
    return _DISABLED_REASON


def _codex_env() -> dict:
    """The child's environment, read live rather than snapshotted at import.

    A snapshot made this wrapper ignore any env change a caller made afterwards —
    including scrubbing secrets before handing an agent untrusted text, and
    including PATH. Callers can now scope the environment with the usual
    os.environ juggling (or `limbic.cerebellum.sandbox.sanitized_environment`)
    and have it actually reach the subprocess.
    """
    return {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV_KEYS}


def _allow_null(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    out = copy.deepcopy(node)
    typ = out.get("type")
    if isinstance(typ, str) and typ != "null":
        out["type"] = [typ, "null"]
    elif isinstance(typ, list) and "null" not in typ:
        out["type"] = [*typ, "null"]
    return out


def strict_response_schema(schema: dict) -> dict:
    """Convert a permissive JSON Schema into Codex strict structured-output shape:
    ``additionalProperties:false``, every property in ``required``, formerly-optional
    fields made nullable. (Ported from alif/polyglot.)
    """
    def _walk(node: Any) -> Any:
        if isinstance(node, list):
            return [_walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        out = {k: _walk(v) for k, v in node.items()}
        props = out.get("properties")
        if isinstance(props, dict):
            req = set(node.get("required", []) or [])
            out["properties"] = {n: v if n in req else _allow_null(v) for n, v in props.items()}
            out["additionalProperties"] = False
            out["required"] = list(props.keys())
        return out

    return _walk(copy.deepcopy(schema))


def _parse_json(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    return None


class _Tail:
    """Thread-safe bounded accumulator keeping the LAST ``limit`` characters.

    An agentic run can emit unbounded output, and the diagnostic value is at the
    end (see ``_finish``), so the head is what gets dropped.
    """

    def __init__(self, limit: int):
        self._limit = limit
        self._value = ""
        self._discarded = 0
        self._lock = threading.Lock()

    def add(self, chunk: str) -> None:
        with self._lock:
            value = self._value + chunk
            if len(value) > self._limit:
                cut = len(value) - self._limit
                self._discarded += cut
                value = value[cut:]
            self._value = value

    def get(self) -> str:
        with self._lock:
            prefix = (f"[... {self._discarded} earlier characters discarded ...]\n"
                      if self._discarded else "")
            return prefix + self._value


def _kill_process_tree(proc: subprocess.Popen, grace: float = 2.0) -> None:
    """SIGTERM then SIGKILL the child's whole process group.

    ``codex exec`` spawns helpers; killing only the direct child (which is all
    ``subprocess.run(timeout=...)`` does) leaves them running and holding the
    workspace open.
    """
    if os.name != "posix":  # pragma: no cover - POSIX is the supported target
        try:
            proc.terminate()
            proc.wait(timeout=grace)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.05)


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Run `codex exec` with bounded output and a process-group kill on timeout."""
    if not is_available():
        raise CodexCLIError("codex CLI not available — install from https://github.com/openai/codex and run `codex auth`")
    if temporarily_disabled():
        raise CodexCLIError(f"codex CLI temporarily disabled (quota): {_DISABLED_REASON}")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=_codex_env(), stdin=subprocess.DEVNULL,  # codex exec blocks on stdin otherwise
            start_new_session=(os.name == "posix"),      # own process group, so we can kill the tree
        )
    except (FileNotFoundError, OSError) as exc:
        raise CodexCLIError(f"codex CLI subprocess error: {exc}") from exc

    out, err = _Tail(OUTPUT_LIMIT), _Tail(OUTPUT_LIMIT)

    def drain(stream, tail):
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                tail.add(chunk)
        except (OSError, ValueError):
            return

    readers = [threading.Thread(target=drain, args=(proc.stdout, out), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, err), daemon=True)]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL already sent
            proc.kill()
            proc.wait()
    finally:
        for reader in readers:
            reader.join(timeout=1)
        if any(reader.is_alive() for reader in readers):
            _kill_process_tree(proc, grace=0.2)
            for reader in readers:
                reader.join(timeout=1)
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
    if timed_out:
        raise CodexCLIError(f"codex CLI timed out after {timeout}s")
    return subprocess.CompletedProcess(cmd, proc.returncode, out.get(), err.get())


def _finish(proc: subprocess.CompletedProcess, output_path: str | None, schema: dict | None) -> Any:
    if proc.returncode != 0:
        full = (proc.stderr or "") or (proc.stdout or "")
        mark_unavailable_from_error(full)
        err = full.strip()
        # codex exec opens stderr with its banner (and a harmless "Reading
        # additional input from stdin..." whenever stdin isn't a TTY); the fatal
        # error is at the END — keep the tail, not just the head.
        if len(err) > 700:
            err = err[:200] + " […] " + err[-500:]
        raise CodexCLIError(f"codex CLI exit {proc.returncode}: {err}")
    text = ""
    if output_path:
        try:
            text = Path(output_path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
    if not text:
        text = (proc.stdout or "").strip()
    if schema is None:
        return text
    parsed = _parse_json(text)
    if parsed is None:
        raise CodexCLIError(f"codex CLI returned unparseable JSON: {text[:300]}")
    return parsed


RETRIES = int(os.environ.get("LIMBIC_CODEX_RETRIES", "1"))


def _is_transient(err: CodexCLIError) -> bool:
    msg = str(err)
    if _is_quota_error(msg):
        return False
    return "codex CLI exit" in msg or "unparseable JSON" in msg


def _exec(cmd: list[str], timeout: int, output_path: str | None, schema: dict | None) -> Any:
    """Run + parse with automatic retry on transient failures (a flaky `codex exec`
    non-zero exit or garbled output). Quota errors and timeouts do NOT retry: quota
    needs its cooldown, and a timeout retry would double the stage's worst case.
    """
    for attempt in range(RETRIES + 1):
        try:
            return _finish(_run(cmd, timeout), output_path, schema)
        except CodexCLIError as e:
            if attempt >= RETRIES or not _is_transient(e):
                raise
            time.sleep(5 * (attempt + 1))


def codex_json(
    prompt: str,
    *,
    schema: dict | None = None,
    system: str = "",
    model: str | None = None,
    reasoning: str | None = None,
    timeout: int = 120,
    sandbox: str = "read-only",
) -> Any:
    """Locked-down, single-shot structured generation. Returns parsed JSON (if
    ``schema`` given) or text. No web, no file writes — just classify/transform.
    """
    schema_path = output_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as of:
            output_path = of.name
        cmd = ["codex", "exec", "--model", model or DEFAULT_MODEL,
               "--sandbox", sandbox, "--skip-git-repo-check", "--ephemeral",
               "--ignore-user-config", "--color", "never",
               "--output-last-message", output_path,
               "-c", f'model_reasoning_effort="{reasoning or DEFAULT_REASONING}"']
        if schema is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".schema.json", delete=False) as sf:
                json.dump(strict_response_schema(schema), sf, ensure_ascii=False)
                schema_path = sf.name
            cmd += ["--output-schema", schema_path]
        cmd.append(f"{system}\n\n{prompt}" if system else prompt)
        return _exec(cmd, timeout, output_path, schema)
    finally:
        for p in (schema_path, output_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def codex_research(
    mission: str,
    *,
    schema: dict | None = None,
    scratch_dir: str | None = None,
    add_dirs: list[str] | None = None,
    images: list[str] | None = None,
    model: str | None = None,
    reasoning: str | None = None,
    timeout: int = 900,
    web_search: bool = True,
    network: bool = True,
    isolated: bool = True,
) -> Any:
    """Agentic Codex run: web search + writable workspace with network egress.

    The two flags that unlock the agent (omit both and it degrades to a shallow
    one-shot): ``tools.web_search=true`` and
    ``sandbox_workspace_write.network_access=true``. Returns parsed JSON (if
    ``schema`` given) or text. If ``scratch_dir`` is given, Codex runs there and
    may write files; pass ``add_dirs`` for extra writable roots (e.g. a project
    ``data/`` dir).

    ``images`` attaches local image files to the initial prompt (``codex exec -i``),
    which is the only way the model sees pixels: a path in the prompt text is just
    text. Callers own the files and their lifetime — pass paths inside the same
    short-lived workspace, and remember an attached image is untrusted input that
    can carry rendered instructions.

    ``isolated`` (default on, matching ``codex_json``) adds ``--ephemeral`` and
    ``--ignore-user-config``, so the run leaves no rollout behind and reads none
    of the host's ``~/.codex`` settings. This matters *more* here than for
    ``codex_json``: this is the call that reads untrusted web pages with network
    egress. Model and reasoning are always passed explicitly, so ignoring user
    config changes nothing about which model runs. Pass ``isolated=False`` only
    when the run genuinely needs the host profile (e.g. a locally configured MCP
    server).
    """
    schema_path = output_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as of:
            output_path = of.name
        cmd = ["codex", "exec", "--model", model or DEFAULT_MODEL,
               "--sandbox", "workspace-write", "--skip-git-repo-check", "--color", "never",
               "--output-last-message", output_path,
               "-c", f'model_reasoning_effort="{reasoning or DEFAULT_REASONING}"']
        if isolated:
            cmd[2:2] = ["--ephemeral", "--ignore-user-config"]
        if web_search:
            cmd += ["-c", "tools.web_search=true"]
        if network:
            cmd += ["-c", "sandbox_workspace_write.network_access=true"]
        if scratch_dir:
            Path(scratch_dir).mkdir(parents=True, exist_ok=True)
            cmd += ["-C", scratch_dir, "--add-dir", scratch_dir]
        for d in (add_dirs or []):
            cmd += ["--add-dir", d]
        for image in (images or []):
            path = Path(image)
            if not path.is_file():
                raise CodexCLIError(f"codex image attachment is not a file: {image}")
            cmd += ["-i", str(path)]
        if schema is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".schema.json", delete=False) as sf:
                json.dump(strict_response_schema(schema), sf, ensure_ascii=False)
                schema_path = sf.name
            cmd += ["--output-schema", schema_path]
        cmd.append(mission)
        return _exec(cmd, timeout, output_path, schema)
    finally:
        for p in (schema_path, output_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
