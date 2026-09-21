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

Codex is free under the user's ChatGPT subscription, so no money is billed per
token — but the tokens are real, and where they go was invisible: an audit of
one project found 1.28 billion tokens in eight days, none of it in any ledger,
because every call went through this transport. So both entry points now ask
Codex for its event stream (``--json``), parse the per-turn usage, and write one
``cost_log`` row per attempt with ``billing_mode="subscription"``: ``cost_usd``
is 0 (nothing was spent) and ``notional_cost_usd`` carries what the same tokens
would have cost on the API. See ``docs/cost-log.md``.

Set ``LIMBIC_CODEX_COST_LOG=0``, or pass ``cost_log=False``, to turn the whole
thing off — the call then runs exactly as it did before, ``--json`` and all.

Usage::

    from limbic.cerebellum.codex_cli import codex_json, codex_research

    out = codex_json("Classify sentiment of: I love it", schema=SCHEMA,
                     project="myapp", purpose="sentiment")

    dossier = codex_research(
        "Research X. Web-search anything ambiguous. Write findings to out.json.",
        schema=SCHEMA, scratch_dir="/tmp/run",
    )
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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
    """Raised when `codex exec` is missing, exits non-zero, times out, or returns malformed output.

    Carries whatever output the run produced before it failed, so a caller (and
    the usage logger) can still read the token counts of an attempt that died
    after the model had already worked. Both default to "" — `str(exc)` is
    unchanged, so existing `except CodexCLIError` handlers are unaffected.
    """

    def __init__(self, message: str, *, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


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


# ---------------------------------------------------------------------------
# Usage capture — `codex exec --json` events to a cost_log row
# ---------------------------------------------------------------------------

# Whether this process has seen the CLI reject `--json`. Flipped once, then the
# flag is simply never passed again (see `_run`).
_JSON_EVENTS_SUPPORTED = True


def cost_log_enabled() -> bool:
    """Whether usage capture is on. Read live, so a caller can toggle it."""
    return os.environ.get("LIMBIC_CODEX_COST_LOG", "1").strip().lower() not in (
        "0", "false", "no", "off")


@dataclass
class CodexUsage:
    """Token counts for one `codex exec` run."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    turns: int = 0
    thread_id: str = ""
    found: bool = False


# Alternate spellings seen across Codex CLI versions, newest first. A key the
# running CLI doesn't emit is simply absent.
_USAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "prompt_tokens", "input"),
    "cached_input_tokens": ("cached_input_tokens", "cached_tokens",
                            "cache_read_input_tokens"),
    "cache_write_input_tokens": ("cache_write_input_tokens",
                                 "cache_creation_input_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens", "output"),
    "reasoning_output_tokens": ("reasoning_output_tokens", "reasoning_tokens"),
}


def _pick(block: dict, names: tuple[str, ...]) -> int:
    for name in names:
        value = block.get(name)
        if isinstance(value, bool):  # a stray flag is not a token count
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _iter_events(stdout: str):
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def parse_usage(stdout: str) -> CodexUsage:
    """Token usage from the JSONL that `codex exec --json` writes to stdout.

    The shape, captured from codex-cli 0.153.4 on 2026-09-21::

        {"type": "thread.started", "thread_id": "01a0c404-…"}
        {"type": "turn.started"}
        {"type": "item.completed", "item": {"id": "item_0",
         "type": "agent_message", "text": "ok"}}
        {"type": "turn.completed", "usage": {"input_tokens": 14169,
         "cached_input_tokens": 4480, "cache_write_input_tokens": 0,
         "output_tokens": 5, "reasoning_output_tokens": 0}}

    `input_tokens` includes `cached_input_tokens`, and `output_tokens` includes
    `reasoning_output_tokens` — the OpenAI Responses convention, which is also
    what `cost_for` assumes of its `prompt_tokens`/`cached_tokens` arguments.

    Per-turn `usage` blocks are **summed**: one `codex exec` is normally one
    turn, but `exec resume` adds more. Older CLIs emit a running
    `msg.info.total_token_usage` instead, which is cumulative for the whole
    thread — that form **replaces** rather than adds, or a long run would be
    counted once per event. `found` distinguishes "no tokens" from "this CLI
    told us nothing", which is what keeps a zero row honest.
    """
    usage = CodexUsage()
    cumulative: dict | None = None
    for event in _iter_events(stdout):
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str) and thread_id and not usage.thread_id:
            usage.thread_id = thread_id

        block = event.get("usage")
        if isinstance(block, dict):
            usage.turns += 1
            usage.found = True
            for attr, names in _USAGE_ALIASES.items():
                setattr(usage, attr, getattr(usage, attr) + _pick(block, names))
            continue

        info = event.get("msg")
        info = info.get("info") if isinstance(info, dict) else None
        if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
            cumulative = info["total_token_usage"]

    if cumulative is not None and not usage.found:
        usage.found = True
        usage.turns = max(usage.turns, 1)
        for attr, names in _USAGE_ALIASES.items():
            setattr(usage, attr, _pick(cumulative, names))
    return usage


def final_message(stdout: str) -> str:
    """The agent's last message, recovered from `--json` event output.

    Only used when `--output-last-message` produced nothing. Returns "" if the
    stream holds no agent message, so the caller can fall back to raw stdout
    exactly as it did before event capture existed.
    """
    text = ""
    for event in _iter_events(stdout):
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            if isinstance(item.get("text"), str):
                text = item["text"]
            continue
        msg = event.get("msg")
        if isinstance(msg, dict) and msg.get("type") == "agent_message":
            for key in ("message", "text"):
                if isinstance(msg.get(key), str):
                    text = msg[key]
                    break
    return text.strip()


@dataclass
class _LogContext:
    """What a run needs to attribute its own ledger row."""

    model: str
    script: str
    project: str = ""
    purpose: str = ""
    packet_id: str | None = None
    enabled: bool = True
    metadata: dict = field(default_factory=dict)


_PROJECT_CACHE: dict[str, str] = {}


def _project(explicit: str) -> str:
    """Attribution tag for a ledger row: explicit, else env, else the git root.

    Falls back to "unattributed" rather than raising, unlike `calls._infer_project`
    — this runs *after* a call has already burned its tokens, and a failure to
    name the project must not take the result down with it. The name is
    deliberately conspicuous: an unattributed row is a bug to fix, not a row to
    quietly fold into some other project's total.
    """
    if explicit:
        return explicit
    env = os.environ.get("LIMBIC_CODEX_PROJECT", "").strip()
    if env:
        return env
    cwd = os.getcwd()
    if cwd not in _PROJECT_CACHE:
        try:
            from .calls import _infer_project
            _PROJECT_CACHE[cwd] = _infer_project()
        except Exception:
            _PROJECT_CACHE[cwd] = "unattributed"
    return _PROJECT_CACHE[cwd]


def _log_usage(ctx: _LogContext | None, usage: CodexUsage, *,
               duration_ms: int, error: str = "") -> None:
    """Write one subscription row for a `codex exec` attempt. Never raises.

    Called on the success path *and* on every failed attempt — a timeout that
    burned 300k tokens before dying is exactly the spend the audit went looking
    for, and a failed attempt with no usage at all still leaves a zero-token row
    with an error marker so the attempt itself is countable.
    """
    if ctx is None or not ctx.enabled or not cost_log_enabled():
        return
    try:
        from .cost_log import UnknownModelPriceError, cost_for, cost_log

        try:
            notional = cost_for(ctx.model, usage.input_tokens, usage.output_tokens,
                                usage.cached_input_tokens, strict=True)
        except UnknownModelPriceError:
            # Tokens are still worth recording; an invented price is not.
            notional = None

        metadata = dict(ctx.metadata)
        metadata.update({
            "transport": "codex_cli",
            "billing": "chatgpt_subscription",
            "duration_ms": duration_ms,
            "turns": usage.turns,
        })
        if usage.thread_id:
            metadata["thread_id"] = usage.thread_id
        if usage.reasoning_output_tokens:
            metadata["reasoning_output_tokens"] = usage.reasoning_output_tokens
        if usage.cache_write_input_tokens:
            metadata["cache_write_input_tokens"] = usage.cache_write_input_tokens
        if notional is None:
            metadata["notional_cost"] = f"unknown price for model {ctx.model}"
        if not usage.found:
            metadata["usage"] = "no usage events in output"
        if error:
            metadata["error"] = error[:500]

        cost_log.log(
            project=_project(ctx.project),
            model=ctx.model,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cached_tokens=usage.cached_input_tokens,
            cost_usd=0.0,
            billing_mode="subscription",
            notional_cost_usd=notional,
            script=ctx.script,
            purpose=ctx.purpose,
            metadata=metadata,
            packet_id=ctx.packet_id,
            outcome="error" if error else None,
        )
    except Exception as failure:  # the call already happened; bookkeeping must not undo it
        log.warning("codex usage ledger write failed (%s): %s", ctx.purpose, failure)


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


def _join_readers(readers: list[threading.Thread], budget: float) -> None:
    """Join every reader within one shared ``budget``, not ``budget`` each.

    Per-thread timeouts multiply: the overhead a caller sees on the abnormal path
    has to be a fixed bound, not a bound per stream.
    """
    deadline = time.monotonic() + budget
    for reader in readers:
        reader.join(timeout=max(0.0, deadline - time.monotonic()))


# clap (the Codex CLI's argument parser) exits 2 on a usage error.
_ARGPARSE_EXIT_CODE = 2


def _rejects_json_flag(proc: subprocess.CompletedProcess) -> bool:
    """Whether this exit is the CLI refusing to parse `--json`, not a model failure.

    An argument-parse failure costs nothing and happens before any model runs,
    so retrying without the flag is free. Without this check, a host on a Codex
    build predating `--json` would have every call fail the moment it picked up
    this version of limbic.

    All three conditions have to hold, because the retry is only free while the
    run genuinely never started. A model failure whose text happens to quote the
    flag would otherwise re-run an agentic mission that had already spent its
    tokens — and disable usage logging for the rest of the process on the way.
    So: the parser's own exit code, *no* event on stdout (the run never got as
    far as emitting one), and the complaint on stderr where clap writes it.
    """
    if proc.returncode != _ARGPARSE_EXIT_CODE:
        return False
    if next(_iter_events(proc.stdout or ""), None) is not None:
        return False
    text = (proc.stderr or "").lower()
    if "--json" not in text:
        return False
    return any(m in text for m in (
        "unexpected argument", "unrecognized", "unknown flag", "invalid option",
        "unknown option",
    ))


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Run `codex exec` with bounded output and a process-group kill on timeout."""
    if not is_available():
        raise CodexCLIError("codex CLI not available — install from https://github.com/openai/codex and run `codex auth`")
    if temporarily_disabled():
        raise CodexCLIError(f"codex CLI temporarily disabled (quota): {_DISABLED_REASON}")
    proc = _spawn(cmd, timeout)
    if proc.returncode != 0 and "--json" in cmd and _rejects_json_flag(proc):
        global _JSON_EVENTS_SUPPORTED
        _JSON_EVENTS_SUPPORTED = False
        log.warning("codex CLI does not support --json; retrying without it. "
                    "Token usage will not be logged for this process.")
        proc = _spawn([a for a in cmd if a != "--json"], timeout)
    return proc


def _spawn(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            errors="replace",  # one undecodable byte must not kill a drain thread
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
        _join_readers(readers, budget=1.0)
        if any(reader.is_alive() for reader in readers):
            _kill_process_tree(proc, grace=0.2)
            _join_readers(readers, budget=1.0)
        # Close only the pipes whose reader has finished. A descendant that left
        # the process group (its own setsid) survives the kill and still holds
        # the write end, so its reader is parked in read() holding the buffer
        # lock that close() needs — closing would block for as long as that
        # orphan lives, silently blowing through the timeout just enforced.
        # Leaking one fd to a daemon thread is the cheaper failure.
        for reader, stream in zip(readers, (proc.stdout, proc.stderr)):
            if reader.is_alive():
                continue
            try:
                stream.close()
            except (OSError, ValueError):
                pass
    if timed_out:
        raise CodexCLIError(f"codex CLI timed out after {timeout}s",
                            stdout=out.get(), stderr=err.get())
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
        raise CodexCLIError(f"codex CLI exit {proc.returncode}: {err}",
                            stdout=proc.stdout or "", stderr=proc.stderr or "")
    text = ""
    if output_path:
        try:
            text = Path(output_path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
    if not text:
        args = proc.args if isinstance(proc.args, (list, tuple)) else ()
        if "--json" in args:
            # stdout is an event stream, not an answer. If no recognised event
            # carried the agent's message — an older CLI with an event shape we
            # have never seen — then this run produced no output, and it has to
            # fail as an empty result always did. Handing back raw JSONL would
            # be silent corruption: a schema-less caller would store the
            # transcript as the answer and nothing would look wrong.
            text = final_message(proc.stdout or "")
        else:
            text = (proc.stdout or "").strip()
    if schema is None:
        return text
    parsed = _parse_json(text)
    if parsed is None:
        raise CodexCLIError(f"codex CLI returned unparseable JSON: {text[:300]}",
                            stdout=proc.stdout or "", stderr=proc.stderr or "")
    return parsed


RETRIES = int(os.environ.get("LIMBIC_CODEX_RETRIES", "1"))


def _is_transient(err: CodexCLIError) -> bool:
    msg = str(err)
    if _is_quota_error(msg):
        return False
    return "codex CLI exit" in msg or "unparseable JSON" in msg


def _exec(cmd: list[str], timeout: int, output_path: str | None,
          schema: dict | None, ctx: _LogContext | None = None) -> Any:
    """Run + parse with automatic retry on transient failures (a flaky `codex exec`
    non-zero exit or garbled output). Quota errors and timeouts do NOT retry: quota
    needs its cooldown, and a timeout retry would double the stage's worst case.

    Every attempt gets its own ledger row — a retried call spent its first
    attempt's tokens too.
    """
    for attempt in range(RETRIES + 1):
        started = time.monotonic()
        try:
            proc = _run(cmd, timeout)
            result = _finish(proc, output_path, schema)
        except CodexCLIError as e:
            _log_usage(ctx, parse_usage(getattr(e, "stdout", "")),
                       duration_ms=int((time.monotonic() - started) * 1000),
                       error=str(e))
            if attempt >= RETRIES or not _is_transient(e):
                raise
            time.sleep(5 * (attempt + 1))
            continue
        _log_usage(ctx, parse_usage(proc.stdout or ""),
                   duration_ms=int((time.monotonic() - started) * 1000))
        return result


def codex_json(
    prompt: str,
    *,
    schema: dict | None = None,
    system: str = "",
    model: str | None = None,
    reasoning: str | None = None,
    timeout: int = 120,
    sandbox: str = "read-only",
    project: str = "",
    purpose: str = "",
    packet_id: str | None = None,
    cost_log: bool = True,
) -> Any:
    """Locked-down, single-shot structured generation. Returns parsed JSON (if
    ``schema`` given) or text. No web, no file writes — just classify/transform.

    ``project``/``purpose``/``packet_id`` attribute the usage row this writes;
    ``project`` falls back to ``$LIMBIC_CODEX_PROJECT`` and then the enclosing
    git repo's name. ``cost_log=False`` (or ``LIMBIC_CODEX_COST_LOG=0``) skips
    the row and the ``--json`` flag it needs.
    """
    resolved_model = model or DEFAULT_MODEL
    ctx = _LogContext(model=resolved_model, script="codex_cli.codex_json",
                      project=project, purpose=purpose, packet_id=packet_id,
                      enabled=cost_log)
    schema_path = output_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as of:
            output_path = of.name
        cmd = ["codex", "exec", "--model", resolved_model,
               "--sandbox", sandbox, "--skip-git-repo-check", "--ephemeral",
               "--ignore-user-config", "--color", "never",
               "--output-last-message", output_path,
               "-c", f'model_reasoning_effort="{reasoning or DEFAULT_REASONING}"']
        if cost_log and cost_log_enabled() and _JSON_EVENTS_SUPPORTED:
            cmd.append("--json")
        if schema is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".schema.json", delete=False) as sf:
                json.dump(strict_response_schema(schema), sf, ensure_ascii=False)
                schema_path = sf.name
            cmd += ["--output-schema", schema_path]
        cmd.append(f"{system}\n\n{prompt}" if system else prompt)
        return _exec(cmd, timeout, output_path, schema, ctx)
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
    project: str = "",
    purpose: str = "",
    packet_id: str | None = None,
    cost_log: bool = True,
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
    of the host's ``~/.codex/config.toml``. This matters *more* here than for
    ``codex_json``: this is the call that reads untrusted web pages with network
    egress.

    Model and reasoning are always passed explicitly, so this never changes which
    model runs — but it does drop *everything else* the host profile sets, which
    on a developer machine can include ``service_tier``, ``notify``,
    ``personality`` and any configured MCP servers. Pass ``isolated=False`` when
    the run genuinely needs those.

    ``project``/``purpose``/``packet_id``/``cost_log`` behave as in
    ``codex_json``. This is the call worth attributing: an agentic run reading
    web pages for ten minutes is where the tokens actually go.
    """
    resolved_model = model or DEFAULT_MODEL
    ctx = _LogContext(model=resolved_model, script="codex_cli.codex_research",
                      project=project, purpose=purpose, packet_id=packet_id,
                      enabled=cost_log)
    schema_path = output_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as of:
            output_path = of.name
        cmd = ["codex", "exec", "--model", resolved_model,
               "--sandbox", "workspace-write", "--skip-git-repo-check", "--color", "never",
               "--output-last-message", output_path,
               "-c", f'model_reasoning_effort="{reasoning or DEFAULT_REASONING}"']
        if isolated:
            cmd[2:2] = ["--ephemeral", "--ignore-user-config"]
        if cost_log and cost_log_enabled() and _JSON_EVENTS_SUPPORTED:
            cmd.append("--json")
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
        return _exec(cmd, timeout, output_path, schema, ctx)
    finally:
        for p in (schema_path, output_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
