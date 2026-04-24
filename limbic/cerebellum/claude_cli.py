"""Claude CLI wrapper — structured generation via `claude -p`, auto-logged to cost_log.

Usage:

    from limbic.cerebellum.claude_cli import generate

    result, meta = generate(
        prompt="Classify this sentiment: I love it",
        project="myapp",
        purpose="sentiment",
        schema={"type": "object", "properties": {"label": {"type": "string"}}},
    )

Every invocation writes one or more rows to `limbic.cerebellum.cost_log` with
`script="claude-cli"`. Multi-model sessions (rare for single-call headless mode)
write one row per model, sharing the same `session_id` in metadata.

The wrapper always runs with `--no-session-persistence` and strips
`CLAUDECODE` + `ANTHROPIC_API_KEY` / `ANTHROPIC_KEY` / `ANTHROPIC_AUTH_TOKEN`
from the child env so the CLI uses Max/OAuth auth (matching the cost_log's
subscription-value accounting) rather than silently billing an API key the
caller may have set for an SDK path.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .cost_log import cost_log

log = logging.getLogger(__name__)

# Env vars we strip before spawning `claude -p`:
# - CLAUDECODE: nested invocation from inside a Claude Code session would otherwise
#   inherit the parent's in-session state.
# - ANTHROPIC_API_KEY / ANTHROPIC_KEY / ANTHROPIC_AUTH_TOKEN: the CLI prefers these
#   over its own Max/OAuth login, which would bill API usage *and* break the cost_log
#   attribution model (CLI rows assume subscription-value accounting — see
#   CHANGELOG § "Cost dashboard surfaces CLI subscription value"). Callers who hold
#   an API key for the SDK path should not double-bill when this wrapper falls back
#   to the CLI.
_STRIPPED_ENV_KEYS = frozenset({
    "CLAUDECODE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_KEY",
    "ANTHROPIC_AUTH_TOKEN",
})
_ENV = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV_KEYS}


class ClaudeCLIError(RuntimeError):
    """Raised when `claude -p` exits non-zero, times out, or returns malformed output."""


def is_available() -> bool:
    """Check whether the `claude` CLI is installed and on PATH."""
    return shutil.which("claude") is not None


def _build_cmd(
    *,
    model: str,
    system: str,
    schema: dict | None,
    tools: str | None,
    allowed_tools: str | None,
    max_turns: int | None,
    max_budget: float | None,
    work_dir: str | None,
    dangerously_skip_permissions: bool,
) -> list[str]:
    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--no-session-persistence",
        "--model", model,
    ]
    if tools is not None:
        cmd.extend(["--tools", tools])
    if allowed_tools:
        cmd.extend(["--allowedTools", allowed_tools])
    if schema:
        cmd.extend(["--json-schema", json.dumps(schema)])
    if system:
        cmd.extend(["--system-prompt", system])
    if max_turns is not None:
        cmd.extend(["--max-turns", str(max_turns)])
    if max_budget is not None:
        cmd.extend(["--max-budget-usd", str(max_budget)])
    if work_dir:
        cmd.extend(["--add-dir", work_dir])
    if dangerously_skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    return cmd


def log_cli_usage(
    response: dict,
    *,
    project: str,
    purpose: str = "",
    requested_model: str = "",
) -> None:
    """Public helper: write cost_log rows from a `claude -p --output-format json` response.

    Used by `generate()` internally, and also by projects that call `claude -p`
    via their own subprocess (async, streaming, etc.) but still want rows in the
    shared cost_log. Writes one row per model that the session touched.

    Trusts `costUSD` from Claude's `modelUsage` block. Falls back to top-level
    `usage` + `total_cost_usd` if `modelUsage` is missing (older CLI versions).
    `requested_model` is stored in metadata and used as the row's model when
    `modelUsage` is absent.
    """
    session_id = response.get("session_id")
    base_metadata: dict[str, Any] = {
        "session_id": session_id,
        "duration_ms": response.get("duration_ms"),
        "num_turns": response.get("num_turns"),
        "headless": True,
        "requested_model": requested_model,
    }
    if response.get("is_error"):
        base_metadata["failed"] = True

    model_usage = response.get("modelUsage") or {}
    if model_usage:
        for model_id, usage in model_usage.items():
            metadata = dict(base_metadata)
            wsr = usage.get("webSearchRequests", 0)
            if wsr:
                metadata["web_search_requests"] = wsr
            try:
                cost_log.log(
                    project=project,
                    model=model_id,
                    prompt_tokens=int(usage.get("inputTokens", 0) or 0),
                    completion_tokens=int(usage.get("outputTokens", 0) or 0),
                    cached_tokens=int(usage.get("cacheReadInputTokens", 0) or 0),
                    cost_usd=float(usage.get("costUSD", 0.0) or 0.0),
                    script="claude-cli",
                    purpose=purpose,
                    metadata=metadata,
                )
            except Exception as e:
                log.warning("cost_log.log failed for session %s: %s", session_id, e)
        return

    usage = response.get("usage") or {}
    try:
        cost_log.log(
            project=project,
            model=requested_model,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            cached_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cost_usd=float(response.get("total_cost_usd", 0.0) or 0.0),
            script="claude-cli",
            purpose=purpose,
            metadata=base_metadata,
        )
    except Exception as e:
        log.warning("cost_log.log failed for session %s: %s", session_id, e)


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```")


def _parse_result(response: dict, schema: dict | None) -> Any:
    """Extract the result payload from Claude's JSON response.

    Prefers `structured_output` (from `--json-schema`, constrained decoding).
    Falls back to parsing the `result` string as JSON when a schema was requested,
    stripping markdown code fences if present. Returns the raw string otherwise.
    """
    structured = response.get("structured_output")
    if structured is not None:
        return structured

    text = (response.get("result") or "").strip()
    if schema:
        if text.startswith("```"):
            m = _FENCE_RE.search(text)
            if m:
                text = m.group(1).strip()
        if not text:
            return None
        return json.loads(text)
    return text


def generate(
    prompt: str,
    *,
    project: str,
    purpose: str = "",
    system: str = "",
    schema: dict | None = None,
    model: str = "haiku",
    tools: str | None = "",
    allowed_tools: str | None = None,
    max_turns: int | None = None,
    max_budget: float | None = None,
    work_dir: str | None = None,
    dangerously_skip_permissions: bool = False,
    timeout: int = 120,
) -> tuple[Any, dict]:
    """Run a single `claude -p` invocation.

    Logs token usage and cost to `cost_log` regardless of success or failure.

    Args:
        prompt: The user prompt (piped via stdin).
        project: Required. Attribution tag for cost_log rows.
        purpose: Optional task label (e.g. "classify", "extract"). Goes to cost_log.
        system: System prompt.
        schema: JSON schema for constrained decoding (`--json-schema`).
        model: `haiku` / `sonnet` / `opus` or a full model id.
        tools: Passed to `--tools`. Default `""` disables tools (single-turn).
            Set to `None` to omit the flag entirely, or `"default"` to enable all.
        allowed_tools: Comma-separated allow-list for `--allowedTools`.
        max_turns: Cap on agentic turns.
        max_budget: `--max-budget-usd` safety cap.
        work_dir: Additional directory to grant file access via `--add-dir`.
        timeout: Subprocess timeout in seconds.

    Returns:
        `(result, metadata)`. `result` is a dict if `schema` was provided,
        otherwise the raw response string. `metadata` contains `cost`, `turns`,
        `duration_s`, `session_id`, and `model` (the first model from modelUsage).

    Raises:
        ValueError: If `project` is empty.
        ClaudeCLIError: If the CLI exits non-zero, times out, or returns
            malformed output, or the response has `is_error`.
    """
    if not project:
        raise ValueError("project is required (used for cost_log attribution)")

    cmd = _build_cmd(
        model=model,
        system=system,
        schema=schema,
        tools=tools,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
        max_budget=max_budget,
        work_dir=work_dir,
        dangerously_skip_permissions=dangerously_skip_permissions,
    )

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_ENV,
        )
    except subprocess.TimeoutExpired as e:
        raise ClaudeCLIError(f"claude timed out after {timeout}s") from e
    elapsed = time.time() - start

    response: dict | None = None
    if proc.stdout:
        try:
            response = json.loads(proc.stdout)
        except json.JSONDecodeError:
            response = None

    if isinstance(response, dict):
        log_cli_usage(
            response,
            project=project,
            purpose=purpose,
            requested_model=model,
        )

    if proc.returncode != 0:
        raise ClaudeCLIError(
            f"claude exited {proc.returncode}: {proc.stderr[:300] or proc.stdout[:300]}"
        )
    if response is None:
        raise ClaudeCLIError(
            f"claude returned unparseable output: {proc.stdout[:300]}"
        )
    if response.get("is_error"):
        raise ClaudeCLIError(
            f"claude error: {(response.get('result') or 'unknown')[:300]}"
        )

    try:
        result = _parse_result(response, schema)
    except json.JSONDecodeError as e:
        raise ClaudeCLIError(
            f"claude returned unparseable JSON result: {(response.get('result') or '')[:200]}"
        ) from e

    if result is None or (isinstance(result, str) and not result):
        raise ClaudeCLIError(
            f"empty response: {(response.get('result') or '')[:200]}"
        )

    model_usage = response.get("modelUsage") or {}
    first_model = next(iter(model_usage), model)
    metadata = {
        "cost": response.get("total_cost_usd", 0.0),
        "turns": response.get("num_turns", 0),
        "duration_s": round(elapsed, 1),
        "session_id": response.get("session_id"),
        "model": first_model,
    }
    return result, metadata


@dataclass
class Task:
    """A single generation task for `generate_parallel`."""

    prompt: str
    purpose: str = ""
    system: str = ""
    schema: dict | None = None
    model: str = "haiku"
    tools: str | None = ""
    allowed_tools: str | None = None
    max_turns: int | None = None
    max_budget: float | None = None
    work_dir: str | None = None
    dangerously_skip_permissions: bool = False
    timeout: int = 120
    tag: str = ""


def generate_parallel(
    tasks: list[Task],
    *,
    project: str,
    max_concurrent: int = 4,
) -> list[tuple[Any, dict]]:
    """Run multiple tasks concurrently via a thread pool.

    Returns `[(result_or_None, metadata), ...]` in input order. On failure,
    `result` is `None` and `metadata` contains an `error` key. Cost is logged
    for each call regardless of success (inherited from `generate`).
    """
    if not project:
        raise ValueError("project is required (used for cost_log attribution)")

    def _run(task: Task) -> tuple[Any, dict]:
        try:
            result, meta = generate(
                prompt=task.prompt,
                project=project,
                purpose=task.purpose,
                system=task.system,
                schema=task.schema,
                model=task.model,
                tools=task.tools,
                allowed_tools=task.allowed_tools,
                max_turns=task.max_turns,
                max_budget=task.max_budget,
                work_dir=task.work_dir,
                dangerously_skip_permissions=task.dangerously_skip_permissions,
                timeout=task.timeout,
            )
            if task.tag:
                meta["tag"] = task.tag
            return result, meta
        except Exception as e:
            meta: dict[str, Any] = {"error": str(e)}
            if task.tag:
                meta["tag"] = task.tag
            return None, meta

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        return list(ex.map(_run, tasks))
