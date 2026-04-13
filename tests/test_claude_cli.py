"""Tests for limbic.cerebellum.claude_cli — `claude -p` wrapper with cost logging."""

from __future__ import annotations

import json
import shutil
import subprocess
from unittest.mock import patch

import pytest

from limbic.cerebellum import claude_cli
from limbic.cerebellum.claude_cli import ClaudeCLIError, Task, generate, generate_parallel
from limbic.cerebellum.cost_log import CostLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cost_log(tmp_path, monkeypatch):
    """Redirect the module-level `cost_log` singleton to a throwaway SQLite DB."""
    db = tmp_path / "costs.db"
    fresh = CostLog(db_path=db)
    monkeypatch.setattr(claude_cli, "cost_log", fresh)
    return fresh


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["claude", "-p"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _single_model_response(
    *,
    result: str = "pong",
    model: str = "claude-haiku-4-5-20251001",
    cost: float = 0.0381,
    input_tokens: int = 10,
    output_tokens: int = 43,
    cache_creation: int = 30337,
    cache_read: int = 0,
    is_error: bool = False,
    structured_output: dict | None = None,
) -> dict:
    response = {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "duration_ms": 2026,
        "duration_api_ms": 1982,
        "num_turns": 1,
        "result": result,
        "session_id": "test-session-single",
        "total_cost_usd": cost,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        },
        "modelUsage": {
            model: {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "cacheCreationInputTokens": cache_creation,
                "cacheReadInputTokens": cache_read,
                "webSearchRequests": 0,
                "costUSD": cost,
            }
        },
    }
    if structured_output is not None:
        response["structured_output"] = structured_output
    return response


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------


class TestGenerateHappyPath:
    def test_plain_text_single_model(self, tmp_cost_log):
        response = _single_model_response(result="pong")
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response))):
            result, meta = generate(prompt="ping", project="testproj", purpose="smoke")

        assert result == "pong"
        assert meta["session_id"] == "test-session-single"
        assert meta["turns"] == 1
        assert meta["model"] == "claude-haiku-4-5-20251001"

        rows = tmp_cost_log.query()
        assert len(rows) == 1
        row = rows[0]
        assert row["project"] == "testproj"
        assert row["model"] == "claude-haiku-4-5-20251001"
        assert row["script"] == "claude-cli"
        assert row["purpose"] == "smoke"
        assert row["prompt_tokens"] == 10
        assert row["completion_tokens"] == 43
        assert row["cached_tokens"] == 0  # cache_read, not cache_creation
        assert row["cost_usd"] == pytest.approx(0.0381)
        metadata = json.loads(row["metadata"])
        assert metadata["session_id"] == "test-session-single"
        assert metadata["headless"] is True
        assert metadata["requested_model"] == "haiku"
        assert "failed" not in metadata

    def test_schema_structured_output(self, tmp_cost_log):
        response = _single_model_response(
            result='{"label": "positive"}',
            structured_output={"label": "positive"},
        )
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response))):
            result, _ = generate(
                prompt="good vibes",
                project="testproj",
                schema={"type": "object", "properties": {"label": {"type": "string"}}},
            )

        assert result == {"label": "positive"}

    def test_schema_fenced_json_fallback(self, tmp_cost_log):
        # No structured_output; result contains fenced JSON (older CLI / no --json-schema).
        response = _single_model_response(
            result="```json\n{\"label\": \"neg\"}\n```",
        )
        response.pop("structured_output", None)
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response))):
            result, _ = generate(
                prompt="bad vibes",
                project="testproj",
                schema={"type": "object"},
            )

        assert result == {"label": "neg"}

    def test_multi_model_session_writes_one_row_per_model(self, tmp_cost_log):
        response = _single_model_response()
        response["modelUsage"] = {
            "claude-haiku-4-5-20251001": {
                "inputTokens": 10, "outputTokens": 20,
                "cacheCreationInputTokens": 0, "cacheReadInputTokens": 0,
                "costUSD": 0.01, "webSearchRequests": 0,
            },
            "claude-sonnet-4-5-20250929": {
                "inputTokens": 50, "outputTokens": 200,
                "cacheCreationInputTokens": 0, "cacheReadInputTokens": 10,
                "costUSD": 0.15, "webSearchRequests": 2,
            },
        }
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response))):
            _, meta = generate(prompt="hi", project="testproj", purpose="escalate")

        rows = tmp_cost_log.query()
        assert len(rows) == 2
        by_model = {r["model"]: r for r in rows}
        assert "claude-haiku-4-5-20251001" in by_model
        assert "claude-sonnet-4-5-20250929" in by_model

        sonnet_row = by_model["claude-sonnet-4-5-20250929"]
        assert sonnet_row["prompt_tokens"] == 50
        assert sonnet_row["completion_tokens"] == 200
        assert sonnet_row["cached_tokens"] == 10
        assert sonnet_row["cost_usd"] == pytest.approx(0.15)

        sonnet_meta = json.loads(sonnet_row["metadata"])
        assert sonnet_meta["web_search_requests"] == 2
        assert sonnet_meta["session_id"] == "test-session-single"

        # Both rows share the same session_id so they can be re-aggregated.
        haiku_meta = json.loads(by_model["claude-haiku-4-5-20251001"]["metadata"])
        assert haiku_meta["session_id"] == sonnet_meta["session_id"]
        assert "web_search_requests" not in haiku_meta  # 0 is omitted

    def test_falls_back_to_top_level_usage_when_modelUsage_missing(self, tmp_cost_log):
        response = _single_model_response()
        response.pop("modelUsage")
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response))):
            generate(prompt="hi", project="testproj", model="sonnet")

        rows = tmp_cost_log.query()
        assert len(rows) == 1
        # Falls back to the user-requested model name since modelUsage is absent.
        assert rows[0]["model"] == "sonnet"
        assert rows[0]["prompt_tokens"] == 10
        assert rows[0]["completion_tokens"] == 43


# ---------------------------------------------------------------------------
# generate() — errors
# ---------------------------------------------------------------------------


class TestGenerateErrors:
    def test_project_required(self, tmp_cost_log):
        with pytest.raises(ValueError, match="project is required"):
            generate(prompt="hi", project="")

    def test_nonzero_exit_raises_but_still_logs_usage(self, tmp_cost_log):
        # CLI exits non-zero but stdout contains a parseable JSON result block.
        response = _single_model_response(result="partial", is_error=True)
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response), returncode=1, stderr="oops")):
            with pytest.raises(ClaudeCLIError, match="exited 1"):
                generate(prompt="hi", project="testproj")

        rows = tmp_cost_log.query()
        assert len(rows) == 1
        metadata = json.loads(rows[0]["metadata"])
        assert metadata.get("failed") is True

    def test_is_error_flag_raises_but_logs_usage(self, tmp_cost_log):
        response = _single_model_response(result="error body", is_error=True)
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response))):
            with pytest.raises(ClaudeCLIError, match="claude error"):
                generate(prompt="hi", project="testproj")

        rows = tmp_cost_log.query()
        assert len(rows) == 1
        assert json.loads(rows[0]["metadata"])["failed"] is True

    def test_unparseable_output_raises_no_log(self, tmp_cost_log):
        with patch.object(subprocess, "run", return_value=_completed("not json at all")):
            with pytest.raises(ClaudeCLIError, match="unparseable output"):
                generate(prompt="hi", project="testproj")

        rows = tmp_cost_log.query()
        assert rows == []

    def test_timeout_raises(self, tmp_cost_log):
        def _boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=["claude"], timeout=1)

        with patch.object(subprocess, "run", side_effect=_boom):
            with pytest.raises(ClaudeCLIError, match="timed out"):
                generate(prompt="hi", project="testproj", timeout=1)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


class TestCommandBuilding:
    def test_default_flags(self, tmp_cost_log):
        response = _single_model_response()
        captured = {}

        def _fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["env"] = kw.get("env", {})
            captured["input"] = kw.get("input")
            return _completed(json.dumps(response))

        with patch.object(subprocess, "run", side_effect=_fake_run):
            generate(prompt="ping", project="testproj")

        cmd = captured["cmd"]
        assert cmd[:2] == ["claude", "-p"]
        assert "--output-format" in cmd and "json" in cmd
        assert "--no-session-persistence" in cmd
        assert "--model" in cmd and "haiku" in cmd
        # Default tools="" -> empty tools flag present
        assert "--tools" in cmd
        tools_idx = cmd.index("--tools")
        assert cmd[tools_idx + 1] == ""
        # CLAUDECODE stripped
        assert "CLAUDECODE" not in captured["env"]
        # Prompt piped as stdin
        assert captured["input"] == "ping"

    def test_tools_none_omits_flag(self, tmp_cost_log):
        response = _single_model_response()
        captured = {}

        def _fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return _completed(json.dumps(response))

        with patch.object(subprocess, "run", side_effect=_fake_run):
            generate(prompt="ping", project="testproj", tools=None)

        assert "--tools" not in captured["cmd"]

    def test_schema_becomes_json_schema_flag(self, tmp_cost_log):
        response = _single_model_response(structured_output={"ok": True})
        captured = {}

        def _fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return _completed(json.dumps(response))

        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        with patch.object(subprocess, "run", side_effect=_fake_run):
            generate(prompt="hi", project="testproj", schema=schema)

        cmd = captured["cmd"]
        assert "--json-schema" in cmd
        assert json.loads(cmd[cmd.index("--json-schema") + 1]) == schema

    def test_all_optional_flags_forwarded(self, tmp_cost_log):
        response = _single_model_response()
        captured = {}

        def _fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return _completed(json.dumps(response))

        with patch.object(subprocess, "run", side_effect=_fake_run):
            generate(
                prompt="hi",
                project="testproj",
                system="You are terse.",
                allowed_tools="Read,Grep",
                max_turns=3,
                max_budget=0.5,
                work_dir="/tmp/wd",
                dangerously_skip_permissions=True,
                model="sonnet",
            )

        cmd = captured["cmd"]
        assert "--system-prompt" in cmd and "You are terse." in cmd
        assert "--allowedTools" in cmd and "Read,Grep" in cmd
        assert "--max-turns" in cmd and "3" in cmd
        assert "--max-budget-usd" in cmd and "0.5" in cmd
        assert "--add-dir" in cmd and "/tmp/wd" in cmd
        assert "--model" in cmd and "sonnet" in cmd
        assert "--dangerously-skip-permissions" in cmd

    def test_dangerously_skip_permissions_default_off(self, tmp_cost_log):
        response = _single_model_response()
        captured = {}

        def _fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return _completed(json.dumps(response))

        with patch.object(subprocess, "run", side_effect=_fake_run):
            generate(prompt="hi", project="testproj")

        assert "--dangerously-skip-permissions" not in captured["cmd"]


# ---------------------------------------------------------------------------
# generate_parallel()
# ---------------------------------------------------------------------------


class TestGenerateParallel:
    def test_returns_results_in_input_order(self, tmp_cost_log):
        responses = [
            _single_model_response(result="one"),
            _single_model_response(result="two"),
            _single_model_response(result="three"),
        ]
        call_count = {"n": 0}

        def _fake_run(*a, **kw):
            i = call_count["n"]
            call_count["n"] += 1
            return _completed(json.dumps(responses[i]))

        with patch.object(subprocess, "run", side_effect=_fake_run):
            results = generate_parallel(
                [Task(prompt="a"), Task(prompt="b"), Task(prompt="c")],
                project="testproj",
                max_concurrent=1,  # serialize so the fake counter is deterministic
            )

        assert [r[0] for r in results] == ["one", "two", "three"]
        rows = tmp_cost_log.query()
        assert len(rows) == 3

    def test_failure_returns_none_but_does_not_raise(self, tmp_cost_log):
        good = _single_model_response(result="good")

        call_count = {"n": 0}

        def _fake_run(*a, **kw):
            i = call_count["n"]
            call_count["n"] += 1
            if i == 1:
                return _completed("not json", returncode=1, stderr="boom")
            return _completed(json.dumps(good))

        with patch.object(subprocess, "run", side_effect=_fake_run):
            results = generate_parallel(
                [Task(prompt="a"), Task(prompt="b", tag="bad"), Task(prompt="c")],
                project="testproj",
                max_concurrent=1,
            )

        assert results[0][0] == "good"
        assert results[1][0] is None
        assert "error" in results[1][1]
        assert results[1][1]["tag"] == "bad"
        assert results[2][0] == "good"


# ---------------------------------------------------------------------------
# Live smoke test (opt-in, requires claude CLI)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")
def test_live_smoke_single_call(tmp_cost_log):
    """Hit the real CLI end-to-end. Skipped if `claude` is not on PATH."""
    result, meta = generate(
        prompt="Reply with exactly one word: ping",
        project="limbic-test",
        purpose="live-smoke",
        model="haiku",
        timeout=60,
    )
    assert isinstance(result, str) and result.strip()
    assert meta["session_id"]
    rows = tmp_cost_log.query()
    assert len(rows) >= 1
    assert rows[0]["project"] == "limbic-test"
    assert rows[0]["script"] == "claude-cli"
