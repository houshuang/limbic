"""Tests for limbic.cerebellum.claude_cli — `claude -p` wrapper with cost logging."""

from __future__ import annotations

import json
import base64
import os
import shutil
import subprocess
from pathlib import Path
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
        # `call_id` lets a composing caller (cached_call) recover the row this
        # call already wrote instead of logging a second, cost-doubling row.
        assert meta["call_id"] == row["id"]
        assert row["project"] == "testproj"
        assert row["model"] == "claude-haiku-4-5-20251001"
        assert row["script"] == "claude-cli"
        assert row["purpose"] == "smoke"
        assert row["prompt_tokens"] == 10
        assert row["completion_tokens"] == 43
        assert row["cached_tokens"] == 0  # cache_read, not cache_creation
        assert row["cost_usd"] == 0
        assert row["billing_mode"] == "subscription"
        assert row["notional_cost_usd"] == pytest.approx(0.0381)
        metadata = json.loads(row["metadata"])
        assert metadata["session_id"] == "test-session-single"
        assert metadata["headless"] is True
        assert metadata["requested_model"] == "haiku"
        assert metadata["prompt_chars"] == 4
        assert metadata["prompt_sha256"]
        assert metadata["system_chars"] == 0
        assert metadata["schema_chars"] == 0
        assert metadata["minimal_headless"] is True
        assert metadata["cache_creation_tokens"] == 30337
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
        assert sonnet_row["cost_usd"] == 0
        assert sonnet_row["billing_mode"] == "subscription"
        assert sonnet_row["notional_cost_usd"] == pytest.approx(0.15)

        sonnet_meta = json.loads(sonnet_row["metadata"])
        assert sonnet_meta["web_search_requests"] == 2
        assert sonnet_meta["session_id"] == "test-session-single"

        # Both rows share the same session_id so they can be re-aggregated.
        haiku_meta = json.loads(by_model["claude-haiku-4-5-20251001"]["metadata"])
        assert haiku_meta["session_id"] == sonnet_meta["session_id"]
        assert "web_search_requests" not in haiku_meta  # 0 is omitted

        # call_id points at the first model's row (insertion order: haiku first).
        assert meta["call_id"] == by_model["claude-haiku-4-5-20251001"]["id"]

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

    def test_unparseable_output_raises_and_logs_failure_row(self, tmp_cost_log):
        with patch.object(subprocess, "run", return_value=_completed("not json at all")):
            with pytest.raises(ClaudeCLIError, match="unparseable output"):
                generate(prompt="hi", project="testproj")

        rows = tmp_cost_log.query()
        assert len(rows) == 1
        assert rows[0]["cost_usd"] == 0
        metadata = json.loads(rows[0]["metadata"])
        assert metadata["failed"] is True
        assert metadata["error_type"] == "unparseable_output"
        assert metadata["prompt_chars"] == 2

    def test_timeout_raises(self, tmp_cost_log):
        def _boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=["claude"], timeout=1)

        with patch.object(subprocess, "run", side_effect=_boom):
            with pytest.raises(ClaudeCLIError, match="timed out"):
                generate(prompt="hi", project="testproj", timeout=1)

        rows = tmp_cost_log.query()
        assert len(rows) == 1
        metadata = json.loads(rows[0]["metadata"])
        assert metadata["failed"] is True
        assert metadata["error_type"] == "timeout"


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
        assert "--strict-mcp-config" in cmd
        assert "--mcp-config" in cmd
        assert json.loads(cmd[cmd.index("--mcp-config") + 1]) == {"mcpServers": {}}
        assert "--disable-slash-commands" in cmd
        assert "--no-chrome" in cmd
        assert "--setting-sources" in cmd
        assert cmd[cmd.index("--setting-sources") + 1] == "local"
        # Default tools="" -> empty tools flag present
        assert "--tools" in cmd
        tools_idx = cmd.index("--tools")
        assert cmd[tools_idx + 1] == ""
        # CLAUDECODE + ANTHROPIC_* stripped (prefer Max/OAuth auth, not API-key billing)
        assert "CLAUDECODE" not in captured["env"]
        assert "ANTHROPIC_API_KEY" not in captured["env"]
        assert "ANTHROPIC_KEY" not in captured["env"]
        assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]
        # Prompt piped as stdin
        assert captured["input"] == "ping"

    def test_anthropic_env_vars_stripped(self, tmp_cost_log, monkeypatch):
        """When the caller has ANTHROPIC_* vars set, the CLI must not see them."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
        monkeypatch.setenv("ANTHROPIC_KEY", "legacy-should-not-leak")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oauth-should-not-leak")
        # No reload: the environment is read per call now. It also used to undo
        # the tmp_cost_log fixture, so this test wrote to the real cost DB.
        from limbic.cerebellum import claude_cli as cc

        response = _single_model_response()
        captured = {}

        def _fake_run(cmd, **kw):
            captured["env"] = kw.get("env", {})
            return _completed(json.dumps(response))

        with patch.object(subprocess, "run", side_effect=_fake_run):
            cc.generate(prompt="ping", project="testproj")

        assert "ANTHROPIC_API_KEY" not in captured["env"]
        assert "ANTHROPIC_KEY" not in captured["env"]
        assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]
        # And the caller's parent env was not mutated (still has the vars)
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-should-not-leak"

    def test_tools_none_omits_flag(self, tmp_cost_log):
        response = _single_model_response()
        captured = {}

        def _fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return _completed(json.dumps(response))

        with patch.object(subprocess, "run", side_effect=_fake_run):
            generate(prompt="ping", project="testproj", tools=None)

        assert "--tools" not in captured["cmd"]

    def test_minimal_headless_can_be_disabled(self, tmp_cost_log):
        response = _single_model_response()
        captured = {}

        def _fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return _completed(json.dumps(response))

        with patch.object(subprocess, "run", side_effect=_fake_run):
            generate(prompt="ping", project="testproj", minimal_headless=False)

        cmd = captured["cmd"]
        assert "--strict-mcp-config" not in cmd
        assert "--mcp-config" not in cmd
        assert "--disable-slash-commands" not in cmd
        assert "--setting-sources" not in cmd

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


@pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("LIMBIC_LIVE_CLAUDE") != "1",
    reason="live Claude CLI smoke test is opt-in with LIMBIC_LIVE_CLAUDE=1",
)
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


class TestImageSubscriptionTransport:
    @pytest.fixture
    def image(self, tmp_path):
        image = tmp_path / "scan.png"
        image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZskAAAAASUVORK5CYII="))
        return image

    def test_images_use_stream_both_ways_with_no_tools_and_final_result(self, image, tmp_cost_log):
        response = _single_model_response(structured_output={"answer": 7}, cache_read=5, cache_creation=3)
        events = [{"type": "system", "subtype": "init"}, {"type": "assistant", "message": {"content": []}}, response]
        stdout = "\n".join(json.dumps(event) for event in events) + "\n"
        meta_out = {}
        with patch.object(subprocess, "run", return_value=_completed(stdout)) as run:
            result, meta = generate("Read this scan", project="testproj", model="sonnet", effort="low",
                                    schema={"type": "object"}, images=[str(image)], meta_out=meta_out)
        assert result == {"answer": 7}
        cmd = run.call_args.args[0]
        assert cmd[cmd.index("--input-format") + 1] == "stream-json"
        assert cmd[cmd.index("--output-format") + 1] == "stream-json"
        assert cmd[cmd.index("--tools") + 1] == ""
        assert cmd[cmd.index("--effort") + 1] == "low"
        assert "--verbose" in cmd
        assert "--reasoning" not in cmd
        message = json.loads(run.call_args.kwargs["input"])
        assert message["type"] == "user"
        assert message["message"]["role"] == "user"
        assert message["parent_tool_use_id"] is None
        content = message["message"]["content"]
        assert len(content) == 2
        assert content[0]["source"]["media_type"] == "image/png"
        assert base64.b64decode(content[0]["source"]["data"]) == image.read_bytes()
        assert content[1] == {"type": "text", "text": "Read this scan"}
        assert meta == meta_out
        assert meta["model"] == "claude-haiku-4-5-20251001"
        assert meta["input_tokens"] == 18
        assert meta["cached_tokens"] == 5
        assert meta["cache_write_input_tokens"] == 3
        assert meta["output_tokens"] == 43
        assert meta["cost_usd"] == 0
        assert meta["billing_mode"] == "subscription"
        assert meta["notional_cost_usd"] == pytest.approx(0.0381)
        row, = tmp_cost_log.query()
        assert row["cost_usd"] == 0
        assert row["billing_mode"] == "subscription"
        assert row["notional_cost_usd"] == pytest.approx(0.0381)
        assert json.loads(row["metadata"])["image_count"] == 1

    @pytest.mark.parametrize("images", [None, []])
    def test_text_only_preserves_plain_stdin_and_json_output(self, tmp_cost_log, images):
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(_single_model_response()))) as run:
            generate("literal text\nsecond line", project="testproj", images=images)
        cmd = run.call_args.args[0]
        assert run.call_args.kwargs["input"] == "literal text\nsecond line"
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert "--input-format" not in cmd
        assert "--effort" not in cmd

    @pytest.mark.parametrize("bad", ["missing.png", "directory", "fake.png", "document.pdf"])
    def test_invalid_image_rejected_before_spawn_or_logging(self, tmp_path, tmp_cost_log, bad):
        path = tmp_path / bad
        if bad == "directory":
            path.mkdir()
        elif bad != "missing.png":
            path.write_bytes(b"%PDF-1.7 not an image")
        with patch.object(subprocess, "run") as run, pytest.raises(ValueError):
            generate("read", project="testproj", images=[str(path)])
        run.assert_not_called()
        assert tmp_cost_log.query() == []

    def test_nonzero_response_preserves_usage_tails_and_one_failed_subscription_row(self, tmp_cost_log):
        response = _single_model_response(is_error=True, cache_read=5, cache_creation=3)
        stdout = json.dumps(response)
        stderr = "x" * 5000 + "failure end"
        meta_out = {}
        with patch.object(subprocess, "run", return_value=_completed(stdout, returncode=1, stderr=stderr)):
            with pytest.raises(ClaudeCLIError) as caught:
                generate("read", project="testproj", meta_out=meta_out)
        assert caught.value.stdout == stdout[-4000:]
        assert caught.value.stderr == stderr[-4000:]
        assert caught.value.metadata == meta_out
        assert meta_out["input_tokens"] == 18
        assert meta_out["output_tokens"] == 43
        assert meta_out["cached_tokens"] == 5
        assert meta_out["error_type"] == "nonzero_exit"
        row, = tmp_cost_log.query()
        assert row["cost_usd"] == 0
        assert row["billing_mode"] == "subscription"
        assert row["notional_cost_usd"] == pytest.approx(0.0381)
        assert json.loads(row["metadata"])["failed"]

    def test_timeout_preserves_complete_partial_response_usage_and_byte_tails(self, tmp_cost_log):
        response = _single_model_response(result="x" * 6000, cache_creation=3)
        stdout = json.dumps(response).encode()
        stderr = b"y" * 5000
        failure = subprocess.TimeoutExpired("claude", 1, output=stdout, stderr=stderr)
        with patch.object(subprocess, "run", side_effect=failure), pytest.raises(ClaudeCLIError) as caught:
            generate("read", project="testproj", timeout=1)
        assert caught.value.stdout == stdout.decode()[-4000:]
        assert caught.value.stderr == "y" * 4000
        assert caught.value.metadata["input_tokens"] == 13
        assert caught.value.metadata["output_tokens"] == 43
        assert len(tmp_cost_log.query()) == 1

    @pytest.mark.parametrize("stdout,returncode", [("not json", 0), ("not json", 1), ("[]", 0)])
    def test_malformed_outer_response_logs_once_with_diagnostics(self, tmp_cost_log, stdout, returncode):
        with patch.object(subprocess, "run", return_value=_completed(stdout, returncode, "detail")):
            with pytest.raises(ClaudeCLIError) as caught:
                generate("read", project="testproj")
        assert caught.value.stdout == stdout
        assert caught.value.stderr == "detail"
        assert caught.value.metadata["total_tokens"] == 0
        row, = tmp_cost_log.query()
        assert row["billing_mode"] == "subscription"
        assert row["cost_usd"] == 0
        assert row["notional_cost_usd"] is None

    def test_model_usage_fallback_normalizes_cache_read_and_write_tokens(self, tmp_cost_log):
        response = _single_model_response(cache_read=5, cache_creation=3)
        del response["usage"]
        del response["total_cost_usd"]
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response))):
            _, meta = generate("read", project="testproj")
        assert meta["input_tokens"] == 18
        assert meta["output_tokens"] == 43
        assert meta["cache_write_input_tokens"] == 3
        assert meta["notional_cost_usd"] == pytest.approx(0.0381)

    def test_top_level_usage_subscription_accounting(self, tmp_cost_log):
        response = _single_model_response(cache_read=5, cache_creation=3)
        del response["modelUsage"]
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response))):
            _, meta = generate("read", project="testproj", model="sonnet")
        row, = tmp_cost_log.query()
        assert row["cost_usd"] == 0
        assert row["notional_cost_usd"] == pytest.approx(0.0381)
        assert row["billing_mode"] == "subscription"
        assert meta["model"] == "sonnet"

    def test_effort_validation_precedes_spawn(self, tmp_cost_log):
        with patch.object(subprocess, "run") as run, pytest.raises(ValueError, match="effort"):
            generate("read", project="testproj", effort="unsupported")
        run.assert_not_called()

    def test_invalid_structured_result_preserves_usage_and_marks_one_row_failed(self, tmp_cost_log):
        response = _single_model_response(result="{broken", cache_creation=3)
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(response))):
            with pytest.raises(ClaudeCLIError) as caught:
                generate("read", project="testproj", schema={"type": "object"})
        assert caught.value.metadata["input_tokens"] == 13
        assert caught.value.metadata["output_tokens"] == 43
        assert caught.value.metadata["error_type"] == "unparseable_result"
        row, = tmp_cost_log.query()
        assert json.loads(row["metadata"])["failed"]

    def test_image_stream_without_result_event_is_not_silently_successful(self, image, tmp_cost_log):
        stdout = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "partial"}]}})
        with patch.object(subprocess, "run", return_value=_completed(stdout)), pytest.raises(ClaudeCLIError):
            generate("read", project="testproj", images=[str(image)])
        row, = tmp_cost_log.query()
        assert json.loads(row["metadata"])["failed"]

    @pytest.mark.parametrize("data,mime", [(b"\xff\xd8\xffjpeg-data", "image/jpeg"),
                                            (b"GIF89aimage-data", "image/gif"),
                                            (b"RIFF\x00\x00\x00\x00WEBPimage-data", "image/webp")])
    def test_image_mime_comes_from_bytes_not_filename(self, tmp_path, tmp_cost_log, data, mime):
        image = tmp_path / "unhelpful-name.bin"
        image.write_bytes(data)
        stdout = json.dumps(_single_model_response())
        with patch.object(subprocess, "run", return_value=_completed(stdout)) as run:
            generate("read", project="testproj", images=[str(image)])
        block = json.loads(run.call_args.kwargs["input"])["message"]["content"][0]
        assert block["source"]["media_type"] == mime
        assert base64.b64decode(block["source"]["data"]) == data


class TestStatelessWorkingDirectory:
    def test_stateless_call_uses_fresh_empty_directory_then_cleans_it(self, tmp_cost_log, tmp_path):
        observed = []
        parent_cwd = Path.cwd()
        work_dir = str(tmp_path)
        def fake_run(cmd, **kwargs):
            cwd = Path(kwargs["cwd"])
            assert cwd.is_dir()
            assert list(cwd.iterdir()) == []
            assert Path.cwd() == parent_cwd
            assert cmd[cmd.index("--add-dir") + 1] == work_dir
            observed.append(cwd)
            return _completed(json.dumps(_single_model_response()))
        with patch.object(subprocess, "run", side_effect=fake_run):
            generate("first", project="testproj", stateless=True, work_dir=work_dir)
            generate("second", project="testproj", stateless=True, work_dir=work_dir)
        assert len(observed) == 2
        assert observed[0] != observed[1]
        assert all(not cwd.exists() for cwd in observed)
        assert Path.cwd() == parent_cwd
        assert all(json.loads(row["metadata"])["stateless"] for row in tmp_cost_log.query())

    def test_stateless_directory_is_cleaned_on_timeout(self, tmp_cost_log):
        observed = []
        def fake_run(cmd, **kwargs):
            cwd = Path(kwargs["cwd"])
            assert cwd.is_dir()
            observed.append(cwd)
            raise subprocess.TimeoutExpired(cmd, 1)
        with patch.object(subprocess, "run", side_effect=fake_run), pytest.raises(ClaudeCLIError):
            generate("read", project="testproj", stateless=True)
        assert not observed[0].exists()

    def test_legacy_call_does_not_override_subprocess_cwd(self, tmp_cost_log):
        with patch.object(subprocess, "run", return_value=_completed(json.dumps(_single_model_response()))) as run:
            generate("read", project="testproj")
        assert "cwd" not in run.call_args.kwargs
