"""Tests for limbic.cerebellum.codex_cli — command shaping and subprocess handling.

Retry/quota behaviour lives in tests/test_cerebellum.py::TestCodexCLIRetry.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

from limbic.cerebellum import codex_cli as cc


# ---------------------------------------------------------------------------
# Command shaping
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_cmd(monkeypatch):
    """Run codex_json/codex_research against a stub _run and return the argv."""
    seen: list[list[str]] = []

    def _run(cmd, timeout):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(cc, "_run", _run)
    return seen


class TestIsolationFlags:
    """Both entry points get --ephemeral/--ignore-user-config by default.

    codex_research is the call that reads untrusted web pages with network
    egress, so omitting them there was the worse of the two defaults. hvaskjer
    was rebinding codex_cli._run at runtime to inject them.
    """

    def test_codex_research_is_isolated_by_default(self, captured_cmd):
        cc.codex_research("mission")
        cmd = captured_cmd[0]
        assert "--ephemeral" in cmd
        assert "--ignore-user-config" in cmd

    def test_codex_json_is_isolated(self, captured_cmd):
        cc.codex_json("prompt")
        cmd = captured_cmd[0]
        assert "--ephemeral" in cmd
        assert "--ignore-user-config" in cmd

    def test_isolated_false_restores_host_profile(self, captured_cmd):
        cc.codex_research("mission", isolated=False)
        cmd = captured_cmd[0]
        assert "--ephemeral" not in cmd
        assert "--ignore-user-config" not in cmd

    def test_isolation_does_not_disturb_agentic_flags(self, captured_cmd):
        """The flags that make the run agentic must survive the insertion."""
        cc.codex_research("mission")
        cmd = captured_cmd[0]
        assert "tools.web_search=true" in cmd
        assert "sandbox_workspace_write.network_access=true" in cmd
        assert cmd[:2] == ["codex", "exec"]
        assert cmd[-1] == "mission"


# ---------------------------------------------------------------------------
# Bounded output
# ---------------------------------------------------------------------------


class TestTail:
    def test_keeps_the_end_not_the_start(self):
        tail = cc._Tail(10)
        tail.add("abcdefghijklmnop")
        assert tail.get().endswith("ghijklmnop")

    def test_reports_how_much_it_dropped(self):
        tail = cc._Tail(10)
        tail.add("x" * 25)
        assert "15 earlier characters discarded" in tail.get()

    def test_under_limit_is_verbatim(self):
        tail = cc._Tail(100)
        tail.add("short")
        assert tail.get() == "short"


# ---------------------------------------------------------------------------
# Subprocess handling (real child processes)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    """Put an executable named `codex` on PATH that runs a given python snippet."""

    def install(body: str) -> None:
        script = tmp_path / "codex"
        script.write_text(
            f"#!{sys.executable}\n" + textwrap.dedent(body), encoding="utf-8"
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setattr(cc, "_DISABLED_UNTIL", 0.0)

    return install


@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX-only")
class TestRunSubprocess:
    def test_output_is_capped(self, fake_codex, monkeypatch):
        monkeypatch.setattr(cc, "OUTPUT_LIMIT", 1000)
        fake_codex("""
            import sys
            sys.stdout.write("y" * 50000)
        """)
        proc = cc._run(["codex"], timeout=30)
        assert proc.returncode == 0
        # The cap plus the one-line "discarded" preamble.
        assert len(proc.stdout) < 1100
        assert "earlier characters discarded" in proc.stdout

    def test_large_output_does_not_deadlock(self, fake_codex):
        """A PIPE without concurrent draining blocks the child at ~64KB."""
        fake_codex("""
            import sys
            sys.stdout.write("a" * 300000)
            sys.stderr.write("b" * 300000)
        """)
        proc = cc._run(["codex"], timeout=30)
        assert proc.returncode == 0
        assert proc.stdout.endswith("a")
        assert proc.stderr.endswith("b")

    def test_timeout_raises(self, fake_codex):
        fake_codex("""
            import time
            time.sleep(30)
        """)
        with pytest.raises(cc.CodexCLIError, match="timed out"):
            cc._run(["codex"], timeout=1)

    def test_timeout_kills_the_whole_process_tree(self, fake_codex, tmp_path):
        """subprocess.run(timeout=) kills only the direct child; codex spawns helpers.

        The child forks a grandchild that would outlive it and keep writing.
        """
        marker = tmp_path / "grandchild.txt"
        fake_codex(f"""
            import os, sys, time
            if os.fork() == 0:
                for _ in range(200):
                    with open({str(marker)!r}, "a") as fh:
                        fh.write("tick\\n")
                        fh.flush()
                    time.sleep(0.1)
                os._exit(0)
            time.sleep(30)
        """)
        with pytest.raises(cc.CodexCLIError, match="timed out"):
            cc._run(["codex"], timeout=1)
        settled = marker.read_text() if marker.exists() else ""
        time.sleep(0.5)
        after = marker.read_text() if marker.exists() else ""
        assert after == settled, "grandchild survived the timeout and kept writing"

    def test_nonzero_exit_is_returned_not_raised(self, fake_codex):
        """_run reports; _finish is what turns a bad exit into CodexCLIError."""
        fake_codex("""
            import sys
            sys.stderr.write("boom")
            sys.exit(3)
        """)
        proc = cc._run(["codex"], timeout=30)
        assert proc.returncode == 3
        assert "boom" in proc.stderr

    def test_undecodable_byte_does_not_stall_the_child(self, fake_codex):
        """A dead drain thread stalls the child on a full pipe, and the caller
        sees a timeout rather than the decode problem — costing the full 900s on
        codex_research and pointing the operator the wrong way."""
        fake_codex("""
            import sys
            sys.stdout.buffer.write(b'\\xff' + b'a' * 200000)
            sys.stdout.buffer.flush()
        """)
        proc = cc._run(["codex"], timeout=10)
        assert proc.returncode == 0
        assert len(proc.stdout) > 100_000

    def test_timeout_is_bounded_when_a_descendant_escapes_the_group(
        self, fake_codex, tmp_path
    ):
        """A grandchild with its own session survives killpg and keeps the pipe
        open. close() then blocks on the buffer lock its parked reader holds, so
        _run overran the timeout it had just enforced."""
        fake_codex("""
            import os, sys, time
            if os.fork() == 0:
                os.setsid()          # leave the process group, keep the pipe
                time.sleep(6)
                os._exit(0)
            sys.exit(0)
        """)
        start = time.monotonic()
        try:
            cc._run(["codex"], timeout=1)
        except cc.CodexCLIError:
            pass
        assert time.monotonic() - start < 3.0

    def test_missing_binary_raises(self, monkeypatch):
        monkeypatch.setattr(cc.shutil, "which", lambda name: None)
        with pytest.raises(cc.CodexCLIError, match="not available"):
            cc._run(["codex"], timeout=5)


# ---------------------------------------------------------------------------
# Usage capture → cost ledger
# ---------------------------------------------------------------------------

# Recorded verbatim from codex-cli 0.153.4 on 2026-09-21 (`codex exec --json
# --model gpt-5.5 --sandbox read-only ... 'Reply with exactly: ok'`), with the
# thread id replaced. This is the contract everything below is testing against.
REAL_EVENTS = "\n".join([
    '{"type":"thread.started","thread_id":"00000000-0000-0000-0000-000000000000"}',
    '{"type":"turn.started"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"ok"}}',
    '{"type":"turn.completed","usage":{"input_tokens":14169,"cached_input_tokens":4480,'
    '"cache_write_input_tokens":0,"output_tokens":5,"reasoning_output_tokens":0}}',
])


@pytest.fixture(autouse=True)
def reset_json_support(monkeypatch):
    """`--json` support is a process-wide latch; don't let one test set it for the next."""
    monkeypatch.setattr(cc, "_JSON_EVENTS_SUPPORTED", True)


@pytest.fixture
def ledger():
    from limbic.cerebellum.cost_log import cost_log
    return cost_log


def _stub_run(monkeypatch, stdout, *, returncode=0, stderr=""):
    def _run(cmd, timeout):
        return subprocess.CompletedProcess(args=cmd, returncode=returncode,
                                           stdout=stdout, stderr=stderr)
    monkeypatch.setattr(cc, "_run", _run)


class TestParseUsage:
    def test_real_recorded_shape(self):
        usage = cc.parse_usage(REAL_EVENTS)
        assert usage.found
        assert usage.input_tokens == 14169
        assert usage.cached_input_tokens == 4480
        assert usage.output_tokens == 5
        assert usage.turns == 1
        assert usage.thread_id == "00000000-0000-0000-0000-000000000000"

    def test_per_turn_blocks_are_summed(self):
        stream = REAL_EVENTS + "\n" + (
            '{"type":"turn.completed","usage":{"input_tokens":100,'
            '"cached_input_tokens":10,"output_tokens":7,"reasoning_output_tokens":3}}')
        usage = cc.parse_usage(stream)
        assert (usage.input_tokens, usage.output_tokens, usage.turns) == (14269, 12, 2)
        assert usage.reasoning_output_tokens == 3

    def test_cumulative_old_schema_replaces_rather_than_adds(self):
        """`total_token_usage` is a running total; summing it counts a long run
        once per event."""
        stream = "\n".join([
            '{"id":"0","msg":{"type":"token_count","info":{"total_token_usage":'
            '{"input_tokens":100,"cached_input_tokens":0,"output_tokens":10}}}}',
            '{"id":"1","msg":{"type":"token_count","info":{"total_token_usage":'
            '{"input_tokens":250,"cached_input_tokens":40,"output_tokens":30}}}}',
        ])
        usage = cc.parse_usage(stream)
        assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (250, 40, 30)
        assert usage.found

    def test_plain_text_output_reports_nothing_found(self):
        usage = cc.parse_usage("ok\nnot json at all\n")
        assert not usage.found
        assert usage.input_tokens == 0

    def test_unparseable_lines_are_skipped(self):
        usage = cc.parse_usage("{not json\n" + REAL_EVENTS + "\ntrailing noise")
        assert usage.input_tokens == 14169


class TestFinalMessage:
    def test_recovers_the_agent_message(self):
        assert cc.final_message(REAL_EVENTS) == "ok"

    def test_last_message_wins(self):
        stream = REAL_EVENTS + "\n" + (
            '{"type":"item.completed","item":{"id":"item_1","type":"agent_message",'
            '"text":"final answer"}}')
        assert cc.final_message(stream) == "final answer"

    def test_no_events_is_empty_so_the_caller_can_fall_back(self):
        assert cc.final_message("plain text answer") == ""

    def test_finish_prefers_the_event_message_over_raw_jsonl(self):
        proc = subprocess.CompletedProcess(args=["codex", "--json"], returncode=0,
                                           stdout=REAL_EVENTS, stderr="")
        assert cc._finish(proc, None, None) == "ok"

    def test_finish_still_falls_back_to_raw_stdout_without_json(self):
        proc = subprocess.CompletedProcess(args=["codex"], returncode=0,
                                           stdout="  plain answer  ", stderr="")
        assert cc._finish(proc, None, None) == "plain answer"


class TestUnrecognisedEventShape:
    """An older CLI (production runs 0.146.0) may emit events we cannot read.

    Raw JSONL must never become the answer. A caller that stored a transcript
    as its result would have no way to notice: nothing raises, nothing is
    empty, and the value is plausible JSON.
    """

    # Deliberately not a shape `final_message` knows.
    UNKNOWN_EVENTS = "\n".join([
        '{"id":"0","msg":{"type":"agent_reasoning_delta","delta":"thinking"}}',
        '{"id":"1","msg":{"type":"task_complete","last_message":"ok"}}',
    ])

    def test_schemaless_caller_gets_an_empty_result_not_the_transcript(self):
        proc = subprocess.CompletedProcess(args=["codex", "--json"], returncode=0,
                                           stdout=self.UNKNOWN_EVENTS, stderr="")
        assert cc._finish(proc, None, None) == ""

    def test_a_schema_caller_fails_exactly_as_an_empty_result_always_did(self):
        proc = subprocess.CompletedProcess(args=["codex", "--json"], returncode=0,
                                           stdout=self.UNKNOWN_EVENTS, stderr="")
        with pytest.raises(cc.CodexCLIError, match="unparseable JSON"):
            cc._finish(proc, None, {"type": "object"})

    def test_the_last_message_file_is_still_what_wins(self, tmp_path):
        """The normal path is unaffected: `--output-last-message` is written
        byte-identically with and without `--json`."""
        out = tmp_path / "last.txt"
        out.write_text("the real answer", encoding="utf-8")
        proc = subprocess.CompletedProcess(args=["codex", "--json"], returncode=0,
                                           stdout=self.UNKNOWN_EVENTS, stderr="")
        assert cc._finish(proc, str(out), None) == "the real answer"


class TestUsageLogging:
    def test_success_writes_one_subscription_row(self, monkeypatch, ledger):
        _stub_run(monkeypatch, REAL_EVENTS)
        assert cc.codex_json("hi", project="demo", purpose="classify") == "ok"

        rows = ledger.query()
        assert len(rows) == 1
        row = rows[0]
        assert row["billing_mode"] == "subscription"
        assert row["cost_usd"] == 0.0          # no money moved
        assert row["notional_cost_usd"] > 0    # but the tokens were worth something
        assert (row["prompt_tokens"], row["cached_tokens"], row["completion_tokens"]) == (14169, 4480, 5)
        assert row["project"] == "demo"
        assert row["purpose"] == "classify"
        assert row["script"] == "codex_cli.codex_json"
        assert json.loads(row["metadata"])["transport"] == "codex_cli"

    def test_notional_matches_the_api_price_of_the_same_tokens(self, monkeypatch, ledger):
        from limbic.cerebellum.cost_log import cost_for
        _stub_run(monkeypatch, REAL_EVENTS)
        cc.codex_json("hi", model="gpt-5.4-mini", project="demo")
        assert ledger.query()[0]["notional_cost_usd"] == pytest.approx(
            cost_for("gpt-5.4-mini", 14169, 5, 4480))

    def test_notional_is_never_added_to_real_spend(self, monkeypatch, ledger):
        _stub_run(monkeypatch, REAL_EVENTS)
        cc.codex_json("hi", model="gpt-5.4-mini", project="demo")
        assert ledger.total() == 0.0
        assert ledger.total_notional() > 0
        by_mode = {r["grp"]: r for r in ledger.summary(group_by="billing_mode")}
        assert by_mode["subscription"]["cost_usd"] == 0.0
        assert by_mode["subscription"]["notional_cost_usd"] > 0

    def test_unknown_model_logs_tokens_with_no_price(self, monkeypatch, ledger):
        _stub_run(monkeypatch, REAL_EVENTS)
        cc.codex_json("hi", model="totally-made-up-model-xyz", project="demo")
        row = ledger.query()[0]
        assert row["notional_cost_usd"] is None
        assert row["prompt_tokens"] == 14169
        assert "unknown price" in json.loads(row["metadata"])["notional_cost"]

    def test_failed_call_leaves_a_zero_token_error_row(self, monkeypatch, ledger):
        _stub_run(monkeypatch, "", returncode=1, stderr="stream error: disconnected")
        monkeypatch.setattr(cc, "RETRIES", 0)
        with pytest.raises(cc.CodexCLIError):
            cc.codex_json("hi", project="demo", purpose="classify")

        row = ledger.query()[0]
        assert row["outcome"] == "error"
        assert row["prompt_tokens"] == 0
        assert row["billing_mode"] == "subscription"
        meta = json.loads(row["metadata"])
        assert "disconnected" in meta["error"]
        assert meta["usage"] == "no usage events in output"

    def test_a_timeout_still_bills_the_tokens_it_burned(self, monkeypatch, ledger):
        def _run(cmd, timeout):
            raise cc.CodexCLIError("codex CLI timed out after 900s", stdout=REAL_EVENTS)
        monkeypatch.setattr(cc, "_run", _run)
        with pytest.raises(cc.CodexCLIError):
            cc.codex_research("mission", project="demo")

        row = ledger.query()[0]
        assert row["prompt_tokens"] == 14169
        assert row["outcome"] == "error"
        assert row["script"] == "codex_cli.codex_research"

    def test_every_retried_attempt_gets_its_own_row(self, monkeypatch, ledger):
        calls = {"n": 0}

        def _run(cmd, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                return subprocess.CompletedProcess(cmd, 1, "", "transient blip")
            return subprocess.CompletedProcess(cmd, 0, REAL_EVENTS, "")

        monkeypatch.setattr(cc, "_run", _run)
        monkeypatch.setattr(cc.time, "sleep", lambda s: None)
        assert cc.codex_json("hi", project="demo") == "ok"
        assert len(ledger.query()) == 2

    def test_a_broken_ledger_does_not_break_the_call(self, monkeypatch, ledger):
        _stub_run(monkeypatch, REAL_EVENTS)

        def _explode(**kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(ledger, "log", _explode)
        assert cc.codex_json("hi", project="demo") == "ok"

    def test_opt_out_writes_nothing_and_drops_the_flag(self, monkeypatch, ledger, captured_cmd):
        assert cc.codex_json("hi", project="demo", cost_log=False) == "ok"
        assert "--json" not in captured_cmd[0]
        assert ledger.query() == []

    def test_env_kill_switch(self, monkeypatch, ledger, captured_cmd):
        monkeypatch.setenv("LIMBIC_CODEX_COST_LOG", "0")
        cc.codex_json("hi", project="demo")
        assert "--json" not in captured_cmd[0]
        assert ledger.query() == []

    def test_json_flag_is_on_by_default_for_both_entry_points(self, captured_cmd):
        cc.codex_json("hi", project="demo")
        cc.codex_research("mission", project="demo")
        assert all("--json" in cmd for cmd in captured_cmd)
        assert captured_cmd[0][-1] == "hi"        # still the last argument
        assert captured_cmd[1][-1] == "mission"


class TestJsonFlagUnsupported:
    """A Codex build predating `--json` must not take every call down with it."""

    def test_rejection_retries_without_the_flag(self, monkeypatch):
        seen: list[list[str]] = []

        def _spawn(cmd, timeout):
            seen.append(list(cmd))
            if "--json" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 2, "", "error: unexpected argument '--json' found")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(cc.shutil, "which", lambda name: "/usr/bin/codex")
        monkeypatch.setattr(cc, "_spawn", _spawn)
        assert cc.codex_json("hi", project="demo") == "ok"
        assert len(seen) == 2
        assert "--json" not in seen[1]
        assert cc._JSON_EVENTS_SUPPORTED is False

    def test_a_real_failure_is_not_mistaken_for_a_flag_problem(self, monkeypatch):
        seen: list[list[str]] = []

        def _spawn(cmd, timeout):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 1, "", "model error: overloaded")

        monkeypatch.setattr(cc.shutil, "which", lambda name: "/usr/bin/codex")
        monkeypatch.setattr(cc, "_spawn", _spawn)
        proc = cc._run(["codex", "exec", "--json"], timeout=5)
        assert proc.returncode == 1
        assert len(seen) == 1
        assert cc._JSON_EVENTS_SUPPORTED is True

    def test_a_mid_run_failure_quoting_the_flag_never_respawns(self, monkeypatch):
        """The retry is only free before the run starts.

        A failure that arrives *after* events were emitted has already spent
        its tokens; re-running an agentic mission on the strength of a phrase
        match would spend them twice and disable usage logging for the rest of
        the process on the way out.
        """
        seen: list[list[str]] = []

        def _spawn(cmd, timeout):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(
                cmd, 2, REAL_EVENTS,
                "tool call failed: error: unexpected argument '--json' found "
                "while running `jq --json`")

        monkeypatch.setattr(cc.shutil, "which", lambda name: "/usr/bin/codex")
        monkeypatch.setattr(cc, "_spawn", _spawn)
        proc = cc._run(["codex", "exec", "--json"], timeout=5)
        assert len(seen) == 1
        assert cc._JSON_EVENTS_SUPPORTED is True
        assert cc.parse_usage(proc.stdout).input_tokens == 14169

    def test_only_the_argument_parsers_exit_code_counts(self, monkeypatch):
        seen: list[list[str]] = []

        def _spawn(cmd, timeout):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(
                cmd, 1, "", "error: unexpected argument '--json' found")

        monkeypatch.setattr(cc.shutil, "which", lambda name: "/usr/bin/codex")
        monkeypatch.setattr(cc, "_spawn", _spawn)
        cc._run(["codex", "exec", "--json"], timeout=5)
        assert len(seen) == 1
        assert cc._JSON_EVENTS_SUPPORTED is True

    def test_the_complaint_has_to_be_on_stderr(self, monkeypatch):
        """clap writes usage errors to stderr; a model echoing the phrase on
        stdout is not the CLI refusing the flag."""
        seen: list[list[str]] = []

        def _spawn(cmd, timeout):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(
                cmd, 2, "error: unexpected argument '--json' found", "")

        monkeypatch.setattr(cc.shutil, "which", lambda name: "/usr/bin/codex")
        monkeypatch.setattr(cc, "_spawn", _spawn)
        cc._run(["codex", "exec", "--json"], timeout=5)
        assert len(seen) == 1
        assert cc._JSON_EVENTS_SUPPORTED is True
