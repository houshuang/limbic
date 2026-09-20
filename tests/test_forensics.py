"""Tests for limbic.cerebellum.forensics — session-transcript token forensics.

Fixtures are small, hand-built JSONL files that reproduce the two counting
traps found in the 20 Sep 2026 llm-pipeline-audit: a forked Codex subagent
whose cumulative counter starts already inflated by its parent, and a
resumed Codex session whose cumulative counter resets mid-file. Claude
fixtures reproduce streamed-duplicate `message.id` rows and a sidechain
(subagent) turn.
"""

from __future__ import annotations

import json

import pytest

from limbic.cerebellum.forensics import (
    attrib_session,
    parse_since,
    project_from_cwd,
    project_from_mentioned_paths,
    scan_claude_session,
    scan_claude_sessions,
    scan_codex_session,
    scan_codex_sessions,
)


def _write_jsonl(path, lines: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def _session_meta(cwd="/Users/stian/src/skard", thread_source="user", ts="2026-09-19T10:00:00Z", **extra):
    payload = {"cwd": cwd, "thread_source": thread_source, "timestamp": ts, **extra}
    return {"type": "session_meta", "payload": payload}


def _turn_context(model="gpt-5.6-sol"):
    return {"type": "turn_context", "payload": {"model": model}}


def _token_count(total_tokens, input_tokens, cached=0, output=0, reasoning=0):
    return {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {"total_tokens": total_tokens},
                "last_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached,
                    "output_tokens": output,
                    "reasoning_output_tokens": reasoning,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Codex: core counting rule
# ---------------------------------------------------------------------------


class TestCodexCounting:
    def test_sums_deltas_and_dedups_unchanged_cumulative(self, tmp_path):
        f = tmp_path / "s1.jsonl"
        _write_jsonl(f, [
            _session_meta(),
            _turn_context(),
            _token_count(total_tokens=1000, input_tokens=1000, output=50),
            _token_count(total_tokens=1000, input_tokens=1000, output=50),  # exact dup -> skipped
            _token_count(total_tokens=1500, input_tokens=500, cached=200, output=30),
        ])
        stats = scan_codex_session(f)
        assert stats.requests == 2
        assert stats.input_tokens == 1500  # 1000 + 500, not 2000
        assert stats.cached_tokens == 200
        assert stats.output_tokens == 80
        assert stats.model == "gpt-5.6-sol"
        assert stats.thread_source == "user"

    def test_skips_leading_zero_usage_event(self, tmp_path):
        f = tmp_path / "s2.jsonl"
        _write_jsonl(f, [
            _session_meta(),
            _token_count(total_tokens=100, input_tokens=0, output=0),  # zero-usage -> skipped
            _token_count(total_tokens=250, input_tokens=150, output=10),
        ])
        stats = scan_codex_session(f)
        assert stats.requests == 1
        assert stats.input_tokens == 150

    def test_forked_subagent_counter_does_not_inflate_own_total(self, tmp_path):
        """A forked subagent's cumulative counter starts already carrying the
        parent's usage. The delta-sum method must report only this session's
        own request sizes, not the inherited baseline."""
        f = tmp_path / "subagent.jsonl"
        _write_jsonl(f, [
            _session_meta(thread_source="subagent"),
            # First event: cumulative already includes ~500,000 inherited
            # tokens, but *this request's own* last_token_usage is small.
            _token_count(total_tokens=500_100, input_tokens=100, output=10),
            _token_count(total_tokens=500_240, input_tokens=140, output=15),
        ])
        stats = scan_codex_session(f)
        assert stats.thread_source == "subagent"
        # Correct: sums each request's own delta (100 + 140), never the
        # inflated cumulative total (500,240) a naive last-cumulative-value
        # read would have reported as "this session's input tokens".
        assert stats.input_tokens == 240
        assert stats.input_tokens < 500_240

    def test_resumed_session_cumulative_reset_still_counted_correctly(self, tmp_path):
        """A resumed session's cumulative counter can reset to a smaller
        value mid-file. Because counting relies only on last_token_usage
        (the cumulative is used solely to detect an unchanged duplicate),
        a reset is not mistaken for a duplicate and is not dropped."""
        f = tmp_path / "resumed.jsonl"
        _write_jsonl(f, [
            _session_meta(),
            _token_count(total_tokens=5000, input_tokens=100, output=5),
            _token_count(total_tokens=5000, input_tokens=100, output=5),  # dup -> skipped
            _token_count(total_tokens=1000, input_tokens=300, output=20),  # reset, but real
        ])
        stats = scan_codex_session(f)
        assert stats.requests == 2
        assert stats.input_tokens == 400  # 100 + 300, the reset event is not dropped

    def test_requests_over_150k_and_max_context(self, tmp_path):
        f = tmp_path / "big.jsonl"
        _write_jsonl(f, [
            _session_meta(),
            _token_count(total_tokens=1, input_tokens=200_000, output=10),
            _token_count(total_tokens=2, input_tokens=1_000, output=10),
        ])
        stats = scan_codex_session(f)
        assert stats.requests_over_150k == 1
        assert stats.max_context == 200_000


# ---------------------------------------------------------------------------
# Codex: project assignment
# ---------------------------------------------------------------------------


class TestCodexProjectAssignment:
    def test_scan_codex_sessions_glob_and_since(self, tmp_path):
        _write_jsonl(tmp_path / "2026" / "09" / "19" / "old.jsonl", [
            _session_meta(ts="2026-01-01T00:00:00Z"),
            _token_count(total_tokens=1, input_tokens=10, output=1),
        ])
        _write_jsonl(tmp_path / "2026" / "09" / "19" / "new.jsonl", [
            _session_meta(ts="2026-09-19T00:00:00Z"),
            _token_count(total_tokens=1, input_tokens=20, output=1),
        ])
        all_sessions = scan_codex_sessions(tmp_path)
        assert len(all_sessions) == 2

        recent = scan_codex_sessions(tmp_path, since=parse_since("1d"))
        # "since" is relative to now, and both fixture timestamps are in the
        # past, so with a tight enough window only sessions dated after the
        # cutoff survive. Use an explicit future-anchored cutoff instead for
        # a deterministic assertion.
        from datetime import datetime, timezone
        cutoff = datetime(2026, 6, 1, tzinfo=timezone.utc)
        filtered = scan_codex_sessions(tmp_path, since=cutoff)
        assert len(filtered) == 1
        assert filtered[0].input_tokens == 20

    def test_project_from_cwd_known_project(self):
        assert project_from_cwd("/Users/stian/src/skard/tools") == "skard"

    def test_project_from_cwd_research_subdir(self):
        assert project_from_cwd("/Users/stian/src/research/estonia-book") == "estonia-book"

    def test_project_from_cwd_unknown_returns_none(self):
        assert project_from_cwd("/Users/stian/Documents/misc") is None

    def test_project_from_mentioned_paths_below_threshold_is_none(self):
        text = "touched /otak/foo.py once"
        assert project_from_mentioned_paths(text, min_mentions=5) is None

    def test_project_from_mentioned_paths_above_threshold(self):
        text = " /otak/a.py " * 6
        assert project_from_mentioned_paths(text, min_mentions=5) == "otak"

    def test_project_disagreement_flag(self, tmp_path):
        f = tmp_path / "s.jsonl"
        body = " /petrarca/x.py " * 10
        _write_jsonl(f, [
            _session_meta(cwd="/Users/stian/src/otak"),
            _token_count(total_tokens=1, input_tokens=10, output=1),
        ])
        # Append raw noise mentioning a different project many times so the
        # paths-based signal disagrees with the cwd-based one.
        with open(f, "a") as fh:
            fh.write(body + "\n")
        stats = scan_codex_session(f)
        assert stats.project_by_cwd == "otak"
        assert stats.project_by_paths == "petrarca"
        assert stats.project_disagreement is True


# ---------------------------------------------------------------------------
# Codex: tool-call attribution
# ---------------------------------------------------------------------------


class TestAttribSession:
    def test_attributes_tokens_to_preceding_tool_category(self, tmp_path):
        f = tmp_path / "attrib.jsonl"
        _write_jsonl(f, [
            {"type": "event_msg", "payload": {"type": "custom_tool_call",
             "input": "git status"}},
            _token_count(total_tokens=1, input_tokens=100, output=5),
            {"type": "event_msg", "payload": {"type": "custom_tool_call",
             "input": "pytest tests/"}},
            _token_count(total_tokens=2, input_tokens=300, output=5),
            _token_count(total_tokens=3, input_tokens=50, output=5),  # no preceding tool call
        ])
        rows = attrib_session(f)
        by_cat = {r["category"]: r for r in rows}
        assert by_cat["git"]["input_tokens"] == 100
        assert by_cat["verify loop (unittest/verify/rebuild/validate/build)"]["input_tokens"] == 300
        assert by_cat["(user turn / no tool)"]["input_tokens"] == 50


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------


def _claude_assistant(msg_id, model, usage, sidechain=False, ts="2026-09-01T10:00:00Z"):
    return {
        "type": "assistant", "isSidechain": sidechain, "timestamp": ts,
        "message": {"id": msg_id, "model": model, "usage": usage},
    }


class TestClaudeSessions:
    def test_dedups_by_message_id(self, tmp_path):
        f = tmp_path / "sess.jsonl"
        usage = {"input_tokens": 2, "cache_creation_input_tokens": 100,
                 "cache_read_input_tokens": 50, "output_tokens": 30}
        _write_jsonl(f, [
            {"type": "user", "cwd": "/Users/stian/src/otak"},
            _claude_assistant("msg_1", "claude-opus-5", usage),
            _claude_assistant("msg_1", "claude-opus-5", usage),  # streamed dup -> skipped
            _claude_assistant("msg_1", "claude-opus-5", usage),  # streamed dup -> skipped
        ])
        stats = scan_claude_session(f)
        assert stats.cwd == "/Users/stian/src/otak"
        main = stats.main_by_model["claude-opus-5"]
        assert main["requests"] == 1
        assert main["input_tokens"] == 2
        assert main["cache_read_tokens"] == 50

    def test_splits_main_vs_sidechain(self, tmp_path):
        f = tmp_path / "sess2.jsonl"
        usage_main = {"input_tokens": 2, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 1000, "output_tokens": 30}
        usage_sub = {"input_tokens": 5, "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 200, "output_tokens": 10}
        _write_jsonl(f, [
            _claude_assistant("m1", "claude-opus-5", usage_main, sidechain=False),
            _claude_assistant("m2", "claude-haiku-4-5-20251001", usage_sub, sidechain=True),
        ])
        stats = scan_claude_session(f)
        assert "claude-opus-5" in stats.main_by_model
        assert "claude-haiku-4-5-20251001" in stats.sidechain_by_model
        assert stats.main_by_model.get("claude-haiku-4-5-20251001") is None
        assert stats.totals("main")["input_tokens"] == 2
        assert stats.totals("sidechain")["input_tokens"] == 5
        assert stats.totals("all")["input_tokens"] == 7

    def test_by_model_breakdown_across_two_models(self, tmp_path):
        f = tmp_path / "sess3.jsonl"
        u = {"input_tokens": 1, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "output_tokens": 1}
        _write_jsonl(f, [
            _claude_assistant("m1", "claude-opus-5", u),
            _claude_assistant("m2", "claude-sonnet-5", u),
        ])
        stats = scan_claude_session(f)
        assert set(stats.main_by_model.keys()) == {"claude-opus-5", "claude-sonnet-5"}

    def test_scan_claude_sessions_glob(self, tmp_path):
        u = {"input_tokens": 1, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "output_tokens": 1}
        _write_jsonl(tmp_path / "-Users-stian-src-otak" / "a.jsonl", [_claude_assistant("m1", "claude-opus-5", u)])
        _write_jsonl(tmp_path / "-Users-stian-src-skard" / "b.jsonl", [_claude_assistant("m2", "claude-opus-5", u)])
        sessions = scan_claude_sessions(tmp_path)
        assert len(sessions) == 2


class TestClaudeSubagents:
    """Subagent transcripts live at <project-dir>/<session-id>/subagents/agent-*.jsonl,
    not as inline isSidechain lines in the main file — verified against real
    transcripts under ~/.claude/projects/.../<session-id>/subagents/."""

    def _write_session_with_subagents(self, tmp_path, session_id, main_lines, subagent_files):
        project_dir = tmp_path / "-Users-stian-src-otak"
        _write_jsonl(project_dir / f"{session_id}.jsonl", main_lines)
        for name, lines in subagent_files.items():
            _write_jsonl(project_dir / session_id / "subagents" / f"{name}.jsonl", lines)
        return project_dir / f"{session_id}.jsonl"

    def test_subagent_file_counted_as_sidechain_for_the_parent(self, tmp_path):
        main_usage = {"input_tokens": 2, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 1000, "output_tokens": 30}
        sub_usage = {"input_tokens": 2, "cache_creation_input_tokens": 55806,
                    "cache_read_input_tokens": 0, "output_tokens": 279}
        main_path = self._write_session_with_subagents(
            tmp_path, "sess-1",
            main_lines=[_claude_assistant("m1", "claude-opus-5", main_usage)],
            subagent_files={
                "agent-apresenter-audit-e419ba7": [
                    {"type": "user", "isSidechain": True, "agentId": "apresenter-audit-e419ba7"},
                    _claude_assistant("sub1", "claude-opus-5", sub_usage, sidechain=True),
                ],
            },
        )
        stats = scan_claude_session(main_path)
        assert len(stats.subagents) == 1
        sub = stats.subagents[0]
        assert sub.agent_id == "apresenter-audit-e419ba7"
        assert sub.model == "claude-opus-5"
        # entrance fee = input + cache_creation + cache_read of its first request
        assert sub.first_turn_context == 2 + 55806 + 0

        # The subagent's usage is folded into sidechain_by_model for the parent.
        assert stats.sidechain_by_model["claude-opus-5"]["cache_creation_tokens"] == 55806
        assert stats.totals("main")["input_tokens"] == 2
        assert stats.totals("sidechain")["input_tokens"] == 2

    def test_multiple_subagents_and_no_subagents_dir(self, tmp_path):
        u = {"input_tokens": 1, "cache_creation_input_tokens": 100,
             "cache_read_input_tokens": 0, "output_tokens": 1}
        main_path = self._write_session_with_subagents(
            tmp_path, "sess-2",
            main_lines=[_claude_assistant("m1", "claude-opus-5", u)],
            subagent_files={
                "agent-a": [_claude_assistant("sa1", "claude-haiku-4-5-20251001", u, sidechain=True)],
                "agent-b": [_claude_assistant("sb1", "claude-haiku-4-5-20251001", u, sidechain=True)],
            },
        )
        stats = scan_claude_session(main_path)
        assert len(stats.subagents) == 2
        assert {s.agent_id for s in stats.subagents} == {"a", "b"}
        assert stats.sidechain_by_model["claude-haiku-4-5-20251001"]["requests"] == 2

    def test_session_without_subagents_dir_has_empty_list(self, tmp_path):
        u = {"input_tokens": 1, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "output_tokens": 1}
        f = tmp_path / "sess3.jsonl"
        _write_jsonl(f, [_claude_assistant("m1", "claude-opus-5", u)])
        stats = scan_claude_session(f)
        assert stats.subagents == []
        assert stats.sidechain_by_model == {}

    def test_scan_claude_sessions_does_not_double_count_subagent_files_as_sessions(self, tmp_path):
        u = {"input_tokens": 1, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "output_tokens": 1}
        self._write_session_with_subagents(
            tmp_path, "sess-4",
            main_lines=[_claude_assistant("m1", "claude-opus-5", u)],
            subagent_files={"agent-x": [_claude_assistant("sx1", "claude-opus-5", u, sidechain=True)]},
        )
        # scan_claude_sessions globs one level deep under the project dir, so
        # the subagents/*.jsonl files (two levels deeper) must not appear as
        # their own top-level "sessions".
        sessions = scan_claude_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].subagents[0].agent_id == "x"
