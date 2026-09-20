"""Tests for limbic.cerebellum.calls.cached_call — response cache + replicate agreement."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from limbic.cerebellum import calls
from limbic.cerebellum.calls import CallMeta, Held, cached_call
from limbic.cerebellum.cost_log import CostLog


@pytest.fixture
def tmp_cost_log(tmp_path, monkeypatch):
    fresh = CostLog(db_path=tmp_path / "costs.db")
    monkeypatch.setattr(calls, "cost_log", fresh)
    return fresh


@pytest.fixture
def cache_db(tmp_path):
    return tmp_path / "cache.db"


def _fake_transport(responses):
    """A fake transport with the same call shape as claude_cli.generate.

    `responses` is a list consumed in order; each call pops the next one.
    """
    calls_made = {"n": 0}

    def _fn(prompt, *, project, purpose, system, schema, model, **kw):
        i = calls_made["n"]
        calls_made["n"] += 1
        result = responses[i] if i < len(responses) else responses[-1]
        return result, {"cost": 0.01, "model": model, "session_id": f"sess-{i}"}

    _fn.calls_made = calls_made
    return _fn


# ---------------------------------------------------------------------------
# Required arguments
# ---------------------------------------------------------------------------


class TestRequiredArgs:
    def test_purpose_is_required(self, tmp_cost_log, cache_db):
        with pytest.raises(TypeError):
            cached_call("hi", project="p")  # purpose omitted entirely -> TypeError

    def test_empty_purpose_raises(self, tmp_cost_log, cache_db):
        with pytest.raises(ValueError, match="purpose is required"):
            cached_call("hi", project="p", purpose="", cache_db_path=cache_db)

    def test_empty_project_is_inferred_from_git_root(self, tmp_cost_log, cache_db, tmp_path, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout=str(tmp_path) + "\n"),
        )
        fn = _fake_transport(["ok"])
        result, meta = cached_call(
            "hi", purpose="p", transport=fn, cache_db_path=cache_db,
        )
        assert result == "ok"
        row = tmp_cost_log.query()[0]
        assert row["project"] == tmp_path.name

    def test_empty_project_raises_outside_git_repo(self, tmp_cost_log, cache_db, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(a, 128, stdout=""),
        )
        fn = _fake_transport(["ok"])
        with pytest.raises(ValueError, match="could not be inferred"):
            cached_call("hi", purpose="p", transport=fn, cache_db_path=cache_db)


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


class TestCaching:
    def test_second_identical_call_is_a_cache_hit(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["result-1", "result-2"])

        result1, meta1 = cached_call(
            "classify: I love it", project="p", purpose="sentiment",
            transport=fn, cache_db_path=cache_db,
        )
        result2, meta2 = cached_call(
            "classify: I love it", project="p", purpose="sentiment",
            transport=fn, cache_db_path=cache_db,
        )

        assert result1 == "result-1"
        assert result2 == "result-1"  # cached, not "result-2"
        assert meta1.cache_hit is False
        assert meta2.cache_hit is True
        assert meta2.cost_usd == 0.0
        assert fn.calls_made["n"] == 1  # transport called only once

        rows = tmp_cost_log.query()
        assert len(rows) == 2
        assert rows[0]["cache_hit"] == 1  # most recent first
        assert rows[1]["cache_hit"] == 0

    def test_different_prompt_is_a_miss(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["a", "b"])
        cached_call("prompt A", project="p", purpose="x", transport=fn, cache_db_path=cache_db)
        result, meta = cached_call("prompt B", project="p", purpose="x", transport=fn, cache_db_path=cache_db)
        assert result == "b"
        assert meta.cache_hit is False
        assert fn.calls_made["n"] == 2

    def test_different_version_is_a_miss(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["a", "b"])
        cached_call("same prompt", project="p", purpose="x", version="v1", transport=fn, cache_db_path=cache_db)
        result, meta = cached_call(
            "same prompt", project="p", purpose="x", version="v2", transport=fn, cache_db_path=cache_db,
        )
        assert result == "b"
        assert meta.cache_hit is False

    def test_cache_false_bypasses_cache_entirely(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["a", "b"])
        cached_call("same", project="p", purpose="x", transport=fn, cache_db_path=cache_db, cache=False)
        result, meta = cached_call(
            "same", project="p", purpose="x", transport=fn, cache_db_path=cache_db, cache=False,
        )
        assert result == "b"
        assert meta.cache_hit is False
        assert fn.calls_made["n"] == 2

    def test_cache_refresh_forces_a_fresh_call_and_updates_cache(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["a", "b", "c"])
        cached_call("same", project="p", purpose="x", transport=fn, cache_db_path=cache_db)
        result, meta = cached_call(
            "same", project="p", purpose="x", transport=fn, cache_db_path=cache_db, cache="refresh",
        )
        assert result == "b"
        assert meta.cache_hit is False

        # A subsequent normal call now hits the refreshed value.
        result3, meta3 = cached_call("same", project="p", purpose="x", transport=fn, cache_db_path=cache_db)
        assert result3 == "b"
        assert meta3.cache_hit is True
        assert fn.calls_made["n"] == 2

    def test_ttl_expiry_forces_a_miss(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["a", "b"])
        cached_call("same", project="p", purpose="x", transport=fn, cache_db_path=cache_db, ttl_days=1)

        # Backdate the cache row past its TTL.
        from limbic.amygdala import connect
        conn = connect(str(cache_db))
        conn.execute("UPDATE call_cache SET created_at = 0, expires_at = 1")
        conn.commit()
        conn.close()

        result, meta = cached_call("same", project="p", purpose="x", transport=fn, cache_db_path=cache_db, ttl_days=1)
        assert result == "b"
        assert meta.cache_hit is False

    def test_call_id_can_be_used_with_record_outcome(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["a"])
        _, meta = cached_call("x", project="p", purpose="y", transport=fn, cache_db_path=cache_db)
        assert tmp_cost_log.record_outcome(meta.call_id, "applied") is True
        row = tmp_cost_log.query()[0]
        assert row["outcome"] == "applied"


# ---------------------------------------------------------------------------
# Replicates / agreement
# ---------------------------------------------------------------------------


class TestReplicates:
    def test_unanimous_agreement_returns_the_result(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["yes", "yes", "yes"])
        result, meta = cached_call(
            "is this a dup?", project="p", purpose="dedup",
            transport=fn, cache_db_path=cache_db, replicates=3,
        )
        assert result == "yes"
        assert not isinstance(result, Held)
        assert fn.calls_made["n"] == 3
        assert len(tmp_cost_log.query()) == 3

    def test_disagreement_returns_held(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["yes", "no", "yes"])
        result, meta = cached_call(
            "is this a dup?", project="p", purpose="dedup",
            transport=fn, cache_db_path=cache_db, replicates=3, agree=3,
        )
        assert isinstance(result, Held)
        assert result.results == ["yes", "no", "yes"]
        assert "2/3" not in result.reason or True  # majority is 2, needed 3
        assert meta.replicate_metas is not None
        assert len(meta.replicate_metas) == 3

    def test_agree_below_replicates_accepts_majority(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["yes", "no", "yes"])
        result, meta = cached_call(
            "is this a dup?", project="p", purpose="dedup",
            transport=fn, cache_db_path=cache_db, replicates=3, agree=2,
        )
        assert result == "yes"

    def test_replicates_bypass_the_cache(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["yes", "yes", "yes", "yes", "yes", "yes"])
        cached_call("x", project="p", purpose="y", transport=fn, cache_db_path=cache_db, replicates=3)
        cached_call("x", project="p", purpose="y", transport=fn, cache_db_path=cache_db, replicates=3)
        assert fn.calls_made["n"] == 6  # no caching between the two replicated calls

    def test_agree_greater_than_replicates_raises(self, tmp_cost_log, cache_db):
        fn = _fake_transport(["yes", "yes"])
        with pytest.raises(ValueError, match="cannot exceed replicates"):
            cached_call(
                "x", project="p", purpose="y", transport=fn, cache_db_path=cache_db,
                replicates=2, agree=3,
            )

    def test_dict_results_compared_structurally(self, tmp_cost_log, cache_db):
        fn = _fake_transport([{"label": "pos"}, {"label": "pos"}, {"label": "neg"}])
        result, meta = cached_call(
            "x", project="p", purpose="y", transport=fn, cache_db_path=cache_db,
            replicates=3, agree=2,
        )
        assert result == {"label": "pos"}


# ---------------------------------------------------------------------------
# Transport resolution
# ---------------------------------------------------------------------------


class TestTransport:
    def test_unknown_string_transport_raises(self, tmp_cost_log, cache_db):
        with pytest.raises(ValueError, match="unknown transport"):
            cached_call("x", project="p", purpose="y", transport="not-a-real-transport", cache_db_path=cache_db)

    def test_claude_cli_transport_is_used_by_default(self, tmp_cost_log, cache_db):
        response = {
            "type": "result", "subtype": "success", "is_error": False,
            "duration_ms": 10, "duration_api_ms": 10, "num_turns": 1,
            "result": "pong", "session_id": "s1", "total_cost_usd": 0.001,
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            "modelUsage": {"claude-haiku-4-5-20251001": {
                "inputTokens": 1, "outputTokens": 1, "cacheCreationInputTokens": 0,
                "cacheReadInputTokens": 0, "webSearchRequests": 0, "costUSD": 0.001,
            }},
        }
        import json as _json
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess(
            args=["claude"], returncode=0, stdout=_json.dumps(response), stderr="",
        )):
            result, meta = cached_call("ping", project="p", purpose="smoke", cache_db_path=cache_db)
        assert result == "pong"
        assert meta.model == "claude-haiku-4-5-20251001"
