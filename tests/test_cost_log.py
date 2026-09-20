"""Tests for the cost_log ledger additions: migration, outcome, price_for, multi_group_summary."""

from __future__ import annotations

import json
import sqlite3

import pytest

from limbic.cerebellum.cost_log import (
    CostLog,
    UnknownModelPriceError,
    _migrate_columns,
    price_for,
    record_outcome,
)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _make_pre_migration_db(path):
    """Build a DB with the *old* llm_costs schema (no cache_hit/outcome/packet_id)."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE llm_costs (
            id          TEXT PRIMARY KEY,
            ts          TEXT NOT NULL,
            project     TEXT NOT NULL,
            host        TEXT NOT NULL,
            model       TEXT NOT NULL,
            api_key_hint TEXT DEFAULT '',
            prompt_tokens    INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            cached_tokens    INTEGER DEFAULT 0,
            cost_usd    REAL DEFAULT 0.0,
            script      TEXT DEFAULT '',
            purpose     TEXT DEFAULT '',
            metadata    TEXT DEFAULT '{}'
        )
    """)
    conn.execute(
        "INSERT INTO llm_costs (id, ts, project, host, model) VALUES (?, ?, ?, ?, ?)",
        ("row1", "2026-01-01T00:00:00Z", "oldproj", "host1", "haiku"),
    )
    conn.commit()
    conn.close()


class TestMigration:
    def test_adds_missing_columns_idempotently(self, tmp_path):
        db = tmp_path / "old.db"
        _make_pre_migration_db(db)

        cl = CostLog(db_path=db)
        conn = cl._connect()  # triggers migration
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(llm_costs)")}
        assert {"cache_hit", "outcome", "packet_id"} <= cols

        # Old row survives with sane defaults.
        row = conn.execute("SELECT * FROM llm_costs WHERE id = 'row1'").fetchone()
        assert row["cache_hit"] == 0
        assert row["outcome"] is None
        assert row["packet_id"] is None

        # Running the migration again is a no-op, not an error.
        _migrate_columns(conn)
        _migrate_columns(conn)

    def test_fresh_db_has_columns_from_create(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "fresh.db")
        conn = cl._connect()
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(llm_costs)")}
        assert {"cache_hit", "outcome", "packet_id"} <= cols


# ---------------------------------------------------------------------------
# log() new columns + record_outcome
# ---------------------------------------------------------------------------


class TestOutcomeAndCacheHit:
    def test_log_writes_cache_hit_and_packet_id(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        record = cl.log(project="p", model="haiku", cache_hit=True, packet_id="pkt-1")
        row = cl.query()[0]
        assert row["cache_hit"] == 1
        assert row["packet_id"] == "pkt-1"
        assert record.cache_hit is True

    def test_record_outcome_updates_existing_row(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        record = cl.log(project="p", model="haiku")
        assert cl.record_outcome(record.id, "applied") is True
        row = cl.query()[0]
        assert row["outcome"] == "applied"

    def test_record_outcome_with_detail_merges_into_metadata(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        record = cl.log(project="p", model="haiku", metadata={"foo": "bar"})
        cl.record_outcome(record.id, "rejected", detail="schema mismatch")
        row = cl.query()[0]
        assert row["outcome"] == "rejected"
        meta = json.loads(row["metadata"])
        assert meta["foo"] == "bar"
        assert meta["outcome_detail"] == "schema mismatch"

    def test_record_outcome_missing_row_returns_false(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        assert cl.record_outcome("does-not-exist", "applied") is False

    def test_module_level_record_outcome(self, tmp_path, monkeypatch):
        # `limbic.cerebellum.__init__` does `from .cost_log import cost_log`,
        # which shadows the `cost_log` submodule with the singleton on the
        # *package* attribute of the same name — `import
        # limbic.cerebellum.cost_log` after that returns the singleton, not
        # the module. Go through `sys.modules` to get the real module and
        # patch its own global, matching how `record_outcome()` looks it up.
        import sys
        cost_log_module = sys.modules["limbic.cerebellum.cost_log"]
        fresh = CostLog(db_path=tmp_path / "costs.db")
        monkeypatch.setattr(cost_log_module, "cost_log", fresh)
        record = fresh.log(project="p", model="haiku")
        assert record_outcome(record.id, "applied") is True


# ---------------------------------------------------------------------------
# price_for
# ---------------------------------------------------------------------------


class TestPriceFor:
    def test_known_model_from_fallback_table(self):
        inp, out = price_for("claude-haiku-4-5-20251001")
        assert inp == 1.00
        assert out == 5.00

    def test_unknown_model_raises_by_default(self):
        with pytest.raises(UnknownModelPriceError, match="no known price"):
            price_for("totally-made-up-model-xyz")

    def test_unknown_model_non_strict_returns_zero(self, caplog):
        assert price_for("totally-made-up-model-xyz", strict=False) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# multi_group_summary
# ---------------------------------------------------------------------------


class TestMultiGroupSummary:
    def _seed(self, cl: CostLog):
        cl.log(project="petrarca", model="haiku", purpose="extract", cost_usd=0.01,
               prompt_tokens=10, completion_tokens=5)
        cl.log(project="petrarca", model="haiku", purpose="extract", cost_usd=0.0,
               cache_hit=True)
        r = cl.log(project="petrarca", model="haiku", purpose="extract", cost_usd=0.02,
                   prompt_tokens=20, completion_tokens=10)
        cl.record_outcome(r.id, "applied")
        cl.log(project="skard", model="sonnet", purpose="dedup", cost_usd=0.05)

    def test_groups_by_multiple_columns(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        self._seed(cl)
        rows = cl.multi_group_summary(by=["project", "purpose"])
        by_key = {(r["project"], r["purpose"]): r for r in rows}
        assert by_key[("petrarca", "extract")]["calls"] == 3
        assert by_key[("skard", "dedup")]["calls"] == 1

    def test_cost_per_applied(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        self._seed(cl)
        rows = cl.multi_group_summary(by=["project"])
        petrarca = next(r for r in rows if r["project"] == "petrarca")
        assert petrarca["applied_count"] == 1
        assert petrarca["cost_per_applied"] == pytest.approx(0.03)  # 0.01+0+0.02 / 1

        skard = next(r for r in rows if r["project"] == "skard")
        assert skard["applied_count"] == 0
        assert skard["cost_per_applied"] is None

    def test_cache_hit_rate(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        self._seed(cl)
        rows = cl.multi_group_summary(by=["project"])
        petrarca = next(r for r in rows if r["project"] == "petrarca")
        assert petrarca["cache_hit_rate"] == pytest.approx(1 / 3)

    def test_since_filters(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        self._seed(cl)
        rows = cl.multi_group_summary(by=["project"], since="2099-01-01T00:00:00Z")
        assert rows == []

    def test_rejects_unknown_column(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        with pytest.raises(ValueError, match="unknown group column"):
            cl.multi_group_summary(by=["project; DROP TABLE llm_costs"])

    def test_requires_at_least_one_column(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "costs.db")
        with pytest.raises(ValueError, match="at least one column"):
            cl.multi_group_summary(by=[])
