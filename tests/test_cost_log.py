"""Tests for the cost_log ledger additions: migration, outcome, price_for, multi_group_summary."""

from __future__ import annotations

import json
import sqlite3

import pytest

from limbic.cerebellum.cost_log import (
    CostLog,
    UnknownModelPriceError,
    _migrate_columns,
    cached_input_price_for,
    compute_cost,
    cost_for,
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


# ---------------------------------------------------------------------------
# Backdated rows
# ---------------------------------------------------------------------------

class TestLogTimestamp:
    def test_default_is_now(self, tmp_path):
        from datetime import datetime, timezone

        cl = CostLog(db_path=tmp_path / "c.db")
        before = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        record = cl.log(project="p", model="m", cost_usd=0.0)
        after = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        assert before <= record.ts <= after

    def test_backfilled_row_keeps_its_date(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "c.db")
        record = cl.log(project="skard", model="m", cost_usd=0.5, purpose="backfill",
                        ts="2026-09-12T08:15:30Z")
        assert record.ts == "2026-09-12T08:15:30.000000Z"
        stored = cl.query(project="skard")[0]
        assert stored["ts"] == "2026-09-12T08:15:30.000000Z"
        assert cl.query(since="2026-09-13") == []

    def test_offsets_and_naive_datetimes_are_stored_as_utc(self, tmp_path):
        from datetime import datetime, timedelta, timezone

        cl = CostLog(db_path=tmp_path / "c.db")
        oslo = datetime(2026, 9, 12, 10, 15, 30, tzinfo=timezone(timedelta(hours=2)))
        assert cl.log(project="p", model="m", cost_usd=0.0, ts=oslo).ts == "2026-09-12T08:15:30.000000Z"
        assert cl.log(project="p", model="m", cost_usd=0.0,
                      ts=datetime(2026, 9, 12, 8, 15, 30)).ts == "2026-09-12T08:15:30.000000Z"
        assert cl.log(project="p", model="m", cost_usd=0.0,
                      ts="2026-09-12T10:15:30+02:00").ts == "2026-09-12T08:15:30.000000Z"


# ---------------------------------------------------------------------------
# Cached-input pricing
# ---------------------------------------------------------------------------

class TestCachedInputPricing:
    def test_openai_cached_rate(self):
        assert cached_input_price_for("gpt-5.4-mini") == 0.075
        assert cached_input_price_for("openai/gpt-4.1-mini") == 0.10

    def test_gemini_cached_rate(self):
        assert cached_input_price_for("gemini-2.5-flash") == 0.03

    def test_priced_model_without_a_known_discount_pays_full_input(self):
        assert cached_input_price_for("claude-haiku-4-5-20251001") == 1.00

    def test_unknown_model_strict_and_not(self):
        with pytest.raises(UnknownModelPriceError):
            cached_input_price_for("totally-made-up-model-xyz")
        assert cached_input_price_for("totally-made-up-model-xyz", strict=False) == 0.0
        assert cost_for("totally-made-up-model-xyz", 1000, 10, 500, strict=False) == 0.0
        with pytest.raises(UnknownModelPriceError):
            cost_for("totally-made-up-model-xyz", 1000, 10)

    def test_cost_for_bills_the_cached_share_at_the_cached_rate(self):
        # 5000 input of which 4096 cached, 200 output, gpt-5.4-mini 0.75 / 0.075 / 4.50
        expected = (904 * 0.75 + 4096 * 0.075 + 200 * 4.50) / 1_000_000
        assert cost_for("gpt-5.4-mini", 5000, 200, 4096) == pytest.approx(expected)

    def test_no_cached_tokens_is_the_old_formula(self):
        assert cost_for("gpt-5.4-mini", 5000, 200) == pytest.approx((5000 * 0.75 + 200 * 4.50) / 1_000_000)

    def test_cached_never_exceeds_prompt(self):
        assert cost_for("gpt-5.4-mini", 100, 0, 999) == pytest.approx(100 * 0.075 / 1_000_000)

    def test_compute_cost_applies_the_discount(self):
        assert compute_cost("gpt-5.4-mini", 5000, 200, 4096) == pytest.approx(cost_for("gpt-5.4-mini", 5000, 200, 4096))
        assert compute_cost("gpt-5.4-mini", 5000, 200) == pytest.approx(cost_for("gpt-5.4-mini", 5000, 200))
        assert compute_cost("totally-made-up-model-xyz", 5000, 200, 4096) is None

    def test_existing_rows_are_not_repriced(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "c.db")
        full_rate = (5000 * 0.75 + 200 * 4.50) / 1_000_000
        old = cl.log(project="skard", model="gpt-5.4-mini", prompt_tokens=5000,
                     completion_tokens=200, cached_tokens=4096, cost_usd=full_rate)
        cl.log(project="skard", model="gpt-5.4-mini", prompt_tokens=5000,
               completion_tokens=200, cached_tokens=4096)
        rows = {r["id"]: r["cost_usd"] for r in CostLog(db_path=tmp_path / "c.db").query()}
        assert rows[old.id] == pytest.approx(full_rate)


# ---------------------------------------------------------------------------
# Subscription rows: notional dollars that must never read as spend
# ---------------------------------------------------------------------------

_PRE_BILLING_MODE_SCHEMA = """
CREATE TABLE llm_costs (
    id TEXT PRIMARY KEY, ts TEXT NOT NULL, project TEXT NOT NULL,
    host TEXT NOT NULL, model TEXT NOT NULL, api_key_hint TEXT DEFAULT '',
    prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
    cached_tokens INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0.0,
    script TEXT DEFAULT '', purpose TEXT DEFAULT '', metadata TEXT DEFAULT '{}',
    cache_hit INTEGER DEFAULT 0, outcome TEXT DEFAULT NULL, packet_id TEXT DEFAULT NULL
);
"""


class TestBillingMode:
    def test_rows_written_before_the_column_existed_read_as_billed(self, tmp_path):
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.executescript(_PRE_BILLING_MODE_SCHEMA)
        conn.execute(
            "INSERT INTO llm_costs (id, ts, project, host, model, cost_usd) "
            "VALUES ('old1', '2026-01-01T00:00:00Z', 'skard', 'mac', 'gpt-5.4-mini', 1.25)")
        conn.commit()
        conn.close()

        cl = CostLog(db_path=db)
        row = cl.query()[0]
        assert row["billing_mode"] == "billed"
        assert row["notional_cost_usd"] is None
        assert cl.total() == pytest.approx(1.25)

    def test_a_subscription_row_spends_nothing(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "c.db")
        cl.log(project="hvaskjer", model="gpt-5.4-mini", prompt_tokens=10_000,
               completion_tokens=500, cost_usd=0.0, billing_mode="subscription",
               notional_cost_usd=0.01, purpose="enrich")
        cl.log(project="hvaskjer", model="gpt-5.4-mini", prompt_tokens=1_000,
               completion_tokens=100, purpose="adjudicate")

        assert cl.total() == pytest.approx(cost_for("gpt-5.4-mini", 1_000, 100))
        assert cl.total_notional() == pytest.approx(0.01)

    def test_notional_dollars_cannot_reach_cost_usd(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "c.db")
        with pytest.raises(ValueError, match="spends no money"):
            cl.log(project="hvaskjer", model="gpt-5.4-mini", cost_usd=0.01,
                   billing_mode="subscription")

    def test_unknown_billing_mode_is_refused(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "c.db")
        with pytest.raises(ValueError, match="billing_mode"):
            cl.log(project="x", model="gpt-5.4-mini", billing_mode="free")

    def test_summaries_report_the_two_separately(self, tmp_path):
        cl = CostLog(db_path=tmp_path / "c.db")
        cl.log(project="hvaskjer", model="gpt-5.4-mini", purpose="enrich",
               prompt_tokens=10_000, completion_tokens=500, cost_usd=0.0,
               billing_mode="subscription", notional_cost_usd=0.04, outcome="applied")
        cl.log(project="hvaskjer", model="gpt-5.4-mini", purpose="enrich",
               prompt_tokens=1_000, completion_tokens=100, outcome="applied")

        grouped = {r["grp"]: r for r in cl.summary(group_by="billing_mode")}
        assert grouped["subscription"]["cost_usd"] == 0.0
        assert grouped["subscription"]["notional_cost_usd"] == pytest.approx(0.04)
        assert grouped["billed"]["notional_cost_usd"] == 0.0

        multi = {r["billing_mode"]: r for r in cl.multi_group_summary(by=["billing_mode"])}
        assert multi["subscription"]["cost_per_applied"] == 0.0
        assert multi["subscription"]["notional_cost_per_applied"] == pytest.approx(0.04)

    def test_dashboard_keeps_notional_out_of_the_api_total(self, tmp_path):
        from limbic.cerebellum.cost_log import _build_summary

        cl = CostLog(db_path=tmp_path / "c.db")
        cl.log(project="hvaskjer", model="gpt-5.4-mini", prompt_tokens=10_000,
               completion_tokens=500, cost_usd=0.0, billing_mode="subscription",
               notional_cost_usd=0.04, script="codex_cli.codex_research")
        cl.log(project="skard", model="gpt-5.4-mini", prompt_tokens=1_000,
               completion_tokens=100)

        out = _build_summary(cl, days=None)
        assert out["api"]["calls"] == 1
        assert out["api"]["cost_usd"] == pytest.approx(cost_for("gpt-5.4-mini", 1_000, 100))
        assert out["subscription"]["calls"] == 1
        assert out["subscription"]["notional_usd"] == pytest.approx(0.04)
        assert out["subscription"]["prompt_tokens"] == 10_000

    def test_dashboard_counts_unpriced_subscription_calls(self, tmp_path):
        from limbic.cerebellum.cost_log import _build_summary

        cl = CostLog(db_path=tmp_path / "c.db")
        cl.log(project="hvaskjer", model="mystery-model", prompt_tokens=10, cost_usd=0.0,
               billing_mode="subscription", notional_cost_usd=None)
        assert _build_summary(cl, days=None)["subscription"]["unpriced_calls"] == 1


class TestMergeAcrossSchemaVersions:
    def test_a_remote_on_the_older_schema_still_merges(self, tmp_path):
        """A host running an older limbic writes a narrower table; `SELECT *`
        across the two fails on the column count and strands every row."""
        remote = tmp_path / "remote.db"
        conn = sqlite3.connect(remote)
        conn.executescript(_PRE_BILLING_MODE_SCHEMA)
        conn.execute(
            "INSERT INTO llm_costs (id, ts, project, host, model, cost_usd, purpose) "
            "VALUES ('r1', '2026-01-01T00:00:00Z', 'hvaskjer', 'alif', 'gpt-5.4-mini', 2.5, 'enrich')")
        conn.commit()
        conn.close()

        cl = CostLog(db_path=tmp_path / "local.db")
        assert cl.merge_from(remote) == 1
        row = cl.query()[0]
        assert row["purpose"] == "enrich"
        assert row["billing_mode"] == "billed"
        assert cl.total() == pytest.approx(2.5)

    def test_merging_twice_is_still_idempotent(self, tmp_path):
        remote = tmp_path / "remote.db"
        source = CostLog(db_path=remote)
        source.log(project="hvaskjer", model="gpt-5.4-mini", prompt_tokens=100,
                   cost_usd=0.0, billing_mode="subscription", notional_cost_usd=0.02)
        source.close()

        cl = CostLog(db_path=tmp_path / "local.db")
        assert cl.merge_from(remote) == 1
        assert cl.merge_from(remote) == 0
        assert cl.query()[0]["billing_mode"] == "subscription"
        assert cl.total_notional() == pytest.approx(0.02)
