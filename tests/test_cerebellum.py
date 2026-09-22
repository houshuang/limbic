"""Tests for limbic.cerebellum — batch processing, orchestration, audit logging, context."""

import json
import os
import time
from pathlib import Path

import pytest

from limbic.cerebellum import (
    AuditEntry,
    AuditLogger,
    BatchProcessor,
    BatchResult,
    BatchState,
    ContextBuilder,
    ItemResult,
    LogSummary,
    OrchestratorStatus,
    StateStore,
    TieredOrchestrator,
    VerificationResult,
    VerificationTier,
    build_batch_context,
    extract_operations,
    read_logs,
    summarize_logs,
    timeout_for,
)


# ===========================================================================
# StateStore
# ===========================================================================


class TestStateStore:
    def test_load_empty(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        state = store.load()
        assert isinstance(state, BatchState)
        assert state.items == {}
        assert state.total_cost == 0.0

    def test_save_and_load_roundtrip(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        state = BatchState()
        state.items["abc"] = {"status": "done", "ts": "2026-01-01"}
        state.total_cost = 1.23
        state.batches_run = 5
        store.save(state)

        loaded = store.load()
        assert loaded.items["abc"]["status"] == "done"
        assert loaded.total_cost == 1.23
        assert loaded.batches_run == 5

    def test_persistence(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        state = BatchState()
        state.items["x"] = {"status": "done"}
        store.save(state)

        # SQLite db should exist (auto-renamed from .json to .db)
        assert (tmp_path / "state.db").exists()

    def test_update_item(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.save(BatchState())
        store.update_item("item_1", "done", cost=0.5)
        store.update_item("item_2", "error", error="timeout")

        state = store.load()
        assert state.items["item_1"]["status"] == "done"
        assert state.items["item_1"]["cost"] == 0.5
        assert state.items["item_2"]["status"] == "error"

    def test_get_pending(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        state = BatchState()
        state.items["a"] = {"status": "done"}
        state.items["b"] = {"status": "error"}
        state.items["c"] = {"status": "verified"}
        store.save(state)

        pending = store.get_pending(["a", "b", "c", "d"])
        assert "a" not in pending  # done
        assert "b" in pending  # error is not in default done_statuses
        assert "c" not in pending  # verified
        assert "d" in pending  # never seen

    def test_get_pending_custom_done_statuses(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        state = BatchState()
        state.items["a"] = {"status": "done"}
        state.items["b"] = {"status": "error"}
        store.save(state)

        pending = store.get_pending(["a", "b"], done_statuses={"done", "error"})
        assert pending == []

    def test_get_status_counts(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        state = BatchState()
        state.items["a"] = {"status": "done"}
        state.items["b"] = {"status": "done"}
        state.items["c"] = {"status": "error"}
        store.save(state)

        counts = store.get_status_counts()
        assert counts == {"done": 2, "error": 1}


# ===========================================================================
# BatchProcessor
# ===========================================================================


class TestBatchProcessor:
    def _make_items(self, n=10):
        return [{"id": str(i), "name": f"item_{i}"} for i in range(n)]

    def test_basic_processing(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.save(BatchState())

        def process_fn(batch):
            return [
                ItemResult(id=item["id"], status="done", cost=0.1)
                for item in batch
            ]

        processor = BatchProcessor(store, batch_size=3)
        items = self._make_items(5)
        result = processor.process(items, process_fn, id_fn=lambda x: x["id"])

        assert result.processed == 5
        assert result.errors == 0
        assert abs(result.total_cost - 0.5) < 0.001

        state = store.load()
        assert len(state.items) == 5
        assert all(info["status"] == "done" for info in state.items.values())

    def test_budget_cap(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.save(BatchState())

        call_count = 0

        def process_fn(batch):
            nonlocal call_count
            call_count += 1
            return [
                ItemResult(id=item["id"], status="done", cost=0.5)
                for item in batch
            ]

        # max_cost=1.0, each batch of 2 costs 1.0 -> should stop after 1 batch
        processor = BatchProcessor(store, max_cost=1.0, batch_size=2)
        items = self._make_items(6)
        result = processor.process(items, process_fn, id_fn=lambda x: x["id"])

        assert call_count <= 2  # should not run all 3 batches
        assert result.total_cost <= 1.5  # may process 1 full batch before checking

    def test_resume_skips_completed(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        # Pre-populate some items as done
        state = BatchState()
        state.items["0"] = {"status": "done", "ts": "2026-01-01"}
        state.items["1"] = {"status": "done", "ts": "2026-01-01"}
        store.save(state)

        processed_ids = []

        def process_fn(batch):
            results = []
            for item in batch:
                processed_ids.append(item["id"])
                results.append(ItemResult(id=item["id"], status="done", cost=0.1))
            return results

        processor = BatchProcessor(store, batch_size=10)
        items = self._make_items(5)
        result = processor.process(items, process_fn, id_fn=lambda x: x["id"])

        # Items 0 and 1 should be skipped
        assert "0" not in processed_ids
        assert "1" not in processed_ids
        assert result.skipped == 2
        assert result.processed == 3

    def test_error_handling(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.save(BatchState())

        def process_fn(batch):
            raise RuntimeError("API down")

        processor = BatchProcessor(store, batch_size=3)
        items = self._make_items(3)
        result = processor.process(items, process_fn, id_fn=lambda x: x["id"])

        assert result.errors == 3
        assert result.processed == 0

        state = store.load()
        assert all(info["status"] == "error" for info in state.items.values())


# ===========================================================================
# AuditLogger / read_logs
# ===========================================================================


class TestAuditLogger:
    def test_write_and_read(self, tmp_path):
        logger = AuditLogger(tmp_path / "logs", prefix="test")
        logger.log({"item_id": "abc", "action": "verify", "cost": 0.5})
        logger.log({"item_id": "def", "action": "error", "cost": 0.0})

        entries = list(read_logs(tmp_path / "logs", prefix="test"))
        assert len(entries) == 2
        assert entries[0].item_id == "abc"
        assert entries[1].action == "error"

    def test_log_entry_typed(self, tmp_path):
        logger = AuditLogger(tmp_path / "logs", prefix="typed")
        entry = AuditEntry(
            timestamp="2026-01-01T00:00:00",
            item_id="x",
            action="verify",
            cost=1.0,
            tier="tier1",
        )
        logger.log_entry(entry)

        entries = list(read_logs(tmp_path / "logs", prefix="typed"))
        assert len(entries) == 1
        assert entries[0].tier == "tier1"
        assert entries[0].cost == 1.0

    def test_read_logs_since_filter(self, tmp_path):
        logger = AuditLogger(tmp_path / "logs", prefix="filt")
        logger.log({"ts": "2026-01-01T00:00:00", "item_id": "old", "action": "a"})
        logger.log({"ts": "2026-06-01T00:00:00", "item_id": "new", "action": "b"})

        entries = list(read_logs(tmp_path / "logs", prefix="filt", since="2026-03-01"))
        assert len(entries) == 1
        assert entries[0].item_id == "new"

    def test_read_logs_empty_dir(self, tmp_path):
        entries = list(read_logs(tmp_path / "nonexistent"))
        assert entries == []


class TestExtractOperations:
    def test_basic_extraction(self):
        entries = [
            AuditEntry(
                timestamp="2026-01-01",
                item_id="a",
                action="verify",
                details={
                    "operations": [
                        {"type": "merge", "ids": [1, 2]},
                        {"type": "fix", "field": "year"},
                    ]
                },
            ),
        ]
        ops = extract_operations(entries)
        assert "merge" in ops
        assert "fix" in ops
        assert len(ops["merge"]) == 1
        assert len(ops["fix"]) == 1

    def test_dedup_by_key(self):
        entries = [
            AuditEntry(timestamp="2026-01-01", item_id="a", action="merge",
                        details={"operations": [{"type": "merge", "ids": [1, 2], "_ts": "2026-01-01"}]}),
            AuditEntry(timestamp="2026-01-02", item_id="b", action="merge",
                        details={"operations": [{"type": "merge", "ids": [1, 2], "_ts": "2026-01-02"}]}),
        ]
        ops = extract_operations(
            entries,
            dedup_key_fn=lambda op: (op.get("type"), tuple(sorted(op.get("ids", [])))),
        )
        assert len(ops["merge"]) == 1
        # Should keep the latest
        assert ops["merge"][0]["_ts"] == "2026-01-02"

    def test_filter_by_op_types(self):
        entries = [
            AuditEntry(timestamp="t", item_id="a", action="verify",
                        details={"operations": [
                            {"type": "merge", "ids": [1, 2]},
                            {"type": "fix", "field": "year"},
                        ]}),
        ]
        ops = extract_operations(entries, op_types=["fix"])
        assert "fix" in ops
        assert "merge" not in ops

    def test_entry_as_operation_fallback(self):
        entries = [
            AuditEntry(timestamp="t", item_id="x", action="check", details={"note": "ok"}),
        ]
        ops = extract_operations(entries)
        assert "check" in ops
        assert ops["check"][0]["item_id"] == "x"


class TestSummarizeLogs:
    def test_basic_summary(self):
        entries = [
            AuditEntry(timestamp="t1", item_id="a", action="verify", cost=1.0, tier="tier1"),
            AuditEntry(timestamp="t2", item_id="b", action="verify", cost=2.0, tier="tier1"),
            AuditEntry(timestamp="t3", item_id="c", action="error", cost=0.5, tier="tier2",
                        details={"status": "error"}),
        ]
        summary = summarize_logs(entries)
        assert summary.total_cost == 3.5
        assert summary.items_processed == 3
        assert summary.error_count == 1
        assert summary.by_tier["tier1"]["count"] == 2
        assert summary.by_tier["tier2"]["cost"] == 0.5
        assert summary.by_action["verify"] == 2
        assert summary.by_action["error"] == 1


# ===========================================================================
# ContextBuilder
# ===========================================================================


class TestContextBuilder:
    def test_markdown_output(self):
        ctx = ContextBuilder()
        ctx.add_entity("work", "42", {"title": "Peer Gynt", "year": 1867})
        ctx.add_related("performances", [{"id": 1, "venue": "DNS"}, {"id": 2, "venue": "NTO"}])
        ctx.add_metadata("category", "teater")

        md = ctx.build(format="markdown")
        assert "## work/42" in md
        assert "Peer Gynt" in md
        assert "### performances (2)" in md
        assert "DNS" in md
        assert "category: teater" in md

    def test_json_output(self):
        ctx = ContextBuilder()
        ctx.add_entity("person", "7", {"name": "Ibsen"})
        ctx.add_related("works", [{"id": 42}])
        ctx.add_metadata("role", "playwright")

        result = ctx.build(format="json")
        assert isinstance(result, dict)
        assert result["entities"][0]["name"] == "Ibsen"
        assert result["related"]["works"] == [{"id": 42}]
        assert result["metadata"]["role"] == "playwright"

    def test_chaining(self):
        ctx = (
            ContextBuilder()
            .add_entity("a", "1", {"x": 1})
            .add_related("b", [])
            .add_metadata("c", 3)
        )
        md = ctx.build()
        assert "## a/1" in md

    def test_batch_context_markdown(self):
        items = [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}]

        def ctx_fn(item):
            return ContextBuilder().add_entity("item", item["id"], {"name": item["name"]})

        result = build_batch_context(items, ctx_fn)
        assert "## item/1" in result
        assert "## item/2" in result
        assert "---" in result

    def test_batch_context_json(self):
        items = [{"id": "1"}, {"id": "2"}]

        def ctx_fn(item):
            return ContextBuilder().add_entity("item", item["id"], {"val": True})

        result = build_batch_context(items, ctx_fn, format="json")
        parsed = json.loads(result)
        assert len(parsed) == 2


# ===========================================================================
# TieredOrchestrator
# ===========================================================================


class TestTieredOrchestrator:
    def test_single_tier_execution(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.save(BatchState())

        def tier1_fn(batch):
            return [
                VerificationResult(item_id=item["id"], status="verified", confidence=0.95, cost=0.1)
                for item in batch
            ]

        tier1 = VerificationTier(name="fast", process_fn=tier1_fn, description="Quick check")
        orch = TieredOrchestrator([tier1], store)

        items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        results = orch.run(items, id_fn=lambda x: x["id"], tier_name="fast", batch_size=10)

        assert "fast" in results
        assert len(results["fast"]) == 3
        assert all(r.status == "verified" for r in results["fast"])

    def test_auto_escalation(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.save(BatchState())

        tier2_called_with = []

        def tier1_fn(batch):
            results = []
            for item in batch:
                # Flag items with even IDs
                if int(item["id"]) % 2 == 0:
                    results.append(VerificationResult(
                        item_id=item["id"], status="flagged", confidence=0.4, cost=0.05))
                else:
                    results.append(VerificationResult(
                        item_id=item["id"], status="verified", confidence=0.9, cost=0.05))
            return results

        def tier2_fn(batch):
            tier2_called_with.extend([item["id"] for item in batch])
            return [
                VerificationResult(item_id=item["id"], status="verified", confidence=0.95, cost=0.3)
                for item in batch
            ]

        tier1 = VerificationTier(name="fast", process_fn=tier1_fn)
        tier2 = VerificationTier(name="deep", process_fn=tier2_fn)
        orch = TieredOrchestrator([tier1, tier2], store)

        items = [{"id": str(i)} for i in range(4)]
        results = orch.run(items, id_fn=lambda x: x["id"], escalate=True, batch_size=10)

        # Even IDs (0, 2) should have been escalated to tier2
        assert "0" in tier2_called_with
        assert "2" in tier2_called_with
        assert "1" not in tier2_called_with

    def test_status_report(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        state = BatchState()
        state.items["a"] = {"status": "done", "tier": "fast"}
        state.items["b"] = {"status": "done", "tier": "fast"}
        state.items["c"] = {"status": "needs_review", "tier": "fast"}
        state.total_cost = 1.5
        store.save(state)

        tier = VerificationTier(name="fast", process_fn=lambda b: [])
        orch = TieredOrchestrator([tier], store)

        status = orch.status(all_ids=["a", "b", "c", "d"])
        assert status.remaining_items == 2  # "c" (needs_review) and "d" (unseen)
        assert status.total_cost == 1.5
        assert status.tier_counts["fast"]["done"] == 2

    def test_budget_across_tiers(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.save(BatchState())

        def expensive_fn(batch):
            return [
                VerificationResult(item_id=item["id"], status="flagged", confidence=0.5, cost=5.0)
                for item in batch
            ]

        tier1 = VerificationTier(name="t1", process_fn=expensive_fn)
        tier2 = VerificationTier(name="t2", process_fn=expensive_fn)
        orch = TieredOrchestrator([tier1, tier2], store)

        items = [{"id": str(i)} for i in range(5)]
        # Budget of 10 should be consumed by tier1 (5 items * 5.0 = 25 cost, but stops at budget)
        results = orch.run(items, id_fn=lambda x: x["id"], max_cost=10.0, escalate=True, batch_size=2)

        state = store.load()
        assert state.total_cost <= 15.0  # allows some overshoot within a batch


# ===========================================================================
# timeout_for
# ===========================================================================


class TestTimeoutFor:
    def test_default_no_scale(self):
        assert timeout_for("anything") == 30

    def test_with_scale_fn(self):
        result = timeout_for(
            {"work_count": 10},
            base_timeout=30,
            scale_fn=lambda item: item["work_count"],
        )
        assert result == 300  # 30 * 10

    def test_respects_max(self):
        result = timeout_for(
            "x",
            base_timeout=30,
            scale_fn=lambda _: 1000,
            max_timeout=600,
        )
        assert result == 600

    def test_minimum_is_base(self):
        result = timeout_for(
            "x",
            base_timeout=60,
            scale_fn=lambda _: 0.1,  # would give 6, but base is 60
        )
        assert result == 60


# ===========================================================================
# Regression tests for P1 bugs
# ===========================================================================

class TestEscalationPreservesContext:
    """Regression: escalation must preserve tier-1 metadata, not overwrite it."""

    def test_update_item_merges(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.save(BatchState())

        # Simulate tier-1 completion with rich metadata
        store.update_item("item1", "needs_review",
                          confidence=0.45, tier="fast",
                          findings=["suspicious attribution"])
        state = store.load()
        assert state.items["item1"]["confidence"] == 0.45
        assert state.items["item1"]["findings"] == ["suspicious attribution"]

        # Simulate escalation -- should merge, not overwrite
        store.update_item("item1", "pending", escalated_from="fast")
        state = store.load()
        assert state.items["item1"]["status"] == "pending"
        assert state.items["item1"]["escalated_from"] == "fast"
        # Prior metadata must survive
        assert state.items["item1"]["confidence"] == 0.45
        assert state.items["item1"]["findings"] == ["suspicious attribution"]
        assert state.items["item1"]["tier"] == "fast"


class TestCodexCLIRetry:
    """_exec retries transient failures once; quota and timeout don't retry."""

    def _proc(self, rc=1, stderr="", stdout=""):
        import subprocess
        return subprocess.CompletedProcess(args=["codex"], returncode=rc,
                                           stdout=stdout, stderr=stderr)

    def test_transient_exit_retries_then_succeeds(self, monkeypatch):
        from limbic.cerebellum import codex_cli as cc
        calls = []
        procs = [self._proc(rc=1, stderr="banner\nboom"), self._proc(rc=0, stdout="ok")]
        monkeypatch.setattr(cc, "_run", lambda cmd, timeout, stdin_text=None: calls.append(1) or procs[len(calls) - 1])
        monkeypatch.setattr(cc.time, "sleep", lambda s: None)
        assert cc._exec(["codex"], 10, None, None) == "ok"
        assert len(calls) == 2

    def test_quota_error_does_not_retry(self, monkeypatch):
        from limbic.cerebellum import codex_cli as cc
        calls = []
        monkeypatch.setattr(cc, "_run",
                            lambda cmd, timeout, stdin_text=None: calls.append(1) or self._proc(rc=1, stderr="usage limit reached"))
        monkeypatch.setattr(cc, "_DISABLED_UNTIL", 0.0)
        with pytest.raises(cc.CodexCLIError):
            cc._exec(["codex"], 10, None, None)
        assert len(calls) == 1
        assert cc.temporarily_disabled()
        monkeypatch.setattr(cc, "_DISABLED_UNTIL", 0.0)

    def test_timeout_does_not_retry(self, monkeypatch):
        from limbic.cerebellum import codex_cli as cc
        calls = []

        def _run(cmd, timeout, stdin_text=None):
            calls.append(1)
            raise cc.CodexCLIError("codex CLI timed out after 10s")

        monkeypatch.setattr(cc, "_run", _run)
        with pytest.raises(cc.CodexCLIError, match="timed out"):
            cc._exec(["codex"], 10, None, None)
        assert len(calls) == 1

    def test_error_keeps_stderr_tail(self, monkeypatch):
        from limbic.cerebellum import codex_cli as cc
        banner = "Reading additional input from stdin...\n" + "x" * 800
        stderr = banner + "\nFATAL: the real reason"
        monkeypatch.setattr(cc, "RETRIES", 0)
        monkeypatch.setattr(cc, "_run", lambda cmd, timeout, stdin_text=None: self._proc(rc=1, stderr=stderr))
        with pytest.raises(cc.CodexCLIError, match="the real reason"):
            cc._exec(["codex"], 10, None, None)
