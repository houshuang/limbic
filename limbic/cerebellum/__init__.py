"""Cerebellum - LLM audit orchestration, budget tracking, and resumable batch processing."""

from .batch import BatchState, StateStore, BatchProcessor, ItemResult, BatchResult
from .orchestrator import (
    VerificationResult,
    VerificationTier,
    OrchestratorStatus,
    TieredOrchestrator,
    timeout_for,
)
from .audit_log import AuditEntry, AuditLogger, LogSummary, read_logs, extract_operations, summarize_logs
from .context import ContextBuilder, build_batch_context
from .cost_log import CostLog, CostRecord, cost_log, compute_cost
from .claude_cli import (
    ClaudeCLIError,
    Task as ClaudeTask,
    generate as claude_generate,
    generate_parallel as claude_generate_parallel,
    is_available as claude_is_available,
)

__all__ = [
    "BatchState",
    "StateStore",
    "BatchProcessor",
    "ItemResult",
    "BatchResult",
    "VerificationResult",
    "VerificationTier",
    "OrchestratorStatus",
    "TieredOrchestrator",
    "timeout_for",
    "AuditEntry",
    "AuditLogger",
    "LogSummary",
    "read_logs",
    "extract_operations",
    "summarize_logs",
    "ContextBuilder",
    "build_batch_context",
    "CostLog",
    "CostRecord",
    "cost_log",
    "compute_cost",
    "ClaudeCLIError",
    "ClaudeTask",
    "claude_generate",
    "claude_generate_parallel",
    "claude_is_available",
]
