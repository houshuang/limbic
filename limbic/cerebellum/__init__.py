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
from .sandbox import (
    AgentBudgetExceeded,
    untrusted_payload,
    isolated_scratch,
    sanitized_environment,
    call_slot,
)
from .codex_cli import (
    CodexCLIError,
    codex_json,
    codex_research,
    strict_response_schema as codex_strict_response_schema,
    is_available as codex_is_available,
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
    "CodexCLIError",
    "codex_json",
    "codex_research",
    "codex_strict_response_schema",
    "codex_is_available",
    "AgentBudgetExceeded",
    "untrusted_payload",
    "isolated_scratch",
    "sanitized_environment",
    "call_slot",
]
