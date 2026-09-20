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
# Deliberately NOT re-exporting the `cost_log` singleton here: this line used
# to also import it (`from .cost_log import ..., cost_log, ...`), which
# rebinds the *package* attribute `limbic.cerebellum.cost_log` from the
# submodule to that singleton instance. `import limbic.cerebellum.cost_log`
# or `from limbic.cerebellum import cost_log` then silently returns the
# instance, not the module — `limbic.cerebellum.cost_log.CostLog(...)` breaks,
# and `sys.modules["limbic.cerebellum.cost_log"]` is the only reliable way
# back to the real module. No consumer (grepped alif/petrarca/otak/dragoman/
# nrk) does either of those; every one uses the fully-qualified
# `from limbic.cerebellum.cost_log import cost_log`, which resolves via
# `sys.modules` and is unaffected either way — use that form.
from .cost_log import (
    CostLog,
    CostRecord,
    compute_cost,
    price_for,
    record_outcome,
    UnknownModelPriceError,
)
from .calls import cached_call, Held, CallMeta, TransportError
from .claude_cli import (
    ClaudeCLIError,
    Task as ClaudeTask,
    generate as claude_generate,
    generate_parallel as claude_generate_parallel,
    is_available as claude_is_available,
)
from .windowing import (
    Window,
    split_into_windows,
    Reference,
    Collection,
    MergeSchema,
    MergeReport,
    namespace_ids,
    dedup_by_field,
    merge_windows,
    check_references,
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
    "compute_cost",
    "price_for",
    "record_outcome",
    "UnknownModelPriceError",
    "cached_call",
    "Held",
    "CallMeta",
    "TransportError",
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
    "Window",
    "split_into_windows",
    "Reference",
    "Collection",
    "MergeSchema",
    "MergeReport",
    "namespace_ids",
    "dedup_by_field",
    "merge_windows",
    "check_references",
]
