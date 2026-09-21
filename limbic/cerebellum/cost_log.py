"""Centralized LLM cost logger — tracks spend across projects, models, and hosts.

Logs every LLM call to a single SQLite database with cost computed via litellm's
pricing data (2,500+ models).  Works with any SDK: litellm, google-genai, anthropic,
openai, or raw REST.

Usage — standalone (any SDK):

    from limbic.cerebellum.cost_log import cost_log
    cost_log.log(project="petrarca", model="gemini/gemini-2.5-flash",
                 prompt_tokens=1200, completion_tokens=340)

Usage — litellm callback (auto-captures every litellm.completion call):

    import litellm
    from limbic.cerebellum.cost_log import cost_log
    litellm.callbacks = [cost_log.callback("alif")]

DB location (in order of precedence):
    1. COST_LOG_DB environment variable
    2. ~/.local/share/limbic/llm_costs.db

Sync:
    python -m limbic.cerebellum.cost_log sync --host alif
    python -m limbic.cerebellum.cost_log report --days 7
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_costs (
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
    metadata    TEXT DEFAULT '{}',
    cache_hit   INTEGER DEFAULT 0,
    outcome     TEXT DEFAULT NULL,
    packet_id   TEXT DEFAULT NULL,
    billing_mode TEXT DEFAULT 'billed',
    notional_cost_usd REAL DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_costs_ts ON llm_costs(ts);
CREATE INDEX IF NOT EXISTS idx_costs_project ON llm_costs(project);
CREATE INDEX IF NOT EXISTS idx_costs_model ON llm_costs(model);
CREATE INDEX IF NOT EXISTS idx_costs_host ON llm_costs(host);
"""

# Columns added after the original schema shipped. `CREATE TABLE IF NOT EXISTS`
# above is a no-op against an existing database file, so an older `llm_costs`
# table needs these added explicitly. Guarded by `PRAGMA table_info` so it is
# safe to run against every connection, every time (idempotent). The
# `idx_costs_outcome` index is created here too, *after* the column is
# guaranteed to exist — putting it in `_SCHEMA` instead would break on an
# older DB, where `executescript` runs before the ALTER TABLE below and
# `CREATE INDEX ... (outcome)` fails with "no such column".
_MIGRATION_COLUMNS: dict[str, str] = {
    "cache_hit": "INTEGER DEFAULT 0",
    "outcome": "TEXT DEFAULT NULL",
    "packet_id": "TEXT DEFAULT NULL",
    "billing_mode": "TEXT DEFAULT 'billed'",
    "notional_cost_usd": "REAL DEFAULT NULL",
}

# How a row's money is to be read.
#
#   billed       — real money left an account. `cost_usd` is that money.
#   subscription — the call was covered by a flat-rate plan (Codex under a
#                  ChatGPT subscription), so no per-token money was spent.
#                  `cost_usd` is 0 and `notional_cost_usd` holds what the same
#                  tokens would have cost on the API, which is an *estimate of
#                  value, not of spend*.
#
# Keeping the notional figure out of `cost_usd` is the whole point of the
# split: every existing reader — this module's `total()`, the dashboard, an
# older limbic on another host, an ad-hoc `SELECT SUM(cost_usd)` — keeps
# returning real spend without knowing the column exists. A reader that wants
# the subscription figure has to ask for it by name, and therefore has to
# label it.
BILLING_MODES = frozenset({"billed", "subscription"})


def _migrate_columns(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(llm_costs)")}
    for col, decl in _MIGRATION_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE llm_costs ADD COLUMN {col} {decl}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_costs_outcome ON llm_costs(outcome)")

# ---------------------------------------------------------------------------
# Default DB path
# ---------------------------------------------------------------------------

def _default_db_path() -> Path:
    env = os.environ.get("COST_LOG_DB")
    if env:
        return Path(env)
    return Path.home() / ".local" / "share" / "limbic" / "llm_costs.db"


def _detect_host() -> str:
    return os.environ.get("COST_LOG_HOST", platform.node().split(".")[0])


def _parse_since(spec: str) -> str:
    """Parse a relative window like "30d" / "12h" / "45m" into an ISO cutoff.

    Accepts a bare integer as days, for convenience. Used by the `report
    --since` CLI flag and `multi_group_summary`.
    """
    spec = spec.strip()
    m = re.match(r"^(\d+)\s*([dhm]?)$", spec)
    if not m:
        raise ValueError(f"can't parse --since {spec!r}; use e.g. 30d, 12h, 45m")
    n, unit = int(m.group(1)), (m.group(2) or "d")
    delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "m": timedelta(minutes=n)}[unit]
    return (datetime.now(timezone.utc) - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

# Fallback pricing (USD per 1M tokens) when litellm is not installed.
# Covers models actively used across projects.  Updated 2026-09.
_FALLBACK_PRICES: dict[str, tuple[float, float]] = {  # (input/M, output/M)
    "gemini-2.0-flash":           (0.10, 0.40),
    "gemini-2.0-flash-lite":      (0.075, 0.30),
    "gemini-2.5-flash":           (0.30, 2.50),
    "gemini-2.5-flash-lite":      (0.10, 0.40),
    "gemini-2.5-pro":             (1.25, 10.00),
    "gemini-3-flash-preview":     (0.50, 3.00),
    "gemini-3.1-flash-lite":      (0.25, 1.50),
    "gemini-3.1-pro-preview":     (2.00, 12.00),
    "gemini-3.5-flash":           (1.50, 9.00),
    "gemini-3.5-flash-lite":      (0.30, 2.50),
    "gemini-3.8-flash":           (0.75, 3.75),
    "claude-fable-5-1":           (10.00, 50.00),
    "claude-opus-5":              (5.00, 25.00),
    "claude-sonnet-5":            (2.00, 10.00),
    "claude-sonnet-4-20250514":   (3.00, 15.00),
    "claude-haiku-4-5-20251001":  (1.00, 5.00),
    "gpt-5.6-sol":                (4.00, 20.00),
    "gpt-5.6-terra":              (2.00, 12.00),
    "gpt-5.6-luna":               (0.20, 1.20),
    "gpt-5.5":                    (5.00, 30.00),
    "gpt-5.4-mini":               (0.75, 4.50),
    "gpt-5.4-nano":               (0.20, 1.25),
    "gpt-4.1-mini":               (0.40, 1.60),
    "gpt-4.1-nano":               (0.10, 0.40),
}


# Cached-input prices (USD per 1M tokens): what the provider bills for the part
# of `prompt_tokens` it served from its prompt cache. Sources, both read on
# 2026-09-21: OpenAI https://developers.openai.com/api/docs/pricing (standard
# tier, "Cached input"); Gemini https://ai.google.dev/gemini-api/docs/pricing
# (paid tier, "Context caching", page dated 2026-09-16; gemini-2.0-flash from
# its earlier listing). Anthropic is left out on purpose: its cache has a
# separate write surcharge that a single cached-token count cannot express.
# A model absent here is billed at the full input price — never guessed.
_CACHED_INPUT_PRICES: dict[str, float] = {
    "gpt-5.6-sol":            0.40,
    "gpt-5.6-terra":          0.20,
    "gpt-5.6-luna":           0.02,
    "gpt-5.5":                0.50,
    "gpt-5.4-mini":           0.075,
    "gpt-5.4-nano":           0.02,
    "gpt-4.1-mini":           0.10,
    "gpt-4.1-nano":           0.025,
    "gemini-2.0-flash":       0.025,
    "gemini-2.5-flash":       0.03,
    "gemini-2.5-flash-lite":  0.01,
    "gemini-2.5-pro":         0.125,
    "gemini-3-flash-preview": 0.05,
    "gemini-3.1-flash-lite":  0.025,
    "gemini-3.1-pro-preview": 0.20,
    "gemini-3.5-flash":       0.15,
    "gemini-3.5-flash-lite":  0.03,
    "gemini-3.8-flash":       0.075,
}


def _format_ts(ts: str | datetime | None) -> str:
    if ts is None:
        moment = datetime.now(timezone.utc)
    else:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        moment = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fallback_cost(model: str, prompt_tokens: int,
                   completion_tokens: int) -> float | None:
    """Compute cost from built-in price table. Ignores any provider/ prefix."""
    prices = _FALLBACK_PRICES.get(model) or _FALLBACK_PRICES.get(model.rsplit("/", 1)[-1])
    if not prices:
        return None
    inp, out = prices
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int,
                 cached_tokens: int = 0) -> float | None:
    """Compute USD cost using litellm's pricing database (2,500+ models).

    Falls back to a built-in price table for common models when litellm
    is not installed.  Returns None if the model is unknown.
    """
    try:
        import litellm
        prompt_cost, compl_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        cost = float(prompt_cost + compl_cost)
    except Exception:
        cost = _fallback_cost(model, prompt_tokens, completion_tokens)
    if cost is None or not cached_tokens:
        return cost
    return max(0.0, cost - _cached_discount(model, prompt_tokens, cached_tokens))


def _cached_discount(model: str, prompt_tokens: int, cached_tokens: int) -> float:
    """USD to take off a full-input-price cost for the cached share of the prompt."""
    inp, _ = price_for(model, strict=False)
    cached_price = cached_input_price_for(model, strict=False)
    return min(cached_tokens, prompt_tokens) * max(0.0, inp - cached_price) / 1_000_000


class UnknownModelPriceError(ValueError):
    """Raised by `price_for()` when a model has no known per-token price and strict=True."""


def price_for(model: str, *, strict: bool = True) -> tuple[float, float]:
    """Return `(input_price_per_1M, output_price_per_1M)` USD for a model.

    Tries litellm's pricing database first, then the built-in
    `_FALLBACK_PRICES` table. Unlike `compute_cost`, which silently computes
    $0 for an unpriced model (a caller that doesn't check for `None` logs a
    real call as free), this raises by default. Otak carried a private price
    table 5-7x too low for months because nothing forced the question; this
    is the guard against a repeat. Pass `strict=False` to get `(0.0, 0.0)`
    with a logged warning instead of raising — e.g. for a best-effort budget
    estimate where a $0 fallback is an acceptable, visible degradation.
    """
    try:
        import litellm
        info = litellm.get_model_info(model)
        inp = info.get("input_cost_per_token")
        out = info.get("output_cost_per_token")
        if inp is not None and out is not None:
            return float(inp) * 1_000_000, float(out) * 1_000_000
    except Exception:
        pass
    prices = _FALLBACK_PRICES.get(model) or _FALLBACK_PRICES.get(model.rsplit("/", 1)[-1])
    if prices:
        return prices
    if strict:
        raise UnknownModelPriceError(
            f"no known price for model {model!r} — pass strict=False, or add "
            "it to cost_log._FALLBACK_PRICES if it's a real, priced model"
        )
    log.warning("no known price for model %r; treating as $0/1M (strict=False)", model)
    return (0.0, 0.0)


def cached_input_price_for(model: str, *, strict: bool = True) -> float:
    """USD per 1M *cached* input tokens for a model.

    litellm's `cache_read_input_token_cost` first, then `_CACHED_INPUT_PRICES`
    (OpenAI and Gemini list prices read 2026-09-21 — see the table's comment
    for the URLs). A priced model with no known cached rate returns its full
    input price, so an unknown discount is never invented; an unpriced model
    follows `price_for`'s `strict` behaviour.
    """
    try:
        import litellm
        cached = litellm.get_model_info(model).get("cache_read_input_token_cost")
        if cached is not None:
            return float(cached) * 1_000_000
    except Exception:
        pass
    known = _CACHED_INPUT_PRICES.get(model)
    if known is None:
        known = _CACHED_INPUT_PRICES.get(model.rsplit("/", 1)[-1])
    if known is not None:
        return known
    return price_for(model, strict=strict)[0]


def cost_for(model: str, prompt_tokens: int, completion_tokens: int,
             cached_tokens: int = 0, *, strict: bool = True) -> float:
    """USD for one call: fresh input, cached input and output each at its own rate.

    `prompt_tokens` is the provider's total input count and *includes*
    `cached_tokens` (the OpenAI `input_tokens` / Gemini `promptTokenCount`
    convention), so the cached share is billed at `cached_input_price_for`
    and only the remainder at the full input price.

    Ledger rows written before 2026-09-21 billed every input token at the
    full rate and therefore overstate calls that hit a provider prompt cache.
    They are left as written: `cached_tokens` is on each row, so the
    difference is recomputable, and a total that silently changed under a
    reader would be worse than one that is known to run high.
    """
    inp, out = price_for(model, strict=strict)
    cached = min(max(cached_tokens or 0, 0), prompt_tokens or 0)
    cached_price = cached_input_price_for(model, strict=False) if cached else 0.0
    return ((prompt_tokens - cached) * inp + cached * cached_price + completion_tokens * out) / 1_000_000


# ---------------------------------------------------------------------------
# Core logger
# ---------------------------------------------------------------------------

@dataclass
class CostRecord:
    """A single LLM cost record."""
    id: str
    ts: str
    project: str
    host: str
    model: str
    api_key_hint: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float
    script: str
    purpose: str
    metadata: dict
    cache_hit: bool = False
    outcome: str | None = None
    packet_id: str | None = None
    billing_mode: str = "billed"
    notional_cost_usd: float | None = None


class CostLog:
    """Central LLM cost logger backed by SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else _default_db_path()
        self._conn: sqlite3.Connection | None = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path), timeout=30, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        if str(self._db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(_SCHEMA)
        _migrate_columns(conn)
        conn.commit()
        self._conn = conn
        return conn

    def log(self, *, project: str, model: str,
            prompt_tokens: int = 0, completion_tokens: int = 0,
            cached_tokens: int = 0, cost_usd: float | None = None,
            api_key_hint: str = "", host: str | None = None,
            script: str = "", purpose: str = "",
            metadata: dict[str, Any] | None = None,
            cache_hit: bool = False, outcome: str | None = None,
            packet_id: str | None = None,
            billing_mode: str = "billed",
            notional_cost_usd: float | None = None,
            ts: str | datetime | None = None) -> CostRecord:
        """Log an LLM call.  If cost_usd is None, computes it via litellm.

        `ts` dates the row for a call that happened earlier (backfilling a
        project's own ledger into this one); omitted, the row is dated now.
        A datetime or an ISO-8601 string is accepted, a naive one is read as
        UTC, and either is stored in the ledger's UTC format so `since`
        filters and ordering keep working.

        `cache_hit`, `outcome`, and `packet_id` are optional ledger columns:
        `cache_hit` marks a `cached_call` response-cache hit (cost_usd should
        be 0 for those); `outcome` is normally set later via
        `record_outcome()` once the caller knows whether the result was used;
        `packet_id` links the row to a `cerebellum.packet.Packet` (reserved
        for that not-yet-built feature — nullable, no current writer).

        `billing_mode` is `"billed"` (default) or `"subscription"`; see
        `BILLING_MODES`. A subscription row must carry `cost_usd=0` — it is
        refused otherwise, because a notional figure that reached `cost_usd`
        would be indistinguishable from money in every existing total. Put the
        API-equivalent estimate in `notional_cost_usd`, or leave it `None` when
        the model has no known price.
        """
        if billing_mode not in BILLING_MODES:
            raise ValueError(
                f"billing_mode must be one of {sorted(BILLING_MODES)}, got {billing_mode!r}")
        if billing_mode == "subscription":
            if cost_usd is None:
                cost_usd = 0.0
            elif cost_usd:
                raise ValueError(
                    "a subscription row spends no money: pass cost_usd=0 and put the "
                    "API-equivalent figure in notional_cost_usd")

        if cost_usd is None:
            cost_usd = compute_cost(model, prompt_tokens, completion_tokens,
                                    cached_tokens) or 0.0

        record = CostRecord(
            id=uuid.uuid4().hex,
            ts=_format_ts(ts),
            project=project,
            host=host or _detect_host(),
            model=model,
            api_key_hint=api_key_hint,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            script=script,
            purpose=purpose,
            metadata=metadata or {},
            cache_hit=cache_hit,
            outcome=outcome,
            packet_id=packet_id,
            billing_mode=billing_mode,
            notional_cost_usd=notional_cost_usd,
        )

        conn = self._connect()
        conn.execute(
            """INSERT INTO llm_costs
               (id, ts, project, host, model, api_key_hint,
                prompt_tokens, completion_tokens, cached_tokens,
                cost_usd, script, purpose, metadata, cache_hit, outcome, packet_id,
                billing_mode, notional_cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.id, record.ts, record.project, record.host,
             record.model, record.api_key_hint,
             record.prompt_tokens, record.completion_tokens, record.cached_tokens,
             record.cost_usd, record.script, record.purpose,
             json.dumps(record.metadata), int(record.cache_hit),
             record.outcome, record.packet_id,
             record.billing_mode, record.notional_cost_usd),
        )
        conn.commit()
        return record

    def record_outcome(self, call_id: str, outcome: str, detail: str = "") -> bool:
        """Set the `outcome` of an existing ledger row (applied/no_op/rejected/held/error).

        This is the missing half of the ledger the audit called out: without
        it, cost per useful change can't be computed, only cost per call.
        `detail` (optional free text) is merged into the row's `metadata`
        JSON under `outcome_detail` rather than adding another column.
        Returns True if a row was found and updated.
        """
        conn = self._connect()
        if detail:
            row = conn.execute(
                "SELECT metadata FROM llm_costs WHERE id = ?", (call_id,)
            ).fetchone()
            if row is None:
                return False
            try:
                meta = json.loads(row["metadata"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            meta["outcome_detail"] = detail
            cur = conn.execute(
                "UPDATE llm_costs SET outcome = ?, metadata = ? WHERE id = ?",
                (outcome, json.dumps(meta), call_id),
            )
        else:
            cur = conn.execute(
                "UPDATE llm_costs SET outcome = ? WHERE id = ?", (outcome, call_id)
            )
        conn.commit()
        return cur.rowcount > 0

    def set_packet_id(self, call_id: str, packet_id: str) -> bool:
        """Link an existing ledger row to the `cerebellum.packet.Packet` it paid for.

        The column was reserved with no writer; `run_packets` is the writer.
        With it, `report --by purpose,outcome` can be narrowed to one packet's
        whole history — the dry run, the real call, the split halves after a
        truncation — which is what makes "did we already pay for this?"
        answerable before replanning a batch.
        """
        conn = self._connect()
        cur = conn.execute(
            "UPDATE llm_costs SET packet_id = ? WHERE id = ?", (packet_id, call_id))
        conn.commit()
        return cur.rowcount > 0

    # -----------------------------------------------------------------------
    # litellm callback
    # -----------------------------------------------------------------------

    def callback(self, project: str, host: str | None = None):
        """Return a litellm CustomLogger that auto-logs every completion.

        Usage:
            import litellm
            litellm.callbacks = [cost_log.callback("alif")]
        """
        import litellm as _litellm

        outer = self

        class _Callback(_litellm.CustomLogger):
            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                try:
                    slp = kwargs.get("standard_logging_object") or {}
                    model = slp.get("model") or kwargs.get("model", "unknown")
                    cost = slp.get("response_cost") or kwargs.get("response_cost", 0)

                    usage = getattr(response_obj, "usage", None)
                    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
                    ct = getattr(usage, "completion_tokens", 0) if usage else 0

                    api_key = kwargs.get("litellm_params", {}).get("api_key", "")
                    hint = api_key[-4:] if isinstance(api_key, str) and len(api_key) > 4 else ""

                    outer.log(
                        project=project,
                        model=model,
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        cost_usd=float(cost) if cost else None,
                        api_key_hint=hint,
                        host=host,
                        metadata={"litellm_callback": True},
                    )
                except Exception as exc:
                    log.warning("cost_log callback failed: %s", exc)

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                self.log_success_event(kwargs, response_obj, start_time, end_time)

        return _Callback()

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def query(self, *, days: int | None = None, since: str | None = None,
              project: str | None = None, host: str | None = None,
              model: str | None = None) -> list[sqlite3.Row]:
        """Query cost records with optional filters."""
        conn = self._connect()
        clauses, params = [], []
        if days:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            clauses.append("ts >= ?")
            params.append(cutoff)
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        if project:
            clauses.append("project = ?")
            params.append(project)
        if host:
            clauses.append("host = ?")
            params.append(host)
        if model:
            clauses.append("model LIKE ?")
            params.append(f"%{model}%")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return conn.execute(
            f"SELECT * FROM llm_costs{where} ORDER BY ts DESC", params
        ).fetchall()

    def summary(self, *, days: int | None = None, since: str | None = None,
                group_by: str = "project") -> list[dict]:
        """Aggregate costs grouped by project, model, host, or api_key_hint.

        `cost_usd` is real spend only. Subscription-covered calls contribute 0
        to it and their API-equivalent estimate to `notional_cost_usd`, which
        is reported beside it and never folded in — group by `billing_mode` to
        see the split.
        """
        valid = {"project", "model", "host", "api_key_hint", "script", "billing_mode"}
        if group_by not in valid:
            raise ValueError(f"group_by must be one of {valid}")

        conn = self._connect()
        clauses, params = [], []
        if days:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            clauses.append("ts >= ?")
            params.append(cutoff)
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = conn.execute(f"""
            SELECT {group_by} AS grp,
                   COUNT(*) AS calls,
                   SUM(prompt_tokens) AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens,
                   SUM(cached_tokens) AS cached_tokens,
                   SUM(cost_usd) AS cost_usd,
                   COALESCE(SUM(notional_cost_usd), 0) AS notional_cost_usd,
                   MIN(ts) AS first_call,
                   MAX(ts) AS last_call
            FROM llm_costs{where}
            GROUP BY {group_by}
            ORDER BY cost_usd DESC
        """, params).fetchall()
        return [dict(r) for r in rows]

    _MULTI_GROUP_COLUMNS = {
        "project", "model", "host", "api_key_hint", "script", "purpose",
        "outcome", "cache_hit", "billing_mode",
    }

    def multi_group_summary(self, *, by: list[str], since: str | None = None) -> list[dict]:
        """Aggregate costs grouped by any combination of columns, with an
        `outcome`-aware yield metric: `cost_per_applied` (cost_usd / count of
        rows with outcome='applied' in that group; None if the group has no
        applied rows). This is the report the audit was missing — without
        `outcome`, cost per useful change can't be computed, only cost per
        call. Also reports `cache_hit_rate` per group.
        """
        cols = [c.strip() for c in by if c.strip()]
        if not cols:
            raise ValueError("by must name at least one column")
        bad = set(cols) - self._MULTI_GROUP_COLUMNS
        if bad:
            raise ValueError(f"unknown group column(s) {sorted(bad)}; use one of "
                             f"{sorted(self._MULTI_GROUP_COLUMNS)}")

        conn = self._connect()
        clauses, params = [], []
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        group_cols = ", ".join(f"COALESCE({c}, '')" for c in cols)
        select_cols = ", ".join(f"COALESCE({c}, '') AS {c}" for c in cols)
        rows = conn.execute(f"""
            SELECT {select_cols},
                   COUNT(*) AS calls,
                   SUM(prompt_tokens) AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens,
                   SUM(cached_tokens) AS cached_tokens,
                   SUM(cost_usd) AS cost_usd,
                   COALESCE(SUM(notional_cost_usd), 0) AS notional_cost_usd,
                   AVG(cache_hit) AS cache_hit_rate,
                   SUM(CASE WHEN outcome = 'applied' THEN 1 ELSE 0 END) AS applied_count
            FROM llm_costs{where}
            GROUP BY {group_cols}
            ORDER BY cost_usd DESC
        """, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["cost_per_applied"] = (
                d["cost_usd"] / d["applied_count"] if d["applied_count"] else None
            )
            # A subscription group's `cost_per_applied` is honestly 0 — no money
            # moved — which makes an expensive agent loop look free next to a
            # billed one. This is the figure that makes the two comparable.
            d["notional_cost_per_applied"] = (
                d["notional_cost_usd"] / d["applied_count"] if d["applied_count"] else None
            )
            out.append(d)
        return out

    def total(self, *, days: int | None = None) -> float:
        """Total USD spend — real money only.

        Subscription-covered rows carry `cost_usd = 0`, so they cannot inflate
        this. `total_notional()` reports what they would have cost on the API.
        """
        return self._total("cost_usd", days)

    def total_notional(self, *, days: int | None = None) -> float:
        """API-equivalent USD for subscription-covered calls. Not spend.

        Report it beside `total()` under its own label; adding the two gives a
        number that is neither the bill nor the value of anything.
        """
        return self._total("notional_cost_usd", days)

    def _total(self, column: str, days: int | None) -> float:
        conn = self._connect()
        if days:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            row = conn.execute(
                f"SELECT COALESCE(SUM({column}), 0) FROM llm_costs WHERE ts >= ?",
                (cutoff,)).fetchone()
        else:
            row = conn.execute(
                f"SELECT COALESCE(SUM({column}), 0) FROM llm_costs").fetchone()
        return float(row[0])

    # -----------------------------------------------------------------------
    # Merge (for sync from remote)
    # -----------------------------------------------------------------------

    def merge_from(self, remote_db_path: str | Path) -> int:
        """Merge rows from a remote cost DB into this one.

        Uses INSERT OR IGNORE on the UUID primary key, so rows that already
        exist are skipped.  Returns number of new rows inserted.

        Only the columns both databases have are copied; the rest take their
        local defaults. A host still on an older limbic writes a narrower table,
        and `SELECT *` across the two then fails on the column count — which
        would strand every remote row rather than the one new field.
        """
        conn = self._connect()
        remote = str(remote_db_path)
        conn.execute("ATTACH DATABASE ? AS remote", (remote,))
        try:
            local_cols = [r["name"] for r in conn.execute("PRAGMA table_info(llm_costs)")]
            remote_cols = {r["name"] for r in conn.execute("PRAGMA remote.table_info(llm_costs)")}
            shared = [c for c in local_cols if c in remote_cols]
            if not shared:
                raise sqlite3.OperationalError(
                    f"{remote} has no llm_costs columns in common with the local ledger")
            missing = [c for c in local_cols if c not in remote_cols]
            if missing:
                log.info("merging %s without %s (older schema); local defaults apply",
                         remote, ", ".join(missing))
            cols = ", ".join(shared)
            cursor = conn.execute(f"""
                INSERT OR IGNORE INTO llm_costs ({cols})
                SELECT {cols} FROM remote.llm_costs
            """)
            count = cursor.rowcount
            conn.commit()
        finally:
            conn.execute("DETACH DATABASE remote")
        log.info("Merged %d new rows from %s", count, remote)
        return count

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

cost_log = CostLog()


def record_outcome(call_id: str, outcome: str, detail: str = "") -> bool:
    """Module-level convenience: `cost_log.record_outcome(...)` on the singleton."""
    return cost_log.record_outcome(call_id, outcome, detail)


# ---------------------------------------------------------------------------
# Sync helper
# ---------------------------------------------------------------------------

def sync_from_remote(host: str = "alif",
                     remote_db: str = "/opt/limbic-data/llm_costs.db") -> int:
    """Rsync the remote cost DB and merge into local. Returns new row count.

    Runs `PRAGMA wal_checkpoint(TRUNCATE)` on the remote DB over SSH before
    rsyncing. Without this, WAL-only writes (i.e. everything since the last
    passive checkpoint) stay in `<db>-wal` on the remote and never make it
    across, silently dropping hours of production rows.
    """
    import subprocess
    import tempfile

    # Force a full WAL checkpoint on the remote so all committed rows land in
    # the main DB file that rsync is about to pull.
    checkpoint = subprocess.run(
        ["ssh", host, f"sqlite3 {remote_db} 'PRAGMA wal_checkpoint(TRUNCATE);'"],
        capture_output=True, text=True,
    )
    if checkpoint.returncode != 0:
        log.warning("remote WAL checkpoint failed: %s", checkpoint.stderr.strip())
        # Continue anyway — stale sync is better than no sync.

    tmp = tempfile.mktemp(suffix=".db")
    result = subprocess.run(
        ["rsync", "-az", f"{host}:{remote_db}", tmp],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.warning("rsync failed: %s", result.stderr.strip())
        return -1
    cl = CostLog()
    n = cl.merge_from(tmp)
    os.unlink(tmp)
    return n


# ---------------------------------------------------------------------------
# Dashboard server (stdlib only — no Flask/datasette needed)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Cost Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { --bg: #f7f4ec; --ink: #2a2420; --muted: #6a6458; --accent: #8b2500;
          --rule: #e4dfd4; --card: #fff; --green: #2a7a4a; --blue: #4a6fa5; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'DM Sans', -apple-system, sans-serif; background: var(--bg);
         color: var(--ink); max-width: 1100px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .sync-btn { background: var(--accent); color: #fff; border: none; padding: 6px 16px;
              border-radius: 4px; cursor: pointer; font-size: 13px; float: right; margin-top: -38px; }
  .sync-btn:hover { opacity: 0.85; }
  .section-label { font-size: 11px; color: var(--muted); text-transform: uppercase;
                   letter-spacing: 0.08em; margin-bottom: 8px; font-weight: 600; }
  .totals { display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }
  .total-card { background: var(--card); border: 1px solid var(--rule); border-radius: 8px;
                padding: 16px 20px; flex: 1; min-width: 120px; }
  .total-card .label { font-size: 11px; color: var(--muted); text-transform: uppercase;
                       letter-spacing: 0.05em; }
  .total-card .value { font-size: 28px; font-weight: 600; margin-top: 4px; }
  .total-card .value.cost { color: var(--accent); }
  .total-card .value.cli { color: var(--blue); }
  .total-card.cli-card { border-color: #c8d8e8; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
  .panel { background: var(--card); border: 1px solid var(--rule); border-radius: 8px; padding: 20px; }
  .panel h2 { font-size: 14px; font-weight: 600; margin-bottom: 12px; }
  .panel h2 .badge { font-size: 10px; font-weight: 500; padding: 2px 6px; border-radius: 3px;
                     margin-left: 6px; vertical-align: middle; }
  .badge-api { background: #f5e6e2; color: var(--accent); }
  .badge-cli { background: #e2eaf5; color: var(--blue); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--muted); font-weight: 500; padding: 6px 8px;
       border-bottom: 1px solid var(--rule); font-size: 11px; text-transform: uppercase; }
  td { padding: 6px 8px; border-bottom: 1px solid var(--rule); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
  .controls select, .controls input { padding: 6px 10px; border: 1px solid var(--rule);
    border-radius: 4px; font-size: 13px; background: var(--card); }
  canvas { max-height: 260px; }
  .chart-panel { grid-column: 1 / -1; }
  .last-sync { font-size: 11px; color: var(--muted); }
  .spacer { height: 12px; }
  .wide { grid-column: 1 / -1; }
  .muted { color: var(--muted); }
  .warn { color: var(--accent); font-weight: 600; }
</style>
</head>
<body>
<h1>LLM Cost Dashboard</h1>
<div class="subtitle">
  <span id="db-path"></span> &middot; <span class="last-sync" id="last-sync"></span>
</div>
<button class="sync-btn" onclick="doSync()">Sync from Hetzner</button>

<div class="controls">
  <select id="days" onchange="reload()">
    <option value="1">Last 24h</option>
    <option value="7" selected>Last 7 days</option>
    <option value="30">Last 30 days</option>
    <option value="90">Last 90 days</option>
    <option value="0">All time</option>
  </select>
</div>

<div class="section-label">API Usage (billed)</div>
<div class="totals" id="api-totals"></div>
<div class="spacer"></div>
<div class="section-label">Claude CLI Usage (subscription)</div>
<div class="totals" id="cli-totals"></div>
<div class="spacer"></div>
<div class="section-label">Subscription Usage (notional — not billed)</div>
<div class="totals" id="sub-totals"></div>

<div class="grid">
  <div class="panel chart-panel">
    <h2>Daily API Cost</h2>
    <canvas id="daily-chart"></canvas>
  </div>
  <div class="panel">
    <h2>By Project <span class="badge badge-api">API</span></h2>
    <table id="by-project"></table>
  </div>
  <div class="panel">
    <h2>By Model <span class="badge badge-api">API</span></h2>
    <table id="by-model"></table>
  </div>
  <div class="panel">
    <h2>By API Key</h2>
    <table id="by-api-key"></table>
  </div>
  <div class="panel">
    <h2>By Host <span class="badge badge-api">API</span></h2>
    <table id="by-host"></table>
  </div>
  <div class="panel">
    <h2>CLI by Project <span class="badge badge-cli">CLI</span></h2>
    <table id="cli-by-project"></table>
  </div>
  <div class="panel">
    <h2>CLI by Model <span class="badge badge-cli">CLI</span></h2>
    <table id="cli-by-model"></table>
  </div>
  <div class="panel wide">
    <h2>CLI by Purpose <span class="badge badge-cli">CLI</span></h2>
    <table id="cli-by-purpose"></table>
  </div>
  <div class="panel">
    <h2>Subscription by Project <span class="badge badge-cli">NOTIONAL</span></h2>
    <table id="sub-by-project"></table>
  </div>
  <div class="panel">
    <h2>Subscription by Model <span class="badge badge-cli">NOTIONAL</span></h2>
    <table id="sub-by-model"></table>
  </div>
  <div class="panel wide">
    <h2>Subscription by Purpose <span class="badge badge-cli">NOTIONAL</span></h2>
    <table id="sub-by-purpose"></table>
  </div>
  <div class="panel wide">
    <h2>Recent Calls</h2>
    <table id="recent"></table>
  </div>
</div>

<script>
let dailyChart = null;
const $ = id => document.getElementById(id);

async function api(path) {
  const r = await fetch('/api/' + path);
  return r.json();
}

function fmt(n) {
  n = Number(n || 0);
  if (Math.abs(n) >= 1000) return '$' + Math.round(n).toLocaleString('en-US');
  if (Math.abs(n) >= 10) return '$' + n.toFixed(2);
  if (Math.abs(n) >= 1) return '$' + n.toFixed(3);
  if (n > 0) return '$' + n.toFixed(4);
  return '$0';
}
function fmtK(n) {
  n = Number(n || 0);
  if (Math.abs(n) >= 1000000) return (n/1000000).toFixed(1) + 'M';
  if (Math.abs(n) >= 1000) return (n/1000).toFixed(1) + 'k';
  return n.toLocaleString('en-US');
}
function warnIf(cond, value) { return cond ? `<span class="warn">${value}</span>` : value; }

function renderTable(id, rows, cols) {
  const el = $(id);
  let h = '<thead><tr>' + cols.map(c => `<th>${c[0]}</th>`).join('') + '</tr></thead><tbody>';
  for (const r of rows) {
    h += '<tr>' + cols.map(c => {
      const v = c[1](r);
      return `<td class="${c[2]||''}">${v}</td>`;
    }).join('') + '</tr>';
  }
  el.innerHTML = h + '</tbody>';
}

async function reload() {
  const days = $('days').value;
  const d = await api('summary?days=' + days);

  // API totals
  $('api-totals').innerHTML = `
    <div class="total-card"><div class="label">API Cost</div><div class="value cost">${fmt(d.api.cost_usd)}</div></div>
    <div class="total-card"><div class="label">API Calls</div><div class="value">${d.api.calls.toLocaleString()}</div></div>
    <div class="total-card"><div class="label">Prompt Tokens</div><div class="value">${fmtK(d.api.prompt_tokens)}</div></div>
    <div class="total-card"><div class="label">Completion Tokens</div><div class="value">${fmtK(d.api.completion_tokens)}</div></div>
  `;

  // CLI totals
  $('cli-totals').innerHTML = `
    <div class="total-card cli-card"><div class="label">Subscription Value</div><div class="value cli">${fmt(d.cli.cost_usd)}</div></div>
    <div class="total-card cli-card"><div class="label">Sessions</div><div class="value cli">${d.cli.sessions.toLocaleString()}</div></div>
    <div class="total-card cli-card"><div class="label">Calls</div><div class="value cli">${d.cli.calls.toLocaleString()}</div></div>
    <div class="total-card cli-card"><div class="label">Visible Tokens</div><div class="value cli">${fmtK(d.cli.visible_tokens)}</div></div>
    <div class="total-card cli-card"><div class="label">Cached Tokens</div><div class="value cli">${fmtK(d.cli.cached_tokens)}</div></div>
    <div class="total-card cli-card"><div class="label">Total Tokens</div><div class="value cli">${fmtK(d.cli.total_tokens)}</div></div>
    <div class="total-card cli-card"><div class="label">Total Duration</div><div class="value cli">${d.cli.duration_min}m</div></div>
  `;

  // Subscription totals. `notional_usd` is what these tokens would have cost
  // on the API — it is deliberately not added to the API total anywhere.
  const sub = d.subscription || {calls: 0, notional_usd: 0, prompt_tokens: 0,
    completion_tokens: 0, cached_tokens: 0, total_tokens: 0, unpriced_calls: 0,
    by_project: [], by_model: [], by_purpose: []};
  $('sub-totals').innerHTML = `
    <div class="total-card cli-card"><div class="label">Notional (not billed)</div><div class="value cli">${fmt(sub.notional_usd)}</div></div>
    <div class="total-card cli-card"><div class="label">Calls</div><div class="value cli">${sub.calls.toLocaleString()}</div></div>
    <div class="total-card cli-card"><div class="label">Input Tokens</div><div class="value cli">${fmtK(sub.prompt_tokens)}</div></div>
    <div class="total-card cli-card"><div class="label">Cached Input</div><div class="value cli">${fmtK(sub.cached_tokens)}</div></div>
    <div class="total-card cli-card"><div class="label">Output Tokens</div><div class="value cli">${fmtK(sub.completion_tokens)}</div></div>
    <div class="total-card cli-card"><div class="label">Unpriced Calls</div><div class="value cli">${sub.unpriced_calls.toLocaleString()}</div></div>
  `;

  $('db-path').textContent = d.db_path;
  $('last-sync').textContent = 'Last sync: ' + (d.last_sync || 'never');

  const costCols = [
    ['Name', r => r.grp, ''],
    ['Calls', r => r.calls.toLocaleString(), 'num'],
    ['Cost', r => fmt(r.cost_usd), 'num'],
  ];
  renderTable('by-project', d.api.by_project, costCols);
  renderTable('by-model', d.api.by_model, costCols);
  renderTable('by-host', d.api.by_host, costCols);

  // By API key
  renderTable('by-api-key', d.api.by_key, [
    ['Key', r => r.grp || '(none)', ''],
    ['Calls', r => r.calls.toLocaleString(), 'num'],
    ['Cost', r => fmt(r.cost_usd), 'num'],
  ]);

  // CLI tables
  const subCols = [
    ['Name', r => r.grp, ''],
    ['Calls', r => r.calls.toLocaleString(), 'num'],
    ['Input', r => fmtK(r.prompt_tokens), 'num'],
    ['Cached', r => fmtK(r.cached_tokens), 'num'],
    ['Output', r => fmtK(r.completion_tokens), 'num'],
    ['Notional', r => fmt(r.notional_usd), 'num'],
  ];
  renderTable('sub-by-project', sub.by_project, subCols);
  renderTable('sub-by-model', sub.by_model, subCols);
  renderTable('sub-by-purpose', sub.by_purpose, subCols);

  renderTable('cli-by-project', d.cli.by_project, [
    ['Name', r => r.grp, ''],
    ['Sessions', r => r.sessions.toLocaleString(), 'num'],
    ['Visible', r => fmtK(r.visible_tokens), 'num'],
    ['Cached', r => fmtK(r.cached_tokens), 'num'],
    ['Total', r => fmtK(r.total_tokens), 'num'],
    ['Sub Value', r => fmt(r.cost_usd), 'num'],
  ]);
  renderTable('cli-by-model', d.cli.by_model, [
    ['Model', r => r.grp, ''],
    ['Calls', r => r.calls.toLocaleString(), 'num'],
    ['Visible', r => fmtK(r.visible_tokens), 'num'],
    ['Cached', r => fmtK(r.cached_tokens), 'num'],
    ['Total', r => fmtK(r.total_tokens), 'num'],
    ['Sub Value', r => fmt(r.cost_usd), 'num'],
  ]);
  renderTable('cli-by-purpose', d.cli.by_purpose, [
    ['Purpose', r => r.grp || '(none)', ''],
    ['Sessions', r => r.sessions.toLocaleString(), 'num'],
    ['Calls', r => r.calls.toLocaleString(), 'num'],
    ['Visible', r => fmtK(r.visible_tokens), 'num'],
    ['Cached', r => warnIf(r.avg_cached_tokens >= 50000, fmtK(r.cached_tokens)), 'num'],
    ['Avg Cached', r => warnIf(r.avg_cached_tokens >= 10000, fmtK(r.avg_cached_tokens)), 'num'],
    ['Avg Turns', r => Number(r.avg_turns || 0).toFixed(1), 'num'],
    ['Sub Value', r => fmt(r.cost_usd), 'num'],
  ]);

  // Recent calls
  const recent = await api('recent?days=' + days);
  renderTable('recent', recent.slice(0, 20), [
    ['Time', r => r.ts.slice(5, 16).replace('T', ' '), ''],
    ['Src', r => r.billing_mode === 'subscription' ? 'SUB' : (r.script === 'claude-cli' ? 'CLI' : 'API'), ''],
    ['Project', r => r.project, ''],
    ['Purpose', r => r.purpose || '', ''],
    ['Model', r => r.model.replace('gemini/', ''), ''],
    ['Visible', r => fmtK(r.visible_tokens), 'num'],
    ['Cached', r => warnIf(r.cached_tokens >= 10000, fmtK(r.cached_tokens)), 'num'],
    ['Total', r => fmtK(r.total_tokens), 'num'],
    ['Cost', r => fmt(r.cost_usd), 'num'],
  ]);

  // Daily chart (API only)
  const daily = await api('daily?days=' + days);
  if (dailyChart) dailyChart.destroy();
  const projects = [...new Set(daily.flatMap(d => Object.keys(d.projects)))];
  const colors = ['#8b2500','#2a7a4a','#4a6fa5','#d4a056','#7a4a8b','#5a8a6a','#a05040'];
  dailyChart = new Chart($('daily-chart'), {
    type: 'bar',
    data: {
      labels: daily.map(d => d.date.slice(5)),
      datasets: projects.map((p, i) => ({
        label: p,
        data: daily.map(d => d.projects[p] || 0),
        backgroundColor: colors[i % colors.length],
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { stacked: true }, y: { stacked: true, ticks: { callback: v => fmt(v) } } },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } } },
    },
  });
}

async function doSync() {
  const btn = document.querySelector('.sync-btn');
  btn.textContent = 'Syncing...';
  btn.disabled = true;
  try {
    const r = await fetch('/api/sync', { method: 'POST' });
    const d = await r.json();
    btn.textContent = d.new_rows >= 0 ? `Synced (${d.new_rows} new)` : 'Sync failed';
    if (d.new_rows >= 0) reload();
  } catch { btn.textContent = 'Sync failed'; }
  setTimeout(() => { btn.textContent = 'Sync from Hetzner'; btn.disabled = false; }, 3000);
}

reload();
</script>
</body>
</html>
"""


def _build_summary(cl: CostLog, days: int | None) -> dict:
    """Build the split API/CLI summary payload for the dashboard."""
    conn = cl._connect()
    clauses, params = [], []
    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        clauses.append("ts >= ?")
        params.append(cutoff)
    # Three disjoint buckets. `billed` is the only one that is money: a
    # subscription row lands in `sub_where` and is kept out of every API
    # aggregate, so the "API Cost" headline stays a bill.
    billed = "COALESCE(billing_mode, 'billed') != 'subscription'"
    api_where = " WHERE " + " AND ".join(clauses + [billed, "script != 'claude-cli'"])
    cli_where = " WHERE " + " AND ".join(clauses + [billed, "script = 'claude-cli'"])
    sub_where = " WHERE " + " AND ".join(clauses + ["COALESCE(billing_mode, 'billed') = 'subscription'"])

    def _grouped(where, params, group_col):
        rows = conn.execute(f"""
            SELECT {group_col} AS grp, COUNT(*) AS calls,
                   SUM(prompt_tokens) AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens,
                   SUM(cost_usd) AS cost_usd
            FROM llm_costs{where}
            GROUP BY {group_col} ORDER BY cost_usd DESC
        """, params).fetchall()
        return [dict(r) for r in rows]

    # API aggregates
    api_row = conn.execute(f"""
        SELECT COUNT(*) AS calls,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
               COALESCE(SUM(cost_usd), 0) AS cost_usd
        FROM llm_costs{api_where}
    """, params).fetchone()

    # CLI aggregates
    cli_row = conn.execute(f"""
        SELECT COUNT(*) AS calls,
               COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS visible_tokens,
               COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
               COALESCE(SUM(prompt_tokens + completion_tokens + cached_tokens), 0) AS total_tokens,
               COALESCE(SUM(cost_usd), 0) AS cost_usd,
               COUNT(DISTINCT json_extract(metadata, '$.session_id')) AS sessions,
               COALESCE(SUM(json_extract(metadata, '$.duration_ms')), 0) AS duration_ms
        FROM llm_costs{cli_where}
    """, params).fetchone()

    # CLI grouped by project (with session counts)
    cli_by_project = conn.execute(f"""
        SELECT project AS grp, COUNT(*) AS calls,
               COUNT(DISTINCT json_extract(metadata, '$.session_id')) AS sessions,
               SUM(prompt_tokens + completion_tokens) AS visible_tokens,
               SUM(cached_tokens) AS cached_tokens,
               SUM(prompt_tokens + completion_tokens + cached_tokens) AS total_tokens,
               SUM(cost_usd) AS cost_usd
        FROM llm_costs{cli_where}
        GROUP BY project ORDER BY cost_usd DESC
    """, params).fetchall()

    # CLI grouped by model
    cli_by_model = conn.execute(f"""
        SELECT model AS grp, COUNT(*) AS calls,
               SUM(prompt_tokens + completion_tokens) AS visible_tokens,
               SUM(cached_tokens) AS cached_tokens,
               SUM(prompt_tokens + completion_tokens + cached_tokens) AS total_tokens,
               SUM(cost_usd) AS cost_usd
        FROM llm_costs{cli_where}
        GROUP BY model ORDER BY cost_usd DESC
    """, params).fetchall()

    # CLI grouped by purpose.  This is the main view for finding inefficient
    # workflows: many one-turn sessions with high cached-token reads.
    cli_by_purpose = conn.execute(f"""
        SELECT COALESCE(purpose, '') AS grp,
               COUNT(*) AS calls,
               COUNT(DISTINCT json_extract(metadata, '$.session_id')) AS sessions,
               SUM(prompt_tokens + completion_tokens) AS visible_tokens,
               SUM(cached_tokens) AS cached_tokens,
               SUM(prompt_tokens + completion_tokens + cached_tokens) AS total_tokens,
               AVG(cached_tokens) AS avg_cached_tokens,
               AVG(prompt_tokens + completion_tokens) AS avg_visible_tokens,
               AVG(json_extract(metadata, '$.num_turns')) AS avg_turns,
               SUM(cost_usd) AS cost_usd
        FROM llm_costs{cli_where}
        GROUP BY COALESCE(purpose, '')
        ORDER BY cost_usd DESC
    """, params).fetchall()

    # Subscription aggregates (Codex under a ChatGPT plan, and anything else
    # that logs billing_mode='subscription'). `notional_usd` is an
    # API-equivalent estimate, never added to the API total.
    sub_row = conn.execute(f"""
        SELECT COUNT(*) AS calls,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
               COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
               COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS total_tokens,
               COALESCE(SUM(notional_cost_usd), 0) AS notional_usd,
               SUM(CASE WHEN notional_cost_usd IS NULL THEN 1 ELSE 0 END) AS unpriced_calls
        FROM llm_costs{sub_where}
    """, params).fetchone()

    def _sub_grouped(group_col):
        rows = conn.execute(f"""
            SELECT {group_col} AS grp, COUNT(*) AS calls,
                   SUM(prompt_tokens) AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens,
                   SUM(cached_tokens) AS cached_tokens,
                   COALESCE(SUM(notional_cost_usd), 0) AS notional_usd
            FROM llm_costs{sub_where}
            GROUP BY {group_col} ORDER BY notional_usd DESC
        """, params).fetchall()
        return [dict(r) for r in rows]

    return {
        "api": {
            "cost_usd": api_row["cost_usd"],
            "calls": api_row["calls"],
            "prompt_tokens": api_row["prompt_tokens"],
            "completion_tokens": api_row["completion_tokens"],
            "by_project": _grouped(api_where, params, "project"),
            "by_model": _grouped(api_where, params, "model"),
            "by_host": _grouped(api_where, params, "host"),
            "by_key": _grouped(api_where, params, "api_key_hint"),
        },
        "cli": {
            "sessions": cli_row["sessions"],
            "calls": cli_row["calls"],
            "visible_tokens": cli_row["visible_tokens"],
            "cached_tokens": cli_row["cached_tokens"],
            "total_tokens": cli_row["total_tokens"],
            "cost_usd": cli_row["cost_usd"],
            "duration_min": round(cli_row["duration_ms"] / 60_000, 1),
            "by_project": [dict(r) for r in cli_by_project],
            "by_model": [dict(r) for r in cli_by_model],
            "by_purpose": [dict(r) for r in cli_by_purpose],
        },
        "subscription": {
            "calls": sub_row["calls"],
            "prompt_tokens": sub_row["prompt_tokens"],
            "completion_tokens": sub_row["completion_tokens"],
            "cached_tokens": sub_row["cached_tokens"],
            "total_tokens": sub_row["total_tokens"],
            "notional_usd": sub_row["notional_usd"],
            "unpriced_calls": sub_row["unpriced_calls"] or 0,
            "by_project": _sub_grouped("project"),
            "by_model": _sub_grouped("model"),
            "by_purpose": _sub_grouped("COALESCE(purpose, '')"),
        },
        "db_path": str(cl.db_path),
        "last_sync": _last_sync_time(),
    }


def _serve_dashboard(port: int = 8042, open_browser: bool = True):
    """Serve the built-in cost dashboard on localhost."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse
    import socket

    # Check if port is already in use (previous instance still running)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            url = f"http://localhost:{port}"
            print(f"  Dashboard already running at {url}")
            if open_browser:
                import webbrowser
                webbrowser.open(url)
            return

    cl = CostLog()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass  # quiet

        def _json(self, data, status=200):
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            qs = dict(urllib.parse.parse_qsl(parsed.query))
            days = int(qs.get("days", 7)) or None

            if parsed.path == "/":
                body = _DASHBOARD_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)

            elif parsed.path == "/api/summary":
                self._json(_build_summary(cl, days))

            elif parsed.path == "/api/recent":
                rows = cl.query(days=days)[:50]
                recent = []
                for row in rows:
                    d = dict(row)
                    d["visible_tokens"] = d["prompt_tokens"] + d["completion_tokens"]
                    d["total_tokens"] = d["visible_tokens"] + d["cached_tokens"]
                    try:
                        meta = json.loads(d.get("metadata") or "{}")
                    except json.JSONDecodeError:
                        meta = {}
                    d["session_id"] = meta.get("session_id")
                    d["debug_file_bytes"] = meta.get("debug_file_bytes")
                    d["failed"] = bool(meta.get("failed"))
                    recent.append(d)
                self._json(recent)

            elif parsed.path == "/api/daily":
                conn = cl._connect()
                # "Daily API Cost" is a spend chart, so subscription rows —
                # whose cost_usd is 0 by construction — stay out of it rather
                # than adding flat zero-height series per project.
                clauses = ["script != 'claude-cli'",
                           "COALESCE(billing_mode, 'billed') != 'subscription'"]
                params = []
                if days:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ")
                    clauses.append("ts >= ?")
                    params.append(cutoff)
                where = " WHERE " + " AND ".join(clauses)
                rows = conn.execute(f"""
                    SELECT substr(ts, 1, 10) AS date, project,
                           SUM(cost_usd) AS cost
                    FROM llm_costs{where}
                    GROUP BY date, project
                    ORDER BY date
                """, params).fetchall()
                daily = {}
                for r in rows:
                    d = r["date"]
                    if d not in daily:
                        daily[d] = {"date": d, "projects": {}}
                    daily[d]["projects"][r["project"]] = round(r["cost"], 6)
                self._json(list(daily.values()))

            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/sync":
                n = sync_from_remote()
                self._json({"new_rows": n, "total": cl.total()})
            else:
                self.send_error(404)

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  DB: {cl.db_path}")
    print(f"  Press Ctrl+C to stop\n")
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def _last_sync_time() -> str | None:
    """Check when the last sync from hetzner happened (most recent hetzner row)."""
    cl = CostLog()
    conn = cl._connect()
    row = conn.execute(
        "SELECT MAX(ts) FROM llm_costs WHERE host = 'hetzner'"
    ).fetchone()
    return row[0] if row and row[0] else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m limbic.cerebellum.cost_log",
        description="LLM cost tracking — report, sync, and dashboard",
    )
    sub = parser.add_subparsers(dest="cmd")

    # -- report --
    rpt = sub.add_parser("report", help="Show cost summary")
    rpt.add_argument("--days", type=int, default=7, help="Look-back window (default 7)")
    rpt.add_argument("--group-by", default="project",
                     choices=["project", "model", "host", "api_key_hint", "script"])
    rpt.add_argument("--by", default=None,
                     help="Comma-separated columns for a multi-column report, e.g. "
                          "project,purpose,outcome. Overrides --group-by/--days.")
    rpt.add_argument("--since", default=None,
                     help="Relative window for --by, e.g. 30d, 12h, 45m")
    rpt.add_argument("--json", action="store_true", help="JSON output")

    # -- sync --
    syn = sub.add_parser("sync", help="Sync costs from remote host")
    syn.add_argument("--host", default="alif", help="SSH host alias (default: alif)")
    syn.add_argument("--remote-db", default="/opt/limbic-data/llm_costs.db",
                     help="Remote DB path")

    # -- dashboard --
    dash = sub.add_parser("dashboard", help="Launch web dashboard")
    dash.add_argument("--port", type=int, default=8042)
    dash.add_argument("--no-open", action="store_true", help="Don't open browser")

    # -- datasette --
    ds = sub.add_parser("datasette", help="Launch datasette web UI")
    ds.add_argument("--port", type=int, default=8043)

    args = parser.parse_args()

    if args.cmd == "report" and args.by:
        cl = CostLog()
        cols = [c.strip() for c in args.by.split(",") if c.strip()]
        since = _parse_since(args.since) if args.since else None
        rows = cl.multi_group_summary(by=cols, since=since)
        if args.json:
            print(json.dumps({"by": cols, "since": args.since, "rows": rows},
                             indent=2, default=str))
            return
        if not rows:
            print("  (no data)")
            return
        window = f"since {args.since}" if args.since else "all time"
        print(f"\n  LLM costs by {', '.join(cols)} — {window}")
        print(f"  DB: {cl.db_path}\n")
        col_w = 18
        header = "".join(f"{c:<{col_w}}" for c in cols)
        print(f"  {header}{'Calls':>8} {'Total tok':>12} {'Cache%':>8} {'Cost':>10} {'$/applied':>10}")
        print(f"  {'-'*col_w*len(cols)} {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
        for r in rows:
            total_tokens = r["prompt_tokens"] + r["completion_tokens"] + r["cached_tokens"]
            grp = "".join(f"{str(r[c]) or '(none)':<{col_w}}" for c in cols)
            cache_pct = 100 * (r["cache_hit_rate"] or 0.0)
            per_applied = f"${r['cost_per_applied']:.4f}" if r["cost_per_applied"] is not None else "-"
            print(f"  {grp}{r['calls']:>8} {total_tokens:>12,} {cache_pct:>7.1f}% "
                  f"${r['cost_usd']:>9.4f} {per_applied:>10}")
        print()
        return

    if args.cmd == "report":
        cl = CostLog()
        if args.json:
            rows = cl.summary(days=args.days, group_by=args.group_by)
            print(json.dumps({"days": args.days, "group_by": args.group_by,
                              "total_usd": cl.total(days=args.days),
                              "rows": rows}, indent=2, default=str))
        else:
            total = cl.total(days=args.days)
            print(f"\n  LLM costs — last {args.days} days  (total: ${total:.4f})")
            print(f"  DB: {cl.db_path}\n")
            rows = cl.summary(days=args.days, group_by=args.group_by)
            if not rows:
                print("  (no data)")
                return
            print(f"  {'Group':<30} {'Calls':>7} {'Prompt':>10} {'Compl':>10} {'Cached':>10} {'Total':>10} {'Cost':>10}")
            print(f"  {'-'*30} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for r in rows:
                total_tokens = r["prompt_tokens"] + r["completion_tokens"] + r["cached_tokens"]
                print(f"  {r['grp']:<30} {r['calls']:>7} {r['prompt_tokens']:>10,}"
                      f" {r['completion_tokens']:>10,} {r['cached_tokens']:>10,}"
                      f" {total_tokens:>10,} ${r['cost_usd']:>9.4f}")
            print()

    elif args.cmd == "sync":
        print(f"Syncing from {args.host}:{args.remote_db} ...")
        n = sync_from_remote(args.host, args.remote_db)
        if n < 0:
            print("Sync failed.", file=sys.stderr)
            sys.exit(1)
        print(f"Merged {n} new rows.  Total: ${CostLog().total():.4f}")

    elif args.cmd == "dashboard":
        _serve_dashboard(port=args.port, open_browser=not args.no_open)

    elif args.cmd == "datasette":
        db = str(CostLog().db_path)
        print(f"Launching datasette on port {args.port} for {db}")
        subprocess.run(["datasette", db, "--port", str(args.port)])

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
