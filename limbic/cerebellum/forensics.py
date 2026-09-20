"""Token-usage forensics over Claude Code and Codex session transcripts.

Every `cached_call` / `cost_log` row covers calls made *through* limbic.
Interactive agent sessions (a human or a coordinator driving Claude Code or
Codex directly) are invisible to that ledger, and they are where most spend
actually happens — this module is the read-only forensic layer over those
session JSONL files, ported from the prototype scripts in
`~/src/research/llm-pipeline-audit/data/` (20 Sep 2026 audit).

Two counting rules learned there, both non-obvious and both silently wrong
if skipped:

1. **Never trust a session's cumulative token counter.** Codex reports a
   running `total_token_usage` on every `token_count` event. A forked
   subagent's counter starts already carrying its parent's cumulative count
   (`thread_source: "subagent"`, spawned by thread-fork), so reading the
   *last* cumulative value overcounts — the first pass at this ledger came in
   at otak 2.66B / kdp-editions 6.7B input tokens; the corrected figures are
   0.85B / 1.06B. A *resumed* session can go the other way: its cumulative
   counter resets even though the replayed history still costs real input
   tokens on the next request. The fix in both directions is the same: sum
   the **delta** (`last_token_usage`) over events whose cumulative total
   actually changed (dedup), skipping a leading zero-usage event.
2. **Claude Code JSONL repeats the same `message.id`** across several lines
   while a response streams — summing `message.usage` without deduping by
   id multiplies token counts by however many chunks the message took.
3. **Claude Code subagent transcripts are not inline.** A Task-tool subagent's
   turns are NOT `isSidechain: true` lines mixed into the parent session's own
   `<session-id>.jsonl` — they live in separate files under
   `<session-id>/subagents/agent-*.jsonl` (each line still carries
   `isSidechain: true` and an `agentId`). A scan that only reads the main
   transcript will report zero subagent tokens even on a session that spawned
   a dozen of them. Each subagent's own opening turn size (input + cache
   creation + cache read of its first request) is its "entrance fee" — the
   cost of establishing its context before it does any useful work; the audit
   measured a 49.8K median.

Usage:

    from limbic.cerebellum.forensics import scan_codex_sessions, scan_claude_sessions

    codex_sessions = scan_codex_sessions(since=parse_since("30d"))
    claude_sessions = scan_claude_sessions(since=parse_since("30d"))

CLI:

    python -m limbic.cerebellum.forensics codex --since 30d --project-by cwd
    python -m limbic.cerebellum.forensics claude --since 30d
    python -m limbic.cerebellum.forensics codex --session <file> --attrib
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

DEFAULT_CODEX_ROOT = Path.home() / ".codex" / "sessions"
DEFAULT_CLAUDE_ROOT = Path.home() / ".claude" / "projects"

# Project names checked against cwd / mentioned paths. Extend via
# `known_projects=` on the scan functions or `--known-project` on the CLI —
# this default list is this machine's ~/src layout, not a general contract.
DEFAULT_KNOWN_PROJECTS = [
    "skard", "nrk", "otak", "petrarca", "limbic", "mdg", "dragoman", "alif",
    "kdp-editions", "kulturperler",
]

_RESEARCH_DIR_RE = re.compile(r"/src/research/([a-z0-9][a-z0-9-]*)")


def parse_since(spec: str) -> datetime:
    """Parse a relative window like "30d" / "12h" / "45m" into a UTC datetime."""
    spec = spec.strip()
    m = re.match(r"^(\d+)\s*([dhm]?)$", spec)
    if not m:
        raise ValueError(f"can't parse --since {spec!r}; use e.g. 30d, 12h, 45m")
    n, unit = int(m.group(1)), (m.group(2) or "d")
    delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "m": timedelta(minutes=n)}[unit]
    return datetime.now(timezone.utc) - delta


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Project assignment
# ---------------------------------------------------------------------------

def project_from_cwd(cwd: str | None, known_projects: list[str] = DEFAULT_KNOWN_PROJECTS) -> str | None:
    """Assign a project from a session's cwd. Cheap, and right most of the time."""
    if not cwd:
        return None
    m = _RESEARCH_DIR_RE.search(cwd)
    if m:
        return m.group(1)
    normalized = cwd.rstrip("/") + "/"
    for name in known_projects:
        if f"/{name}/" in normalized:
            return name
    return None


def project_from_mentioned_paths(
    raw_text: str, known_projects: list[str] = DEFAULT_KNOWN_PROJECTS, *, min_mentions: int = 5,
) -> str | None:
    """Assign a project by counting `/name/` path fragments mentioned anywhere in
    the session (tool calls, file reads). Slower, and a fallback for sessions
    whose cwd doesn't reflect where the real work happened. Returns None below
    `min_mentions`, matching the audit's threshold for avoiding noise from one
    stray mention (classify.py used 5)."""
    counts = {name: len(re.findall(f"/{re.escape(name)}/", raw_text)) for name in known_projects}
    research_hits = Counter(m for m in _RESEARCH_DIR_RE.findall(raw_text))
    if research_hits:
        slug, n = research_hits.most_common(1)[0]
        counts[slug] = counts.get(slug, 0) + n
    if not counts or max(counts.values()) < min_mentions:
        return None
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------

@dataclass
class CodexSessionStats:
    path: Path
    start: datetime | None = None
    cwd: str | None = None
    thread_source: str | None = None  # "user" | "subagent" | "automation" | ...
    model: str | None = None
    requests: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    max_context: int = 0
    requests_over_150k: int = 0
    project_by_cwd: str | None = None
    project_by_paths: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cached_tokens + self.output_tokens

    @property
    def project_disagreement(self) -> bool:
        return bool(
            self.project_by_paths
            and self.project_by_cwd
            and self.project_by_paths != self.project_by_cwd
        )


def scan_codex_session(
    path: str | Path, known_projects: list[str] = DEFAULT_KNOWN_PROJECTS,
) -> CodexSessionStats:
    """Parse one Codex rollout JSONL file.

    Sums `last_token_usage` over `token_count` events whose cumulative
    `total_token_usage` changed (dedup), skipping a leading zero-usage event —
    the counting rule from `codex_ledger_v2.py`. See module docstring.
    """
    path = Path(path)
    cwd = thread_source = model = None
    start: datetime | None = None
    prev_total = None
    n = input_tokens = cached_tokens = output_tokens = reasoning_tokens = 0
    max_context = 0
    over_150k = 0
    raw_chunks: list[str] = []

    with open(path, errors="ignore") as f:
        for line in f:
            raw_chunks.append(line)
            head = line[:300]
            if '"session_meta"' in head:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = d.get("payload", {})
                cwd = p.get("cwd", cwd)
                thread_source = p.get("thread_source", thread_source)
                start = _parse_ts(p.get("timestamp")) or start
                continue
            if '"turn_context"' in head:
                try:
                    p = json.loads(line).get("payload", {})
                except json.JSONDecodeError:
                    continue
                model = p.get("model", model)
                continue
            if '"token_count"' not in head:
                continue
            try:
                info = (json.loads(line).get("payload") or {}).get("info") or {}
            except json.JSONDecodeError:
                continue
            total = info.get("total_token_usage")
            if not total or total == prev_total:
                continue
            prev_total = total
            lt = info.get("last_token_usage") or {}
            if not lt.get("input_tokens") and not lt.get("output_tokens"):
                continue
            n += 1
            input_tokens += lt.get("input_tokens", 0)
            cached_tokens += lt.get("cached_input_tokens", 0)
            output_tokens += lt.get("output_tokens", 0)
            reasoning_tokens += lt.get("reasoning_output_tokens", 0)
            max_context = max(max_context, lt.get("input_tokens", 0))
            if lt.get("input_tokens", 0) > 150_000:
                over_150k += 1

    stats = CodexSessionStats(
        path=path, start=start, cwd=cwd, thread_source=thread_source, model=model,
        requests=n, input_tokens=input_tokens, cached_tokens=cached_tokens,
        output_tokens=output_tokens, reasoning_tokens=reasoning_tokens,
        max_context=max_context, requests_over_150k=over_150k,
    )
    stats.project_by_cwd = project_from_cwd(cwd, known_projects)
    stats.project_by_paths = project_from_mentioned_paths("".join(raw_chunks), known_projects)
    return stats


def scan_codex_sessions(
    root: str | Path = DEFAULT_CODEX_ROOT, *,
    since: datetime | None = None,
    known_projects: list[str] = DEFAULT_KNOWN_PROJECTS,
) -> list[CodexSessionStats]:
    """Scan every Codex rollout JSONL under `root` (default `~/.codex/sessions`),
    laid out as `<root>/YYYY/MM/DD/*.jsonl`."""
    root = Path(root)
    out = []
    for f in sorted(root.glob("*/*/*/*.jsonl")):
        stats = scan_codex_session(f, known_projects)
        if since and stats.start and stats.start < since:
            continue
        out.append(stats)
    return out


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

@dataclass
class SubagentStats:
    path: Path
    agent_id: str | None = None
    model: str | None = None
    first_turn_context: int = 0  # "entrance fee": input+cache_creation+cache_read of its first request
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class ClaudeSessionStats:
    path: Path
    start: datetime | None = None
    cwd: str | None = None
    # model -> {"requests":..,"input_tokens":..,"cache_creation_tokens":..,"cache_read_tokens":..,"output_tokens":..}
    main_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    sidechain_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    subagents: list[SubagentStats] = field(default_factory=list)

    def totals(self, which: Literal["main", "sidechain", "all"] = "all") -> dict[str, int]:
        buckets = []
        if which in ("main", "all"):
            buckets.append(self.main_by_model)
        if which in ("sidechain", "all"):
            buckets.append(self.sidechain_by_model)
        out: Counter = Counter()
        for bucket in buckets:
            for row in bucket.values():
                out.update(row)
        return dict(out)


def _scan_usage_lines(path: Path) -> tuple[dict[str, Counter], dict[str, Counter], datetime | None, str | None, int | None]:
    """One pass over a transcript file's `assistant` usage lines.

    Returns `(non_sidechain_by_model, sidechain_by_model, start, cwd,
    first_usage_context)`. `first_usage_context` is the
    input+cache_creation+cache_read total of the file's first deduped usage
    event, regardless of sidechain status — the "entrance fee" when the
    caller passes a subagent file. Dedups by `message.id` (a streamed
    response repeats the same id across several lines with identical
    `usage`), and splits `isSidechain` rows from the rest — used both for a
    main transcript (whose own sidechain lines, if any, are an older-format
    inline representation) and for a subagent file (where every line is
    already `isSidechain: true`).
    """
    seen_ids: set[str] = set()
    non_side: dict[str, Counter] = {}
    side: dict[str, Counter] = {}
    start: datetime | None = None
    cwd: str | None = None
    first_context: int | None = None

    with open(path, errors="ignore") as f:
        for line in f:
            if cwd is None and '"cwd"' in line[:400]:
                try:
                    maybe_cwd = json.loads(line).get("cwd")
                except json.JSONDecodeError:
                    maybe_cwd = None
                if maybe_cwd:
                    cwd = maybe_cwd
            if '"usage"' not in line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "assistant":
                continue
            if start is None:
                start = _parse_ts(d.get("timestamp"))
            message = d.get("message") or {}
            usage = message.get("usage")
            if not usage:
                continue
            message_id = message.get("id")
            if message_id:
                if message_id in seen_ids:
                    continue
                seen_ids.add(message_id)
            context = (usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
                      + usage.get("cache_read_input_tokens", 0))
            if first_context is None:
                first_context = context
            model = message.get("model", "unknown")
            bucket = side if d.get("isSidechain") else non_side
            row = bucket.setdefault(model, Counter())
            row["requests"] += 1
            row["input_tokens"] += usage.get("input_tokens", 0)
            row["cache_creation_tokens"] += usage.get("cache_creation_input_tokens", 0)
            row["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)
            row["output_tokens"] += usage.get("output_tokens", 0)

    return non_side, side, start, cwd, first_context


def _merge_model_counters(*sources: dict[str, Counter]) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for src in sources:
        for model, counter in src.items():
            out.setdefault(model, Counter()).update(counter)
    return out


def scan_claude_session(path: str | Path) -> ClaudeSessionStats:
    """Parse one Claude Code transcript JSONL file, plus its subagents.

    A Task-tool subagent's turns are NOT inline `isSidechain: true` lines in
    the main file — they live in `<session-id>/subagents/agent-*.jsonl`
    beside it (`<session-id>` derived from the main file's own stem). Each
    subagent file is scanned and folded into `sidechain_by_model`, and its
    per-subagent stats (including the "entrance fee") are kept in
    `.subagents`. Any *inline* `isSidechain: true` lines found in the main
    file itself (an older format, if it exists) are folded in too.
    """
    path = Path(path)
    main_by_model, inline_side_by_model, start, cwd, _ = _scan_usage_lines(path)

    subagents: list[SubagentStats] = []
    sidechain_sources = [inline_side_by_model]
    subagents_dir = path.parent / path.stem / "subagents"
    if subagents_dir.is_dir():
        for sub_path in sorted(subagents_dir.glob("agent-*.jsonl")):
            sub_non_side, sub_side, _, _, entrance_fee = _scan_usage_lines(sub_path)
            combined = _merge_model_counters(sub_non_side, sub_side)
            agent_id = sub_path.stem
            if agent_id.startswith("agent-"):
                agent_id = agent_id[len("agent-"):]
            subagents.append(SubagentStats(
                path=sub_path, agent_id=agent_id, model=next(iter(combined), None),
                first_turn_context=entrance_fee or 0,
                by_model={m: dict(c) for m, c in combined.items()},
            ))
            sidechain_sources.append(combined)

    sidechain_by_model = _merge_model_counters(*sidechain_sources)

    return ClaudeSessionStats(
        path=path, start=start, cwd=cwd,
        main_by_model={m: dict(c) for m, c in main_by_model.items()},
        sidechain_by_model={m: dict(c) for m, c in sidechain_by_model.items()},
        subagents=subagents,
    )


def scan_claude_sessions(
    root: str | Path = DEFAULT_CLAUDE_ROOT, *, since: datetime | None = None,
) -> list[ClaudeSessionStats]:
    """Scan every top-level transcript JSONL under `root` (default
    `~/.claude/projects`), one folder per project (folder name is the cwd
    with `/` -> `-`), one file per session. `*/*.jsonl` matches only the main
    transcripts — a session's own `<session-id>/subagents/*.jsonl` files are
    two levels deeper and are picked up per-session by `scan_claude_session`,
    not double-counted as their own top-level sessions here."""
    root = Path(root)
    out = []
    for f in sorted(root.glob("*/*.jsonl")):
        stats = scan_claude_session(f)
        if since and stats.start and stats.start < since:
            continue
        out.append(stats)
    return out


def claude_project_dir(path: Path) -> str:
    """The `-Users-...` project folder name a transcript lives under."""
    return path.parent.name


# ---------------------------------------------------------------------------
# Tool-call attribution for one session (Codex only — ports attrib.py)
# ---------------------------------------------------------------------------

_ATTRIB_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("verify loop (unittest/verify/rebuild/validate/build)",
     re.compile(r"unittest|pytest|verify\.py|rebuild\.py|validate_schemas|build_[a-z_]+\.py|evidence_integrity")),
    ("view images", re.compile(r"view_image|image\(")),
    ("git", re.compile(r"\bgit ")),
    ("read files (sed/cat/nl)", re.compile(r"sed -n|\bcat |\bnl ")),
    ("search (rg/grep/jq)", re.compile(r"\brg |\bgrep |\bjq ")),
    ("python one-off", re.compile(r"python3? - <<|python3? -c")),
    ("network/nb.no", re.compile(r"curl|nb\.no")),
]


def _classify_exec_input(inp: str) -> str:
    if "Begin Patch" in inp:
        if re.search(r"File: \S*data/\S+\.json", inp):
            return "patch DATA json by hand"
        if re.search(r"File: \S*(operations|docs|research)/|README|AGENTS|\.md", inp):
            return "patch docs/handoff/log"
        if "tests/" in inp:
            return "patch tests"
        return "patch code/site"
    for label, pattern in _ATTRIB_PATTERNS:
        if pattern.search(inp):
            return label
    return "other exec"


def attrib_session(path: str | Path) -> list[dict]:
    """Attribute each `token_count` event's input tokens to the tool call that
    immediately preceded it: verify loop, file reads, search, git, a patch to
    data/docs/code, agent management/polling, images, or "(user turn / no
    tool)" when no tool call preceded it. Ports `attrib.py`.

    Returns rows sorted by input tokens descending:
    `[{"category": ..., "input_tokens": ..., "requests": ..., "pct": ...}, ...]`
    """
    tokens: Counter = Counter()
    requests: Counter = Counter()
    last_category = "(user turn / no tool)"

    with open(path, errors="ignore") as f:
        for line in f:
            head = line[:400]
            if '"token_count"' in head:
                try:
                    lt = (json.loads(line).get("payload") or {}).get("info", {}).get("last_token_usage") or {}
                except json.JSONDecodeError:
                    lt = {}
                if lt:
                    tokens[last_category] += lt.get("input_tokens", 0)
                    requests[last_category] += 1
                    last_category = "(user turn / no tool)"
            elif '"custom_tool_call"' in head:
                try:
                    p = json.loads(line).get("payload") or {}
                except json.JSONDecodeError:
                    continue
                if p.get("type") == "custom_tool_call":
                    last_category = _classify_exec_input(str(p.get("input") or ""))
            elif '"function_call"' in head and '"function_call_output"' not in head:
                try:
                    p = json.loads(line).get("payload") or {}
                except json.JSONDecodeError:
                    continue
                last_category = "agent-mgmt: " + str(p.get("name"))

    total = sum(tokens.values()) or 1
    return [
        {"category": k, "input_tokens": v, "requests": requests[k], "pct": 100 * v / total}
        for k, v in tokens.most_common()
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _codex_project_table(sessions: list[CodexSessionStats], project_by: str) -> list[dict]:
    by_project: dict[str, list[CodexSessionStats]] = {}
    for s in sessions:
        proj = (s.project_by_paths if project_by == "paths" else s.project_by_cwd) or "(unassigned)"
        by_project.setdefault(proj, []).append(s)
    rows = []
    for proj, group in by_project.items():
        requests = sum(s.requests for s in group)
        input_tokens = sum(s.input_tokens for s in group)
        cached_tokens = sum(s.cached_tokens for s in group)
        output_tokens = sum(s.output_tokens for s in group)
        rows.append({
            "project": proj,
            "sessions": len(group),
            "requests": requests,
            "input_tokens": input_tokens,
            "cached_pct": 100 * cached_tokens / input_tokens if input_tokens else 0.0,
            "output_tokens": output_tokens,
            "mean_context": input_tokens / requests if requests else 0.0,
            "requests_over_150k": sum(s.requests_over_150k for s in group),
        })
    rows.sort(key=lambda r: r["input_tokens"], reverse=True)
    return rows


def _print_codex_report(sessions: list[CodexSessionStats], *, project_by: str, top: int) -> None:
    rows = _codex_project_table(sessions, project_by)
    print(f"\n  Codex sessions: {len(sessions)}  (project-by={project_by})\n")
    print(f"  {'Project':<20} {'Sessions':>9} {'Requests':>9} {'Input':>12} {'Cached%':>8} "
          f"{'Output':>10} {'MeanCtx':>9} {'>150K':>7}")
    for r in rows:
        print(f"  {r['project']:<20} {r['sessions']:>9} {r['requests']:>9} "
              f"{r['input_tokens']:>12,} {r['cached_pct']:>7.1f}% {r['output_tokens']:>10,} "
              f"{r['mean_context']:>9,.0f} {r['requests_over_150k']:>7}")

    disagreements = [s for s in sessions if s.project_disagreement]
    if disagreements:
        print(f"\n  {len(disagreements)} session(s) disagree between cwd and mentioned-paths assignment:")
        for s in disagreements[:20]:
            print(f"    {s.path.name}: cwd={s.project_by_cwd!r} paths={s.project_by_paths!r}")

    top_sessions = sorted(sessions, key=lambda s: s.input_tokens, reverse=True)[:top]
    print(f"\n  Top {len(top_sessions)} sessions by input tokens:")
    for s in top_sessions:
        print(f"    {s.input_tokens:>12,}  {s.thread_source or '?':<10} {s.model or '?':<16} {s.path}")
    print()


def _print_claude_report(sessions: list[ClaudeSessionStats], *, top: int) -> None:
    print(f"\n  Claude Code sessions: {len(sessions)}\n")
    by_model: Counter = Counter()
    main_total = side_total = 0
    entrance_fees: list[int] = []
    n_subagents = 0
    for s in sessions:
        main_t = s.totals("main")
        side_t = s.totals("sidechain")
        main_total += main_t.get("input_tokens", 0) + main_t.get("cache_read_tokens", 0)
        side_total += side_t.get("input_tokens", 0) + side_t.get("cache_read_tokens", 0)
        for model, row in _merge_model_counters(s.main_by_model, s.sidechain_by_model).items():
            by_model[model] += row.get("input_tokens", 0) + row.get("cache_read_tokens", 0)
        n_subagents += len(s.subagents)
        entrance_fees.extend(sub.first_turn_context for sub in s.subagents if sub.first_turn_context)

    print(f"  Main-thread tokens (input+cache-read):    {main_total:,}")
    print(f"  Sidechain/subagent tokens (input+cache-read): {side_total:,}")
    if main_total + side_total:
        pct_subagent = 100 * side_total / (main_total + side_total)
        print(f"  Subagent share: {pct_subagent:.1f}%")
    print(f"  Subagents: {n_subagents}")
    if entrance_fees:
        print(f"  Subagent entrance fee (first-turn context): median {statistics.median(entrance_fees):,.0f}"
              f"  mean {statistics.mean(entrance_fees):,.0f}"
              f"  max {max(entrance_fees):,}")
    print()
    print(f"  {'Model':<30} {'Tokens (input+cache-read)':>28}")
    for model, n in by_model.most_common():
        print(f"  {model:<30} {n:>28,}")

    def _session_total(s: ClaudeSessionStats) -> int:
        t = s.totals("all")
        return t.get("input_tokens", 0) + t.get("cache_read_tokens", 0)

    top_sessions = sorted(sessions, key=_session_total, reverse=True)[:top]
    print(f"\n  Top {len(top_sessions)} sessions by input+cache-read tokens (main+subagent):")
    for s in top_sessions:
        side = s.totals("sidechain")
        side_n = side.get("input_tokens", 0) + side.get("cache_read_tokens", 0)
        print(f"    {_session_total(s):>12,}  (subagent {side_n:>11,}, {len(s.subagents):>3} agents)  "
              f"{claude_project_dir(s.path):<45} {s.path.name}")
    print()


def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m limbic.cerebellum.forensics",
        description="Token-usage forensics over Claude Code / Codex session transcripts",
    )
    parser.add_argument("tool", choices=["codex", "claude"])
    parser.add_argument("--since", default=None, help="Relative window, e.g. 30d, 12h, 45m")
    parser.add_argument("--root", default=None, help="Override the default sessions root")
    parser.add_argument("--project-by", default="cwd", choices=["cwd", "paths"],
                        help="Codex only — which project assignment to report by")
    parser.add_argument("--top", type=int, default=10, help="Top-N sessions to list")
    parser.add_argument("--known-project", action="append", default=None,
                        help="Add a project name to the cwd/paths matcher (repeatable)")
    parser.add_argument("--session", default=None, help="Path to one session file")
    parser.add_argument("--attrib", action="store_true",
                        help="With --session (Codex only): tool-call attribution breakdown")
    args = parser.parse_args()

    known = args.known_project or DEFAULT_KNOWN_PROJECTS

    if args.session:
        if not args.attrib:
            parser.error("--session currently requires --attrib")
        rows = attrib_session(args.session)
        print(f"\n  Attribution for {args.session}\n")
        print(f"  {'Category':<55} {'Input tok':>12} {'Requests':>9} {'Pct':>7}")
        for r in rows:
            print(f"  {r['category']:<55} {r['input_tokens']:>12,} {r['requests']:>9} {r['pct']:>6.1f}%")
        print()
        return

    since_dt = parse_since(args.since) if args.since else None

    if args.tool == "codex":
        root = Path(args.root) if args.root else DEFAULT_CODEX_ROOT
        sessions = scan_codex_sessions(root, since=since_dt, known_projects=known)
        _print_codex_report(sessions, project_by=args.project_by, top=args.top)
    else:
        root = Path(args.root) if args.root else DEFAULT_CLAUDE_ROOT
        sessions = scan_claude_sessions(root, since=since_dt)
        _print_claude_report(sessions, top=args.top)


if __name__ == "__main__":
    _cli()
