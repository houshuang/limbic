#!/usr/bin/env python3
"""Prototype: QMD-inspired search improvements for limbic.

Compares current hybrid search vs. enhanced search with:
1. LLM query expansion (lex + vec + hyde variants) — plain text parsing
2. Top-rank bonuses in RRF
3. Contribution tracing for debuggability
4. Strong signal early exit investigation

Uses real chat logs from claude-chat-search as the test corpus.
"""

import asyncio
import json
import os
import re
import sys
import time

import numpy as np

# Load API keys from ~/.env
_env_path = os.path.expanduser("~/.env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    if "GEMINI_API_KEY" in os.environ and "GEMINI_KEY" not in os.environ:
        os.environ["GEMINI_KEY"] = os.environ["GEMINI_API_KEY"]

_chat_search_src = os.environ.get("CHAT_SEARCH_SRC")
if _chat_search_src:
    sys.path.insert(0, _chat_search_src)

from claude_chat_search.db import get_connection, get_chunks_by_ids, fts_search
from claude_chat_search.embedder import embed_query, _get_model
from claude_chat_search.vector_search import numpy_vector_search

from limbic.amygdala.llm import generate_sync

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

RRF_K = 60

# ---------------------------------------------------------------------------
# Current search (baseline)
# ---------------------------------------------------------------------------

def current_rrf(ranked_lists, k=RRF_K):
    scores = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def current_search(conn, query, limit=10):
    fetch = limit * 5
    query_emb = embed_query(query)
    vec = numpy_vector_search(conn, query_emb, limit=fetch)
    fts = fts_search(conn, query, limit=fetch)
    fused = current_rrf([vec, fts])
    return _build_results(conn, fused, fetch, limit)


def _build_results(conn, fused, fetch, limit, traces=None, allowed_sessions=None):
    top_ids = [cid for cid, _ in fused[:fetch * 2]]
    chunks = get_chunks_by_ids(conn, top_ids)
    chunk_map = {c["id"]: c for c in chunks}

    seen = {}
    for cid, score in fused:
        chunk = chunk_map.get(cid)
        if chunk is None:
            continue
        sid = chunk["session_id"]
        if allowed_sessions is not None and sid not in allowed_sessions:
            continue
        if sid in seen:
            continue
        seen[sid] = {
            "chunk_id": cid, "score": score, "session_id": sid,
            "project_path": chunk["project_path"],
            "slug": chunk.get("slug") or "",
            "user_content": (chunk["user_content"] or "")[:150],
            "assistant_content": (chunk["assistant_content"] or "")[:150],
            "turn_number": chunk["turn_number"],
            "traces": (traces or {}).get(cid, []),
        }
        if len(seen) >= limit:
            break
    return list(seen.values())


# ---------------------------------------------------------------------------
# Query expansion — plain text, no structured output
# ---------------------------------------------------------------------------

EXPAND_PROMPT = """Expand this search query into 5 sub-queries for a hybrid search engine.

Query: {query}

Output format (5 lines, no extra text):
lex: keywords variant 1 (different technical terms)
lex: keywords variant 2 (alternate phrasing)
vec: semantic rephrase of the intent
hyde: hypothetical document excerpt that would answer the query
hyde: another hypothetical excerpt, different angle"""

_EXPAND_LINE_RE = re.compile(r'^(?:\d+\.\s*)?(lex|vec|hyde):\s*(.+)$', re.IGNORECASE)


def expand_query(query):
    """Plain text LLM expansion — much more reliable than structured output."""
    prompt = EXPAND_PROMPT.format(query=query)
    try:
        raw = generate_sync(
            prompt=prompt,
            system_prompt="Output exactly 5 lines. No thinking, no explanation.",
            model="gemini3-flash",
        )
        queries = []
        for line in raw.strip().splitlines():
            m = _EXPAND_LINE_RE.match(line.strip())
            if m:
                queries.append({"type": m.group(1).lower(), "query": m.group(2).strip()})
        if queries:
            return queries
    except Exception as e:
        print(f"  [expand] gemini3-flash failed: {e}")
    # Fallback
    try:
        raw = generate_sync(
            prompt=prompt,
            system_prompt="Output exactly 5 lines. No thinking, no explanation.",
            model="gemini25-flash",
        )
        queries = []
        for line in raw.strip().splitlines():
            m = _EXPAND_LINE_RE.match(line.strip())
            if m:
                queries.append({"type": m.group(1).lower(), "query": m.group(2).strip()})
        return queries
    except Exception as e:
        print(f"  [expand] FAILED: {e}")
        return []


# ---------------------------------------------------------------------------
# Enhanced RRF with top-rank bonuses and tracing
# ---------------------------------------------------------------------------

def enhanced_rrf(ranked_lists, list_labels, k=RRF_K):
    scores = {}
    traces = {}

    for list_idx, ranked in enumerate(ranked_lists):
        for rank, item in enumerate(ranked):
            cid = item["chunk_id"]
            contribution = 1.0 / (k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + contribution
            traces.setdefault(cid, []).append({
                "list": list_labels[list_idx],
                "rank": rank + 1,
                "contribution": contribution,
            })

    # Top-rank bonuses
    for cid, contribs in traces.items():
        top_rank = min(c["rank"] for c in contribs)
        if top_rank == 1:
            scores[cid] += 0.05
        elif top_rank <= 3:
            scores[cid] += 0.02

    return sorted(scores.items(), key=lambda x: x[1], reverse=True), traces


# ---------------------------------------------------------------------------
# Strong signal early exit detection
# ---------------------------------------------------------------------------

def check_strong_signal(vec_results, threshold=0.82, gap=0.12):
    """Check if first-pass vector search already has a clear winner.

    Returns (should_skip, top_score, gap_to_second, details).
    """
    if len(vec_results) < 2:
        return False, 0, 0, "too few results"
    # vec_results have "distance" (1 - cosine_sim), so score = 1 - distance
    top_score = 1.0 - vec_results[0]["distance"]
    second_score = 1.0 - vec_results[1]["distance"]
    score_gap = top_score - second_score
    should_skip = top_score >= threshold and score_gap >= gap
    return should_skip, top_score, score_gap, f"top={top_score:.3f} gap={score_gap:.3f}"


# ---------------------------------------------------------------------------
# Enhanced search
# ---------------------------------------------------------------------------

def enhanced_search(conn, query, limit=10, force_expand=True):
    """Enhanced search with query expansion, multi-query RRF, and tracing.

    Returns (results, info_dict).
    """
    fetch = limit * 5
    model = _get_model()

    # First-pass: always run original query through both backends
    orig_emb = embed_query(query)
    vec_results = numpy_vector_search(conn, orig_emb, limit=fetch)
    fts_results = fts_search(conn, query, limit=fetch)

    # Check strong signal before expanding
    skip, top_score, gap, signal_detail = check_strong_signal(vec_results)

    ranked_lists = [vec_results, fts_results]
    list_labels = ["vec:original", "fts:original"]
    expanded = []
    expand_time = 0
    skipped_expansion = False

    if skip and not force_expand:
        skipped_expansion = True
    else:
        t0 = time.time()
        expanded = expand_query(query)
        expand_time = time.time() - t0

        for eq in expanded:
            qtype = eq["type"]
            qtext = eq["query"]
            if qtype == "lex":
                ranked_lists.append(fts_search(conn, qtext, limit=fetch))
                list_labels.append(f"fts:{qtext[:40]}")
            elif qtype in ("vec", "hyde"):
                emb = model.embed(qtext).tolist()
                ranked_lists.append(numpy_vector_search(conn, emb, limit=fetch))
                list_labels.append(f"vec:{qtype}:{qtext[:35]}")

    fused, traces = enhanced_rrf(ranked_lists, list_labels)
    results = _build_results(conn, fused, fetch, limit, traces)

    info = {
        "expand_time": expand_time,
        "expanded_queries": expanded,
        "num_ranked_lists": len(ranked_lists),
        "strong_signal": signal_detail,
        "skipped_expansion": skipped_expansion,
    }
    return results, info


# ---------------------------------------------------------------------------
# Test queries — broad coverage
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    # --- Vague topic queries (should benefit most from expansion) ---
    "how we improved search quality",
    "dealing with database lock problems",
    "setting up the embedding pipeline",
    # --- Cross-vocabulary (user phrasing != stored text) ---
    "making the daemon watch for file changes",
    "conversation about Norwegian language support",
    # --- Specific technical topics ---
    "autoresearch parameter optimization loop",
    "chunking strategy for long conversations",
    # --- New queries: broader coverage ---
    "debugging CI failures in pull requests",
    "how the cost tracking system works",
    "designing the weekly report feature",
    "migrating from one database schema to another",
    "when we discussed the calendar integration",
    "fixing the FTS5 query sanitization bug",
    "how subagent conversations are stored",
    "conversation about whitening embeddings",
]


# ---------------------------------------------------------------------------
# Output formatting and comparison
# ---------------------------------------------------------------------------

def fmt(r, i):
    proj = (r.get("project_path") or "").split("/")[-1] or "?"
    slug = (r.get("slug") or "")[:25]
    user = (r.get("user_content") or "")[:80].replace("\n", " ")
    return f"  #{i+1} [{r['score']:.4f}] {proj}/{slug}  t={r['turn_number']}\n       {user}"


def fmt_traces(traces):
    if not traces:
        return ""
    parts = [f"{t['list']}(r{t['rank']})" for t in sorted(traces, key=lambda t: t["rank"])[:5]]
    extra = f" +{len(traces)-5}more" if len(traces) > 5 else ""
    return "       sources: " + ", ".join(parts) + extra


def compare(baseline, enhanced):
    b_sids = [r["session_id"] for r in baseline]
    e_sids = [r["session_id"] for r in enhanced]
    b_set, e_set = set(b_sids), set(e_sids)
    new = e_set - b_set
    lost = b_set - e_set
    reordered = b_set == e_set and b_sids != e_sids

    # Score improvement for sessions in both
    shared = b_set & e_set
    b_scores = {r["session_id"]: r["score"] for r in baseline}
    e_scores = {r["session_id"]: r["score"] for r in enhanced}
    avg_lift = 0
    if shared:
        lifts = [(e_scores[s] - b_scores[s]) / max(b_scores[s], 0.001) for s in shared]
        avg_lift = sum(lifts) / len(lifts)

    return {
        "new": len(new), "lost": len(lost), "reordered": reordered,
        "identical": not new and not lost and not reordered,
        "avg_score_lift": avg_lift,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = get_connection()
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    print(f"Corpus: {n_chunks} chunks across {n_sessions} sessions\n")

    # Aggregate stats
    stats = {
        "total": 0, "expanded_ok": 0, "new_surfaced": 0, "lost": 0,
        "identical": 0, "reordered": 0, "avg_lift": [],
        "would_skip": 0, "expand_times": [],
    }

    for query in TEST_QUERIES:
        stats["total"] += 1
        print(f"{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")

        # Baseline
        t0 = time.time()
        baseline = current_search(conn, query, limit=5)
        baseline_time = time.time() - t0

        print(f"\n--- BASELINE ({baseline_time*1000:.0f}ms) ---")
        for i, r in enumerate(baseline):
            print(fmt(r, i))

        # Enhanced (always expand for comparison)
        t0 = time.time()
        enhanced, info = enhanced_search(conn, query, limit=5, force_expand=True)
        enhanced_time = time.time() - t0

        n_expanded = len(info["expanded_queries"])
        if n_expanded > 0:
            stats["expanded_ok"] += 1
        stats["expand_times"].append(info["expand_time"])

        # Strong signal check
        skip_detail = info["strong_signal"]
        would_skip = "top=" in skip_detail and float(skip_detail.split("top=")[1].split()[0]) >= 0.82
        if would_skip:
            stats["would_skip"] += 1

        print(f"\n--- ENHANCED ({enhanced_time*1000:.0f}ms, expand={info['expand_time']*1000:.0f}ms, "
              f"{info['num_ranked_lists']} lists, signal={skip_detail}"
              f"{' WOULD-SKIP' if would_skip else ''}) ---")
        if info["expanded_queries"]:
            for eq in info["expanded_queries"]:
                print(f"    [{eq['type']}] {eq['query'][:70]}")

        for i, r in enumerate(enhanced):
            print(fmt(r, i))
            print(fmt_traces(r.get("traces", [])))

        # Compare
        cmp = compare(baseline, enhanced)
        stats["new_surfaced"] += cmp["new"]
        stats["lost"] += cmp["lost"]
        if cmp["identical"]:
            stats["identical"] += 1
        if cmp["reordered"]:
            stats["reordered"] += 1
        stats["avg_lift"].append(cmp["avg_score_lift"])

        verdict = []
        if cmp["new"]:
            verdict.append(f"+{cmp['new']} new")
        if cmp["lost"]:
            verdict.append(f"-{cmp['lost']} lost")
        if cmp["reordered"]:
            verdict.append("reordered")
        if cmp["identical"]:
            verdict.append("identical")
        if cmp["avg_score_lift"] > 0:
            verdict.append(f"score +{cmp['avg_score_lift']:.0%}")
        print(f"\n  >> {', '.join(verdict)}")
        print()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({stats['total']} queries)")
    print(f"{'='*70}")
    print(f"  Expansion success: {stats['expanded_ok']}/{stats['total']}")
    print(f"  Avg expand time:   {sum(stats['expand_times'])/len(stats['expand_times'])*1000:.0f}ms")
    print(f"  New sessions surfaced: {stats['new_surfaced']} total")
    print(f"  Sessions lost:     {stats['lost']} total")
    print(f"  Identical results: {stats['identical']}/{stats['total']}")
    print(f"  Reordered only:    {stats['reordered']}/{stats['total']}")
    avg = sum(stats["avg_lift"]) / len(stats["avg_lift"]) if stats["avg_lift"] else 0
    print(f"  Avg score lift:    {avg:.0%} (for shared sessions)")
    print(f"  Would-skip (early exit): {stats['would_skip']}/{stats['total']}")

    conn.close()


if __name__ == "__main__":
    main()
