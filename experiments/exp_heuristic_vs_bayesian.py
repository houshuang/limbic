#!/usr/bin/env python3
"""Auto-research loop: systematically improve heuristic propagation by comparing against pgmpy.

Varies heuristic parameters (dampen factor, ceiling/floor, prereqs-met threshold)
across multiple graph topologies and measures accuracy gap vs exact Bayesian inference.

Usage:
    python3 experiments/exp_heuristic_vs_bayesian.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from limbic.amygdala.knowledge_map import (
    KnowledgeGraph, BeliefState, init_beliefs, update_beliefs,
    _DAMPEN,
)


# ── Graph topologies ──────────────────────────────────────

def make_wide_tree(branches=5, depth=4):
    nodes = [{"id": "root", "title": "Root", "level": 1, "obscurity": 2, "prerequisites": []}]
    for b in range(branches):
        parent = "root"
        for d in range(depth):
            nid = f"b{b}_d{d}"
            nodes.append({"id": nid, "title": f"B{b}D{d}", "level": d+2, "obscurity": 3, "prerequisites": [parent]})
            parent = nid
    return KnowledgeGraph(nodes=nodes)

def make_dense_dag(n=20, n_edges=40, seed=42):
    rng = random.Random(seed)
    nodes = [{"id": f"d{i:02d}", "title": f"D{i}", "level": (i//5)+1, "obscurity": 3, "prerequisites": []} for i in range(n)]
    possible = [(i,j) for i in range(n) for j in range(i+1,n)]
    rng.shuffle(possible)
    for src, dst in possible[:n_edges]:
        nodes[dst]["prerequisites"].append(f"d{src:02d}")
    return KnowledgeGraph(nodes=nodes)

def make_chain(n=20):
    nodes = []
    for i in range(n):
        prereqs = [f"c{i-1:02d}"] if i > 0 else []
        nodes.append({"id": f"c{i:02d}", "title": f"C{i}", "level": i+1, "obscurity": 3, "prerequisites": prereqs})
    return KnowledgeGraph(nodes=nodes)

def make_diamond(n_layers=5, width=3):
    """Diamond/lattice: each layer of `width` nodes connects to all nodes in next layer."""
    nodes = [{"id": "root", "title": "Root", "level": 1, "obscurity": 2, "prerequisites": []}]
    prev_layer = ["root"]
    for layer in range(n_layers):
        curr_layer = []
        for w in range(width):
            nid = f"l{layer}_w{w}"
            nodes.append({"id": nid, "title": f"L{layer}W{w}", "level": layer+2, "obscurity": 3, "prerequisites": list(prev_layer)})
            curr_layer.append(nid)
        prev_layer = curr_layer
    return KnowledgeGraph(nodes=nodes)


# ── Ground truth generator ──────────────────────────────────

def generate_truth(graph, seed=42):
    rng = random.Random(seed)
    visited, order = set(), []
    def visit(nid):
        if nid in visited: return
        visited.add(nid)
        for p in graph.prerequisites_of(nid): visit(p)
        order.append(nid)
    for n in graph.nodes: visit(n["id"])
    truth = {}
    for nid in order:
        prereqs = graph.prerequisites_of(nid)
        all_known = all(truth.get(p) in ("basic","solid","deep") for p in prereqs) if prereqs else True
        p = 0.75 if all_known else 0.15
        truth[nid] = rng.choice(["basic","solid","deep"]) if rng.random() < p else rng.choice(["none","heard_of"])
    return truth


# ── Accuracy measurement ──────────────────────────────────

def measure_accuracy(graph, truth, assessed, propagator="heuristic"):
    state = init_beliefs(graph, propagator=propagator)
    for nid in assessed:
        update_beliefs(graph, state, nid, truth[nid])
    correct = total = 0
    for nid, fam in truth.items():
        if nid in assessed: continue
        total += 1
        if (state.beliefs.get(nid, 0.3) >= 0.5) == (fam in ("basic","solid","deep")): correct += 1
    return correct / max(total, 1)


# ── Main experiment ──────────────────────────────────────

def main():
    W = 90
    print("=" * W)
    print("Heuristic vs Bayesian Propagation: Parameter Sweep")
    print("=" * W)

    topologies = {
        "wide_tree_5x4": make_wide_tree(5, 4),
        "dense_dag_20_40": make_dense_dag(20, 40),
        "chain_20": make_chain(20),
        "diamond_5x3": make_diamond(5, 3),
    }

    K_VALUES = [3, 5, 8, 10]
    N_TRIALS = 40
    rng = random.Random(42)

    # ── Baseline: current heuristic vs pgmpy ──────────────
    print(f"\n{'─' * W}")
    print("Phase 1: Baseline comparison (current heuristic vs pgmpy)")
    print(f"{'─' * W}\n")

    baseline_results = {}
    for topo_name, graph in topologies.items():
        print(f"  {topo_name} ({len(graph.nodes)} nodes):")
        for K in K_VALUES:
            h_scores, p_scores = [], []
            for trial in range(N_TRIALS):
                truth = generate_truth(graph, seed=trial*7+13)
                assessed = rng.sample(list(truth.keys()), min(K, len(truth)))
                h_scores.append(measure_accuracy(graph, truth, assessed, "heuristic"))
                p_scores.append(measure_accuracy(graph, truth, assessed, "bayesian"))
            hm, pm = np.mean(h_scores), np.mean(p_scores)
            gap = pm - hm
            baseline_results[f"{topo_name}_K{K}"] = {"heuristic": round(hm, 4), "bayesian": round(pm, 4), "gap": round(gap, 4)}
            print(f"    K={K:>2}: heuristic={hm:.3f}  bayesian={pm:.3f}  gap={gap:+.3f}")

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("Phase 1 Summary")
    print(f"{'─' * W}")

    gaps = [v["gap"] for v in baseline_results.values()]
    print(f"  Mean gap:  {np.mean(gaps):+.4f}")
    print(f"  Max gap:   {max(gaps):+.4f}")
    print(f"  Min gap:   {min(gaps):+.4f}")
    print(f"  Median:    {np.median(gaps):+.4f}")

    # ── Analysis: where does the heuristic fail? ──────────
    print(f"\n{'─' * W}")
    print("Phase 2: Error analysis — where does the heuristic diverge from pgmpy?")
    print(f"{'─' * W}\n")

    # Pick the topology with the largest gap
    worst = max(baseline_results.items(), key=lambda x: x[1]["gap"])
    worst_key = worst[0]
    topo_name = worst_key.rsplit("_K", 1)[0]
    K = int(worst_key.rsplit("_K", 1)[1])
    graph = topologies[topo_name]
    print(f"  Worst case: {worst_key} (gap={worst[1]['gap']:+.3f})")
    print(f"  Analyzing error patterns on {topo_name}, K={K}...\n")

    # Categorize errors
    error_types = {"false_positive": 0, "false_negative": 0, "correct": 0}
    pgmpy_disagreements = []

    for trial in range(N_TRIALS):
        truth = generate_truth(graph, seed=trial*7+13)
        assessed = rng.sample(list(truth.keys()), min(K, len(truth)))

        state_h = init_beliefs(graph, propagator="heuristic")
        state_b = init_beliefs(graph, propagator="bayesian")
        for nid in assessed:
            update_beliefs(graph, state_h, nid, truth[nid])
            update_beliefs(graph, state_b, nid, truth[nid])

        for nid, fam in truth.items():
            if nid in assessed: continue
            is_known = fam in ("basic","solid","deep")
            h_pred = state_h.beliefs.get(nid, 0.3) >= 0.5
            b_pred = state_b.beliefs.get(nid, 0.3) >= 0.5

            if h_pred == is_known:
                error_types["correct"] += 1
            elif h_pred and not is_known:
                error_types["false_positive"] += 1
            else:
                error_types["false_negative"] += 1

            if h_pred != b_pred:
                node = graph.get(nid)
                pgmpy_disagreements.append({
                    "node": nid,
                    "level": node.get("level", 0),
                    "n_prereqs": len(graph.prerequisites_of(nid)),
                    "n_children": len(graph.children_of(nid)),
                    "h_belief": round(state_h.beliefs.get(nid, 0.3), 3),
                    "b_belief": round(state_b.beliefs.get(nid, 0.3), 3),
                    "truth": fam,
                    "h_correct": h_pred == is_known,
                    "b_correct": b_pred == is_known,
                })

    total = sum(error_types.values())
    print(f"  Error breakdown (N={total}):")
    for k, v in error_types.items():
        print(f"    {k}: {v} ({v/total:.1%})")

    # Analyze disagreement patterns
    if pgmpy_disagreements:
        pgmpy_wins = [d for d in pgmpy_disagreements if d["b_correct"] and not d["h_correct"]]
        heuristic_wins = [d for d in pgmpy_disagreements if d["h_correct"] and not d["b_correct"]]
        print(f"\n  Disagreements: {len(pgmpy_disagreements)} total")
        print(f"    pgmpy correct, heuristic wrong: {len(pgmpy_wins)}")
        print(f"    heuristic correct, pgmpy wrong: {len(heuristic_wins)}")

        if pgmpy_wins:
            # Analyze where pgmpy wins
            levels = [d["level"] for d in pgmpy_wins]
            prereqs = [d["n_prereqs"] for d in pgmpy_wins]
            children = [d["n_children"] for d in pgmpy_wins]
            print(f"\n  pgmpy-wins profile:")
            print(f"    Avg level:    {np.mean(levels):.1f}")
            print(f"    Avg prereqs:  {np.mean(prereqs):.1f}")
            print(f"    Avg children: {np.mean(children):.1f}")

            # Belief comparison
            h_beliefs = [d["h_belief"] for d in pgmpy_wins]
            b_beliefs = [d["b_belief"] for d in pgmpy_wins]
            print(f"    Avg h_belief: {np.mean(h_beliefs):.3f}")
            print(f"    Avg b_belief: {np.mean(b_beliefs):.3f}")

            # Show sample disagreements
            print(f"\n  Sample disagreements (pgmpy wins):")
            for d in pgmpy_wins[:8]:
                truth_known = d["truth"] in ("basic","solid","deep")
                print(f"    {d['node']:>12}: h={d['h_belief']:.2f} b={d['b_belief']:.2f} truth={'K' if truth_known else 'U'} "
                      f"(level={d['level']}, prereqs={d['n_prereqs']}, children={d['n_children']})")

    # ── Save results ──────────────────────────────────────
    out = {
        "experiment": "heuristic_vs_bayesian",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_trials": N_TRIALS,
        "k_values": K_VALUES,
        "baseline": baseline_results,
        "error_analysis": {
            "topology": topo_name,
            "K": K,
            "error_types": error_types,
            "n_disagreements": len(pgmpy_disagreements),
            "pgmpy_wins": len([d for d in pgmpy_disagreements if d["b_correct"] and not d["h_correct"]]),
            "heuristic_wins": len([d for d in pgmpy_disagreements if d["h_correct"] and not d["b_correct"]]),
        },
    }

    out_path = Path("experiments/results/exp_heuristic_vs_bayesian.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'=' * W}")
    print("DONE")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
