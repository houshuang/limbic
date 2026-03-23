#!/usr/bin/env python3
"""Autoresearch grid sweep: Bayesian CPD parameters in knowledge_map.py.

Sweeps _CPD_HIGH, _CPD_LOW, _EVIDENCE_THRESHOLD and finds the combination
that maximizes Bayesian propagation accuracy across 4 topologies × 3 K-values × 30 trials.

Full factorial: 6 × 6 × 5 = 180 configs.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import limbic.amygdala.knowledge_map as km
from limbic.amygdala.knowledge_map import KnowledgeGraph, init_beliefs, update_beliefs


# ── Topology generators (same as autoresearch_eval.py) ──

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


# ── Evaluation ──

TOPOLOGIES = {
    "wide_tree": make_wide_tree,
    "dense_dag": lambda: make_dense_dag(20, 40),
    "chain_20": make_chain,
    "diamond_5x3": lambda: make_diamond(5, 3),
}
K_VALUES = [3, 5, 8]
N_TRIALS = 30

# Pre-generate all graphs and truth/assessed sets for consistency
master_rng = random.Random(42)
EVAL_CASES = []
for topo_name, make_fn in TOPOLOGIES.items():
    graph = make_fn()
    for K in K_VALUES:
        for trial in range(N_TRIALS):
            truth = {}
            rng_t = random.Random(trial * 7 + 13)
            visited, order = set(), []
            def visit(nid):
                if nid in visited: return
                visited.add(nid)
                for p in graph.prerequisites_of(nid): visit(p)
                order.append(nid)
            for n in graph.nodes: visit(n["id"])
            for nid in order:
                prereqs = graph.prerequisites_of(nid)
                all_known = all(truth.get(p) in ("basic","solid","deep") for p in prereqs) if prereqs else True
                p = 0.75 if all_known else 0.15
                truth[nid] = rng_t.choice(["basic","solid","deep"]) if rng_t.random() < p else rng_t.choice(["none","heard_of"])
            assessed = master_rng.sample(list(truth.keys()), min(K, len(truth)))
            EVAL_CASES.append((graph, truth, assessed, topo_name))


def evaluate_config(cpd_high, cpd_low, evidence_threshold):
    """Set module constants and evaluate accuracy."""
    km._CPD_HIGH = cpd_high
    km._CPD_LOW = cpd_low
    km._EVIDENCE_THRESHOLD = evidence_threshold

    correct_total = 0
    eval_total = 0
    for graph, truth, assessed, _ in EVAL_CASES:
        state = init_beliefs(graph, propagator="bayesian")
        for nid in assessed:
            update_beliefs(graph, state, nid, truth[nid])
        for nid, fam in truth.items():
            if nid in assessed:
                continue
            eval_total += 1
            predicted_known = state.beliefs.get(nid, 0.3) >= 0.5
            actual_known = fam in ("basic", "solid", "deep")
            if predicted_known == actual_known:
                correct_total += 1

    return correct_total / max(eval_total, 1)


def evaluate_config_by_topology(cpd_high, cpd_low, evidence_threshold):
    """Evaluate accuracy broken down by topology."""
    km._CPD_HIGH = cpd_high
    km._CPD_LOW = cpd_low
    km._EVIDENCE_THRESHOLD = evidence_threshold

    by_topo = {}
    for graph, truth, assessed, topo_name in EVAL_CASES:
        state = init_beliefs(graph, propagator="bayesian")
        for nid in assessed:
            update_beliefs(graph, state, nid, truth[nid])
        correct = total = 0
        for nid, fam in truth.items():
            if nid in assessed:
                continue
            total += 1
            predicted_known = state.beliefs.get(nid, 0.3) >= 0.5
            actual_known = fam in ("basic", "solid", "deep")
            if predicted_known == actual_known:
                correct += 1
        bucket = by_topo.setdefault(topo_name, {"correct": 0, "total": 0})
        bucket["correct"] += correct
        bucket["total"] += total

    return {k: v["correct"] / max(v["total"], 1) for k, v in by_topo.items()}


# ── Parameter space ──

CPD_HIGH_VALUES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CPD_LOW_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
EVIDENCE_THRESHOLD_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]

BASELINE = (0.85, 0.15, 0.5)

# ── Run sweep ──

if __name__ == "__main__":
    print(f"Running full factorial grid: {len(CPD_HIGH_VALUES)}×{len(CPD_LOW_VALUES)}×{len(EVIDENCE_THRESHOLD_VALUES)} = "
          f"{len(CPD_HIGH_VALUES)*len(CPD_LOW_VALUES)*len(EVIDENCE_THRESHOLD_VALUES)} configs")
    print(f"Eval cases per config: {len(EVAL_CASES)}")
    print()

    # Baseline
    baseline_score = evaluate_config(*BASELINE)
    print(f"Baseline (_CPD_HIGH=0.85, _CPD_LOW=0.15, _EVIDENCE_THRESHOLD=0.5): {baseline_score:.4f}")
    print()

    results = []
    t0 = time.time()
    total_configs = len(CPD_HIGH_VALUES) * len(CPD_LOW_VALUES) * len(EVIDENCE_THRESHOLD_VALUES)

    for i, cpd_high in enumerate(CPD_HIGH_VALUES):
        for cpd_low in CPD_LOW_VALUES:
            # Skip configs where cpd_low >= cpd_high (nonsensical)
            if cpd_low >= cpd_high:
                continue
            for ev_thresh in EVIDENCE_THRESHOLD_VALUES:
                score = evaluate_config(cpd_high, cpd_low, ev_thresh)
                delta = score - baseline_score
                results.append({
                    "cpd_high": cpd_high,
                    "cpd_low": cpd_low,
                    "evidence_threshold": ev_thresh,
                    "score": score,
                    "delta": delta,
                })
        elapsed = time.time() - t0
        done = len(results)
        if done > 0:
            rate = elapsed / done
            remaining = (total_configs - done) * rate
            print(f"  Progress: cpd_high={cpd_high:.2f} done ({done} configs, {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)} configs in {elapsed:.1f}s ({elapsed/len(results)*1000:.0f}ms per config)")

    # Sort by score
    results.sort(key=lambda r: r["score"], reverse=True)

    # Top 10
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'CPD_HIGH':>9} {'CPD_LOW':>8} {'EV_THRESH':>10} {'Score':>8} {'Delta':>8}")
    print("-" * 52)
    for i, r in enumerate(results[:10]):
        marker = " *" if r["delta"] > 0 else ""
        print(f"{i+1:>4} {r['cpd_high']:>9.2f} {r['cpd_low']:>8.2f} {r['evidence_threshold']:>10.1f} {r['score']:>8.4f} {r['delta']:>+8.4f}{marker}")

    # Bottom 5 (worst)
    print(f"\nBOTTOM 5:")
    for r in results[-5:]:
        print(f"  CPD_HIGH={r['cpd_high']:.2f} CPD_LOW={r['cpd_low']:.2f} EV_THRESH={r['evidence_threshold']:.1f} → {r['score']:.4f} ({r['delta']:+.4f})")

    # Per-topology breakdown for top config
    best = results[0]
    print(f"\n{'='*80}")
    print(f"BEST CONFIG: CPD_HIGH={best['cpd_high']}, CPD_LOW={best['cpd_low']}, EV_THRESH={best['evidence_threshold']}")
    print(f"Score: {best['score']:.4f} (delta: {best['delta']:+.4f} vs baseline {baseline_score:.4f})")
    print(f"{'='*80}")

    topo_best = evaluate_config_by_topology(best["cpd_high"], best["cpd_low"], best["evidence_threshold"])
    topo_base = evaluate_config_by_topology(*BASELINE)
    print(f"\nPer-topology accuracy:")
    print(f"{'Topology':>15} {'Baseline':>10} {'Best':>10} {'Delta':>10}")
    print("-" * 48)
    for topo in sorted(topo_best.keys()):
        delta = topo_best[topo] - topo_base[topo]
        print(f"{topo:>15} {topo_base[topo]:>10.4f} {topo_best[topo]:>10.4f} {delta:>+10.4f}")

    # Single-parameter sensitivity analysis
    print(f"\n{'='*80}")
    print("SINGLE-PARAMETER SENSITIVITY (others at baseline)")
    print(f"{'='*80}")

    print("\n_CPD_HIGH sweep (CPD_LOW=0.15, EV_THRESH=0.5):")
    for v in CPD_HIGH_VALUES:
        if v <= 0.15:
            continue
        s = evaluate_config(v, 0.15, 0.5)
        print(f"  {v:.2f} → {s:.4f} ({s - baseline_score:+.4f})")

    print("\n_CPD_LOW sweep (CPD_HIGH=0.85, EV_THRESH=0.5):")
    for v in CPD_LOW_VALUES:
        if v >= 0.85:
            continue
        s = evaluate_config(0.85, v, 0.5)
        print(f"  {v:.2f} → {s:.4f} ({s - baseline_score:+.4f})")

    print("\n_EVIDENCE_THRESHOLD sweep (CPD_HIGH=0.85, CPD_LOW=0.15):")
    for v in EVIDENCE_THRESHOLD_VALUES:
        s = evaluate_config(0.85, 0.15, v)
        print(f"  {v:.1f} → {s:.4f} ({s - baseline_score:+.4f})")

    # Save full results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output = {
        "experiment": "bayesian_cpd_grid_sweep",
        "baseline": {"cpd_high": 0.85, "cpd_low": 0.15, "evidence_threshold": 0.5, "score": baseline_score},
        "best": best,
        "top_10": results[:10],
        "all_results": results,
        "n_configs": len(results),
        "elapsed_seconds": elapsed,
    }
    out_path = results_dir / "autoresearch_bayesian_grid.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    # Restore baseline
    km._CPD_HIGH = 0.85
    km._CPD_LOW = 0.15
    km._EVIDENCE_THRESHOLD = 0.5
