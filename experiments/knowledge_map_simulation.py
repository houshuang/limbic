#!/usr/bin/env python3
"""Simulation harness for validating the knowledge_map belief propagation algorithm.

Experiments:
  1. Propagation Accuracy (Leave-K-Out) — how well do K assessed nodes predict the rest?
  2. Convergence Speed (Strategy Comparison) — entropy vs random vs level_order vs most_connected
  3. Graph Topology Effects — how topology shapes convergence across strategies

Usage:
    python3 experiments/knowledge_map_simulation.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

# Add parent dir so amygdala is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amygdala.knowledge_map import (
    BeliefState,
    KnowledgeGraph,
    init_beliefs,
    next_probe,
    update_beliefs,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_TRIALS = 100  # repetitions for leave-K-out
K_VALUES = [5, 10, 15, 20, 25]

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUTPUT_DIR / "simulation_results.json"

# Where real curriculum data might live
CURRICULA_PATHS = [
    Path("data") / "curricula",
    Path("/opt/data/curricula"),
]

# Map from real knowledge labels to familiarity levels used by the algorithm
KNOWLEDGE_TO_FAMILIARITY = {
    "unknown": "none",
    "mentioned": "heard_of",
    "basic": "basic",
    "engaged": "solid",
    "anchored": "deep",
}


# ---------------------------------------------------------------------------
# Ground truth generation
# ---------------------------------------------------------------------------


def _topo_sort(graph: KnowledgeGraph) -> list[str]:
    """Topological sort of the knowledge graph (prerequisites first)."""
    visited = set()
    order = []

    def visit(nid: str):
        if nid in visited:
            return
        visited.add(nid)
        for prereq in graph.prerequisites_of(nid):
            visit(prereq)
        order.append(nid)

    for n in graph.nodes:
        visit(n["id"])
    return order


def generate_synthetic_ground_truth(
    graph: KnowledgeGraph, known_frac: float = 0.5, rng: random.Random | None = None
) -> dict[str, str]:
    """Generate ground truth respecting prerequisite structure.

    Nodes are processed in topological order. A root node has a high probability
    of being known (calibrated so the overall fraction lands near known_frac).
    If any prerequisite is unknown, the node has only a 15% chance of being known
    (leaky but respects structure). Known nodes get 'solid' or 'deep'; unknown
    get 'none' or 'heard_of'.
    """
    if rng is None:
        rng = random.Random(RANDOM_SEED)

    order = _topo_sort(graph)
    truth: dict[str, str] = {}

    # Use a higher base probability for roots to compensate for prerequisite cascading.
    # With deep trees, known_frac=0.5 at each node produces far fewer than 50% known
    # overall because failures cascade. Use 0.75 base to get roughly 50% after cascading.
    base_p = min(0.95, known_frac + 0.25)

    for nid in order:
        prereqs = graph.prerequisites_of(nid)
        all_prereqs_known = all(
            truth.get(p) in ("solid", "deep", "basic") for p in prereqs
        ) if prereqs else True

        if all_prereqs_known:
            p_known = base_p
        else:
            p_known = 0.15  # leaky — possible but unlikely

        if rng.random() < p_known:
            truth[nid] = rng.choice(["basic", "solid", "deep"])
        else:
            truth[nid] = rng.choice(["none", "heard_of"])

    return truth


def truth_to_binary(truth: dict[str, str]) -> dict[str, bool]:
    """Convert familiarity-level truth to binary known/unknown."""
    return {
        nid: fam in ("basic", "solid", "deep")
        for nid, fam in truth.items()
    }


# ---------------------------------------------------------------------------
# Accuracy measurement
# ---------------------------------------------------------------------------


def measure_accuracy(
    state: BeliefState, ground_truth_binary: dict[str, bool], exclude: set[str] | None = None
) -> float:
    """Fraction of non-excluded nodes correctly classified.

    Uses belief >= 0.5 as the decision boundary: above 0.5 → predicted known,
    below 0.5 → predicted unknown. This aligns with the natural interpretation
    of belief probabilities and avoids a dead zone where no prediction is made.
    """
    exclude = exclude or set()
    correct = 0
    total = 0
    for nid, is_known in ground_truth_binary.items():
        if nid in exclude:
            continue
        total += 1
        belief = state.beliefs.get(nid, 0.3)
        predicted_known = belief >= 0.5
        if predicted_known == is_known:
            correct += 1
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Synthetic graph topologies (Experiment 3)
# ---------------------------------------------------------------------------


def make_deep_chain(n: int = 20) -> KnowledgeGraph:
    """Linear chain: A→B→C→...→T (depth = n)."""
    nodes = []
    for i in range(n):
        nid = f"chain_{i:02d}"
        prereqs = [f"chain_{i-1:02d}"] if i > 0 else []
        nodes.append({
            "id": nid, "title": f"Chain Node {i}", "level": i + 1,
            "obscurity": 3, "prerequisites": prereqs,
        })
    return KnowledgeGraph(nodes=nodes)


def make_wide_tree(branches: int = 5, depth: int = 4) -> KnowledgeGraph:
    """1 root, `branches` branches each of `depth` nodes."""
    nodes = [{"id": "root", "title": "Root", "level": 1, "obscurity": 2, "prerequisites": []}]
    for b in range(branches):
        parent = "root"
        for d in range(depth):
            nid = f"b{b}_d{d}"
            nodes.append({
                "id": nid, "title": f"Branch {b} Depth {d}", "level": d + 2,
                "obscurity": 3, "prerequisites": [parent],
            })
            parent = nid
    return KnowledgeGraph(nodes=nodes)


def make_dense_dag(n: int = 20, n_edges: int = 40, seed: int = RANDOM_SEED) -> KnowledgeGraph:
    """n nodes with approximately n_edges prerequisite edges (DAG, no cycles)."""
    rng = random.Random(seed)
    nodes = []
    for i in range(n):
        nodes.append({
            "id": f"dag_{i:02d}", "title": f"DAG Node {i}",
            "level": (i // 5) + 1, "obscurity": 3, "prerequisites": [],
        })
    # Add edges only from lower-id to higher-id (guarantees DAG)
    possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(possible_edges)
    edge_count = min(n_edges, len(possible_edges))
    for src, dst in possible_edges[:edge_count]:
        nodes[dst]["prerequisites"].append(f"dag_{src:02d}")
    return KnowledgeGraph(nodes=nodes)


def make_flat(n: int = 20) -> KnowledgeGraph:
    """n independent nodes with no prerequisites."""
    nodes = []
    for i in range(n):
        nodes.append({
            "id": f"flat_{i:02d}", "title": f"Flat Node {i}",
            "level": 1, "obscurity": 3, "prerequisites": [],
        })
    return KnowledgeGraph(nodes=nodes)


# ---------------------------------------------------------------------------
# Question selection strategies (Experiment 2)
# ---------------------------------------------------------------------------


def strategy_entropy(graph: KnowledgeGraph, state: BeliefState) -> str | None:
    """Use next_probe with greedy max-entropy selection."""
    probe = next_probe(graph, state, strategy="entropy")
    if probe is None:
        unassessed = [n["id"] for n in graph.nodes if n["id"] not in state.assessed]
        return unassessed[0] if unassessed else None
    return probe["node_id"]


def strategy_eig(graph: KnowledgeGraph, state: BeliefState) -> str | None:
    """Use next_probe with Expected Information Gain selection."""
    probe = next_probe(graph, state, strategy="eig")
    if probe is None:
        unassessed = [n["id"] for n in graph.nodes if n["id"] not in state.assessed]
        return unassessed[0] if unassessed else None
    return probe["node_id"]


def strategy_random(graph: KnowledgeGraph, state: BeliefState, rng: random.Random) -> str | None:
    """Pick a random unassessed node."""
    unassessed = [n["id"] for n in graph.nodes if n["id"] not in state.assessed]
    if not unassessed:
        return None
    return rng.choice(unassessed)


def strategy_level_order(graph: KnowledgeGraph, state: BeliefState) -> str | None:
    """Assess level 1 first, then level 2, etc."""
    unassessed = [n for n in graph.nodes if n["id"] not in state.assessed]
    if not unassessed:
        return None
    unassessed.sort(key=lambda n: (n.get("level", 1), n["id"]))
    return unassessed[0]["id"]


def strategy_most_connected(graph: KnowledgeGraph, state: BeliefState) -> str | None:
    """Pick the unassessed node with highest in+out degree."""
    unassessed = [n for n in graph.nodes if n["id"] not in state.assessed]
    if not unassessed:
        return None

    def degree(node):
        in_deg = len(node.get("prerequisites", []))
        out_deg = len(graph.children_of(node["id"]))
        return in_deg + out_deg

    unassessed.sort(key=lambda n: degree(n), reverse=True)
    return unassessed[0]["id"]


# ---------------------------------------------------------------------------
# Experiment 1: Leave-K-Out propagation accuracy
# ---------------------------------------------------------------------------


def run_leave_k_out(
    graph: KnowledgeGraph,
    ground_truth: dict[str, str],
    k_values: list[int] = K_VALUES,
    n_trials: int = N_TRIALS,
    seed: int = RANDOM_SEED,
) -> dict:
    """Assess K random nodes, propagate, measure accuracy on the rest."""
    rng = random.Random(seed)
    truth_binary = truth_to_binary(ground_truth)
    all_node_ids = list(ground_truth.keys())
    n_nodes = len(all_node_ids)

    results = {}
    for k in k_values:
        if k > n_nodes:
            continue
        accuracies = []
        for trial in range(n_trials):
            assessed_ids = set(rng.sample(all_node_ids, k))
            state = init_beliefs(graph, prior_fn=lambda n: 0.5)

            for nid in assessed_ids:
                update_beliefs(graph, state, nid, ground_truth[nid])

            acc = measure_accuracy(state, truth_binary, exclude=assessed_ids)
            accuracies.append(acc)

        arr = np.array(accuracies)
        results[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_trials": n_trials,
            "n_nodes": n_nodes,
            "n_remaining": n_nodes - k,
        }
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Convergence speed (strategy comparison)
# ---------------------------------------------------------------------------


def _run_single_convergence(
    graph: KnowledgeGraph,
    ground_truth: dict[str, str],
    pick_fn,
    max_questions: int,
    seed: int,
) -> list[float]:
    """Run a single convergence trial for one strategy."""
    truth_binary = truth_to_binary(ground_truth)
    rng = random.Random(seed)
    state = init_beliefs(graph, prior_fn=lambda n: 0.5)
    accuracies = []

    acc0 = measure_accuracy(state, truth_binary)
    accuracies.append(acc0)

    for q in range(max_questions):
        nid = pick_fn(graph, state, rng)
        if nid is None:
            accuracies.extend([accuracies[-1]] * (max_questions - q))
            break
        update_beliefs(graph, state, nid, ground_truth[nid])
        acc = measure_accuracy(state, truth_binary)
        accuracies.append(acc)

    return accuracies[:max_questions + 1]


def run_convergence(
    graph: KnowledgeGraph,
    ground_truth: dict[str, str],
    max_questions: int | None = None,
    seed: int = RANDOM_SEED,
    n_random_trials: int = 20,
) -> dict[str, list[float]]:
    """Run all four strategies and track accuracy after each question.

    Deterministic strategies (entropy, level_order, most_connected) run once.
    The random strategy is averaged over n_random_trials to reduce variance.
    """
    all_node_ids = [n["id"] for n in graph.nodes]
    if max_questions is None:
        max_questions = len(all_node_ids)

    strategies = {
        "eig": lambda g, s, rng: strategy_eig(g, s),
        "entropy": lambda g, s, rng: strategy_entropy(g, s),
        "random": lambda g, s, rng: strategy_random(g, s, rng),
        "level_order": lambda g, s, rng: strategy_level_order(g, s),
        "most_connected": lambda g, s, rng: strategy_most_connected(g, s),
    }

    results = {}
    for name, pick_fn in strategies.items():
        if name == "random":
            # Average over multiple trials
            all_trials = []
            for trial in range(n_random_trials):
                trial_seed = seed + trial
                accs = _run_single_convergence(graph, ground_truth, pick_fn, max_questions, trial_seed)
                all_trials.append(accs)
            # Average across trials at each question count
            arr = np.array(all_trials)
            results[name] = [float(np.mean(arr[:, q])) for q in range(arr.shape[1])]
        else:
            results[name] = _run_single_convergence(graph, ground_truth, pick_fn, max_questions, seed)

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Topology effects
# ---------------------------------------------------------------------------


def run_topology_experiment(seed: int = RANDOM_SEED) -> dict:
    """Run convergence experiment on four different graph topologies."""
    topologies = {
        "deep_chain": make_deep_chain(20),
        "wide_tree": make_wide_tree(5, 4),  # 1 + 5*4 = 21 nodes
        "dense_dag": make_dense_dag(20, 40, seed),
        "flat": make_flat(20),
    }

    results = {}
    for topo_name, graph in topologies.items():
        rng = random.Random(seed)
        truth = generate_synthetic_ground_truth(graph, known_frac=0.5, rng=rng)
        n_known = sum(1 for v in truth.values() if v in ("basic", "solid", "deep"))
        n_total = len(truth)

        convergence = run_convergence(graph, truth, seed=seed)
        results[topo_name] = {
            "n_nodes": n_total,
            "n_edges": sum(len(n.get("prerequisites", [])) for n in graph.nodes),
            "n_known": n_known,
            "convergence": convergence,
        }
    return results


# ---------------------------------------------------------------------------
# Real data loading
# ---------------------------------------------------------------------------


def load_real_curriculum() -> tuple[KnowledgeGraph, dict[str, str]] | None:
    """Try to load the Ancient Greece curriculum with knowledge states as ground truth."""
    for base in CURRICULA_PATHS:
        curriculum_file = base / "ancient_greece_800300_bc_political_military_cultural_and.json"
        knowledge_file = base / "knowledge_ancient_greece_800300_bc_political_military_cultural_and.json"
        if curriculum_file.exists() and knowledge_file.exists():
            with open(curriculum_file) as f:
                curriculum = json.load(f)
            with open(knowledge_file) as f:
                knowledge = json.load(f)

            # Build the graph
            graph = KnowledgeGraph(nodes=curriculum["nodes"])

            # Build ground truth: only for nodes that have knowledge assessments
            truth = {}
            for nid, kstate in knowledge.items():
                klevel = kstate.get("knowledge", "unknown")
                if klevel in KNOWLEDGE_TO_FAMILIARITY:
                    truth[nid] = KNOWLEDGE_TO_FAMILIARITY[klevel]

            if len(truth) >= 10:
                return graph, truth
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_leave_k_out(results: dict, title_suffix: str = "", filename: str = "leave_k_out.png"):
    """Plot accuracy vs K for leave-K-out experiment."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not available, skipping plot]")
        return

    ks = sorted(results.keys())
    means = [results[k]["mean"] for k in ks]
    stds = [results[k]["std"] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, means, yerr=stds, fmt="o-", capsize=5, linewidth=2, markersize=8,
                color="#8b2500", ecolor="#6a6458")
    ax.set_xlabel("K (nodes assessed)", fontsize=13)
    ax.set_ylabel("Accuracy on remaining nodes", fontsize=13)
    ax.set_title(f"Leave-K-Out Propagation Accuracy{title_suffix}", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add annotations
    for k, m, s in zip(ks, means, stds):
        ax.annotate(f"{m:.2f}", (k, m), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {OUTPUT_DIR / filename}")


def plot_convergence(
    convergence: dict[str, list[float]],
    title: str = "Convergence Speed",
    filename: str = "convergence.png",
):
    """Plot accuracy vs questions asked for each strategy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not available, skipping plot]")
        return

    colors = {"eig": "#c9a84c", "entropy": "#8b2500", "random": "#6a6458", "level_order": "#2a7a4a", "most_connected": "#1a5276"}
    markers = {"eig": "*", "entropy": "o", "random": "s", "level_order": "^", "most_connected": "D"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, accs in convergence.items():
        xs = list(range(len(accs)))
        ax.plot(xs, accs, label=name, color=colors.get(name, "gray"),
                marker=markers.get(name, "."), markersize=4, linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Questions asked", fontsize=13)
    ax.set_ylabel("Accuracy (vs ground truth)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {OUTPUT_DIR / filename}")


def plot_topology_comparison(topo_results: dict, filename: str = "topology_comparison.png"):
    """2x2 grid of convergence plots, one per topology."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not available, skipping plot]")
        return

    colors = {"eig": "#c9a84c", "entropy": "#8b2500", "random": "#6a6458", "level_order": "#2a7a4a", "most_connected": "#1a5276"}
    topo_names = list(topo_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, topo_name in enumerate(topo_names):
        ax = axes[idx // 2][idx % 2]
        info = topo_results[topo_name]
        convergence = info["convergence"]
        for strat_name, accs in convergence.items():
            xs = list(range(len(accs)))
            ax.plot(xs, accs, label=strat_name, color=colors.get(strat_name, "gray"),
                    linewidth=1.5, alpha=0.85)
        ax.set_title(f"{topo_name} ({info['n_nodes']}n, {info['n_edges']}e, {info['n_known']} known)",
                     fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Questions", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Topology Effects on Convergence Speed", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {OUTPUT_DIR / filename}")


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def print_table(headers: list[str], rows: list[list], col_widths: list[int] | None = None):
    """Print a nicely formatted ASCII table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=5)) + 2
                      for i, h in enumerate(headers)]

    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "".join("-" * w for w in col_widths)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def format_pct(v: float) -> str:
    return f"{v*100:.1f}%"


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Knowledge Map Simulation Harness")
    print("=" * 70)

    all_results = {}
    rng = random.Random(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Try to load real data
    # ------------------------------------------------------------------
    real_data = load_real_curriculum()
    if real_data:
        real_graph, real_truth = real_data
        # Filter graph to only nodes with ground truth
        assessed_node_ids = set(real_truth.keys())
        assessed_nodes = [n for n in real_graph.nodes if n["id"] in assessed_node_ids]
        # Also keep prerequisites that exist within assessed set
        for n in assessed_nodes:
            n["prerequisites"] = [p for p in n.get("prerequisites", []) if p in assessed_node_ids]
        real_graph_filtered = KnowledgeGraph(nodes=assessed_nodes)

        n_known = sum(1 for f in real_truth.values() if f in ("basic", "solid", "deep"))
        print(f"\nLoaded real data: Ancient Greece curriculum")
        print(f"  {len(assessed_nodes)} assessed nodes, {n_known} known, "
              f"{len(assessed_nodes) - n_known} unknown")
        print(f"  {sum(len(n.get('prerequisites', [])) for n in assessed_nodes)} prerequisite edges")
    else:
        real_graph_filtered = None
        real_truth = None
        print("\nNo real curriculum data found, using only synthetic data.")

    # ------------------------------------------------------------------
    # Generate synthetic graph for experiments 1 & 2
    # ------------------------------------------------------------------
    print("\n--- Generating synthetic graph for Experiments 1 & 2 ---")
    # Use a moderate-size graph: wide_tree gives good structure
    synth_graph = make_wide_tree(branches=6, depth=5)  # 31 nodes
    synth_truth = generate_synthetic_ground_truth(synth_graph, known_frac=0.5, rng=random.Random(RANDOM_SEED))
    n_synth_known = sum(1 for f in synth_truth.values() if f in ("basic", "solid", "deep"))
    n_synth_total = len(synth_truth)
    synth_edges = sum(len(n.get("prerequisites", [])) for n in synth_graph.nodes)
    print(f"  wide_tree: {n_synth_total} nodes, {synth_edges} edges, "
          f"{n_synth_known} known ({n_synth_known/n_synth_total*100:.0f}%)")

    # ==================================================================
    # EXPERIMENT 1: Leave-K-Out
    # ==================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Leave-K-Out Propagation Accuracy")
    print("=" * 70)

    # Synthetic
    print(f"\n--- Synthetic data (wide_tree, {n_synth_total} nodes, {N_TRIALS} trials per K) ---")
    lko_synth = run_leave_k_out(synth_graph, synth_truth, K_VALUES, N_TRIALS)
    all_results["exp1_leave_k_out_synthetic"] = lko_synth

    rows = []
    for k in sorted(lko_synth.keys()):
        r = lko_synth[k]
        rows.append([k, r["n_remaining"], format_pct(r["mean"]), format_pct(r["std"]),
                      format_pct(r["min"]), format_pct(r["max"])])
    print_table(["K", "Remaining", "Mean Acc", "Std", "Min", "Max"],
                rows, [8, 12, 12, 10, 10, 10])

    plot_leave_k_out(lko_synth, " (Synthetic wide_tree)", "leave_k_out_synthetic.png")

    # Real data
    if real_graph_filtered and real_truth:
        n_real = len(real_truth)
        real_k_values = [k for k in K_VALUES if k < n_real - 2]
        if real_k_values:
            print(f"\n--- Real data (Ancient Greece, {n_real} nodes, {N_TRIALS} trials per K) ---")
            lko_real = run_leave_k_out(real_graph_filtered, real_truth, real_k_values, N_TRIALS)
            all_results["exp1_leave_k_out_real"] = lko_real

            rows = []
            for k in sorted(lko_real.keys()):
                r = lko_real[k]
                rows.append([k, r["n_remaining"], format_pct(r["mean"]), format_pct(r["std"]),
                              format_pct(r["min"]), format_pct(r["max"])])
            print_table(["K", "Remaining", "Mean Acc", "Std", "Min", "Max"],
                        rows, [8, 12, 12, 10, 10, 10])

            plot_leave_k_out(lko_real, " (Real: Ancient Greece)", "leave_k_out_real.png")

    # ==================================================================
    # EXPERIMENT 2: Convergence Speed
    # ==================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Convergence Speed (Strategy Comparison)")
    print("=" * 70)

    # Synthetic
    print(f"\n--- Synthetic data (wide_tree, {n_synth_total} nodes) ---")
    conv_synth = run_convergence(synth_graph, synth_truth, seed=RANDOM_SEED)
    all_results["exp2_convergence_synthetic"] = conv_synth

    # Print accuracy at key milestones
    milestones = [0, 5, 10, 15, 20, 25, 30]
    milestones = [m for m in milestones if m < n_synth_total]
    rows = []
    for name in ["eig", "entropy", "random", "level_order", "most_connected"]:
        accs = conv_synth[name]
        row = [name] + [format_pct(accs[m]) if m < len(accs) else "---" for m in milestones]
        rows.append(row)
    print_table(["Strategy"] + [f"Q={m}" for m in milestones], rows,
                [18] + [10] * len(milestones))

    # Questions to reach 80% and 90% accuracy
    print("\n  Questions to reach accuracy threshold:")
    for name in ["eig", "entropy", "random", "level_order", "most_connected"]:
        accs = conv_synth[name]
        q80 = next((i for i, a in enumerate(accs) if a >= 0.80), None)
        q90 = next((i for i, a in enumerate(accs) if a >= 0.90), None)
        q80_s = str(q80) if q80 is not None else ">all"
        q90_s = str(q90) if q90 is not None else ">all"
        print(f"    {name:20s}  80%: {q80_s:>5s}   90%: {q90_s:>5s}")

    plot_convergence(conv_synth, "Convergence Speed (Synthetic wide_tree)", "convergence_synthetic.png")

    # Real data
    if real_graph_filtered and real_truth:
        print(f"\n--- Real data (Ancient Greece, {len(real_truth)} nodes) ---")
        conv_real = run_convergence(real_graph_filtered, real_truth, seed=RANDOM_SEED)
        all_results["exp2_convergence_real"] = conv_real

        milestones_real = [0, 5, 10, 15, 20, 25, 30, 35]
        milestones_real = [m for m in milestones_real if m < len(real_truth)]
        rows = []
        for name in ["eig", "entropy", "random", "level_order", "most_connected"]:
            accs = conv_real[name]
            row = [name] + [format_pct(accs[m]) if m < len(accs) else "---" for m in milestones_real]
            rows.append(row)
        print_table(["Strategy"] + [f"Q={m}" for m in milestones_real], rows,
                    [18] + [10] * len(milestones_real))

        print("\n  Questions to reach accuracy threshold:")
        for name in ["eig", "entropy", "random", "level_order", "most_connected"]:
            accs = conv_real[name]
            q80 = next((i for i, a in enumerate(accs) if a >= 0.80), None)
            q90 = next((i for i, a in enumerate(accs) if a >= 0.90), None)
            q80_s = str(q80) if q80 is not None else ">all"
            q90_s = str(q90) if q90 is not None else ">all"
            print(f"    {name:20s}  80%: {q80_s:>5s}   90%: {q90_s:>5s}")

        plot_convergence(conv_real, "Convergence Speed (Real: Ancient Greece)", "convergence_real.png")

    # ==================================================================
    # EXPERIMENT 3: Topology Effects
    # ==================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Graph Topology Effects")
    print("=" * 70)

    topo_results = run_topology_experiment(seed=RANDOM_SEED)
    all_results["exp3_topology"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "convergence"}
        for k, v in topo_results.items()
    }
    # Store convergence data separately (it's large)
    all_results["exp3_topology_convergence"] = {
        k: v["convergence"] for k, v in topo_results.items()
    }

    # Summary table: questions to 80% for each strategy x topology
    print("\nQuestions to reach 80% accuracy:")
    topo_names = list(topo_results.keys())
    strat_names = ["eig", "entropy", "random", "level_order", "most_connected"]
    rows = []
    for strat in strat_names:
        row = [strat]
        for topo in topo_names:
            accs = topo_results[topo]["convergence"][strat]
            q80 = next((i for i, a in enumerate(accs) if a >= 0.80), None)
            row.append(str(q80) if q80 is not None else ">all")
        rows.append(row)
    print_table(["Strategy"] + topo_names, rows, [18] + [14] * len(topo_names))

    print("\nQuestions to reach 90% accuracy:")
    rows = []
    for strat in strat_names:
        row = [strat]
        for topo in topo_names:
            accs = topo_results[topo]["convergence"][strat]
            q90 = next((i for i, a in enumerate(accs) if a >= 0.90), None)
            row.append(str(q90) if q90 is not None else ">all")
        rows.append(row)
    print_table(["Strategy"] + topo_names, rows, [18] + [14] * len(topo_names))

    # Strategy advantage: how many fewer questions vs random to reach 80%?
    print("\nStrategy advantage (questions saved vs random to reach 80%):")
    for topo in topo_names:
        accs_eig = topo_results[topo]["convergence"]["eig"]
        accs_e = topo_results[topo]["convergence"]["entropy"]
        accs_r = topo_results[topo]["convergence"]["random"]
        q80_eig = next((i for i, a in enumerate(accs_eig) if a >= 0.80), len(accs_eig))
        q80_e = next((i for i, a in enumerate(accs_e) if a >= 0.80), len(accs_e))
        q80_r = next((i for i, a in enumerate(accs_r) if a >= 0.80), len(accs_r))
        info = topo_results[topo]
        print(f"  {topo:16s}  eig={q80_eig:2d}  entropy={q80_e:2d}  random={q80_r:2d}  "
              f"eig_saved={q80_r - q80_eig:+d}  entropy_saved={q80_r - q80_e:+d} "
              f"({info['n_nodes']}n, {info['n_edges']}e)")

    plot_topology_comparison(topo_results, "topology_comparison.png")

    # ==================================================================
    # Final accuracy at full assessment (sanity check)
    # ==================================================================
    print("\n" + "=" * 70)
    print("SANITY CHECK: Final accuracy when all nodes assessed")
    print("=" * 70)
    for topo in topo_names:
        for strat in strat_names:
            accs = topo_results[topo]["convergence"][strat]
            final = accs[-1] if accs else 0
            if final < 0.99:
                print(f"  WARNING: {topo}/{strat} final accuracy = {format_pct(final)}")
    print("  (All strategies should reach 100% when every node is assessed)")

    # ==================================================================
    # Save results
    # ==================================================================
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
