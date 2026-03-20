import os
"""Experiment 13: Similarity Graph Layer — Does graph traversal add value beyond vector search?

txtai builds a similarity graph where documents are nodes and edges connect documents
above a cosine threshold. This enables graph-based exploration: BFS neighborhoods,
bridge detection, community discovery, knowledge trails.

This experiment tests whether these graph operations surface discoveries that plain
vector search misses, using ~2.5K education claims from claims database.

Key question: Does graph neighborhood at depth 2 retrieve items NOT in top-50
vector search that ARE still topically relevant? If yes, a graph layer adds value.
"""

import json
import sqlite3
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex, greedy_centroid_cluster

OTAK_DB = Path(os.environ.get("AMYGDALA_EVAL_DB", "eval_claims.db"))
RESULTS_PATH = Path("experiments/results/exp13_results.json")

SAMPLE_SIZE = 2500
THRESHOLDS = [0.6, 0.7, 0.8]
N_SEEDS = 20
RNG = np.random.default_rng(42)


# --- Data Loading ---

def load_claims(n: int) -> list[dict]:
    """Load n random claims from claims database with id and text."""
    conn = sqlite3.connect(str(OTAK_DB))
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT n.id, n.name FROM idx_knowledge_item_claim_type k "
        "JOIN nodes n ON n.id = k.node_id "
        "WHERE n.deleted_at IS NULL AND n.name IS NOT NULL "
        "AND length(n.name) > 20 AND length(n.name) < 500 "
        "ORDER BY RANDOM() LIMIT ?",
        (n,),
    )
    rows = cur.fetchall()
    conn.close()
    claims = [{"id": r[0], "text": r[1]} for r in rows]
    # Deduplicate by id (the join can produce duplicates)
    seen = set()
    unique = []
    for c in claims:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
    print(f"Loaded {len(unique)} unique claims from claims database")
    return unique


# --- Graph Construction ---

def build_adjacency(embeddings: np.ndarray, threshold: float) -> dict[int, list[tuple[int, float]]]:
    """Build adjacency list: for each node, neighbors with cosine >= threshold.

    Uses batched matrix multiplication for efficiency on 2.5K claims.
    """
    n = len(embeddings)
    # Compute full similarity matrix (2500x2500 = ~24MB for float32, fine)
    sim_matrix = embeddings @ embeddings.T

    adj = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                adj[i].append((j, float(sim_matrix[i, j])))
                adj[j].append((i, float(sim_matrix[i, j])))
    return dict(adj), sim_matrix


def graph_stats(adj: dict, n_total: int) -> dict:
    """Basic graph statistics."""
    n_nodes_with_edges = len(adj)
    n_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
    degrees = [len(adj.get(i, [])) for i in range(n_total)]
    return {
        "n_nodes_with_edges": n_nodes_with_edges,
        "n_isolated": n_total - n_nodes_with_edges,
        "n_edges": n_edges,
        "mean_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(np.max(degrees)),
        "pct_connected": n_nodes_with_edges / n_total * 100,
    }


# --- Graph Operations ---

def bfs_neighborhood(adj: dict, seed: int, max_depth: int) -> dict[int, int]:
    """BFS from seed node. Returns {node_id: depth} for all reachable nodes."""
    visited = {seed: 0}
    queue = deque([(seed, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor, _ in adj.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
    return visited


def connected_components(adj: dict, n_total: int) -> list[set[int]]:
    """Find connected components using BFS."""
    visited = set()
    components = []
    for i in range(n_total):
        if i in visited:
            continue
        if i not in adj:
            # Isolated node — skip (or count as singleton)
            visited.add(i)
            continue
        component = set()
        queue = deque([i])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            visited.add(node)
            for neighbor, _ in adj.get(node, []):
                if neighbor not in component:
                    queue.append(neighbor)
        if len(component) >= 2:
            components.append(component)
    return components


def approximate_betweenness(adj: dict, n_total: int, n_samples: int = 100) -> np.ndarray:
    """Approximate betweenness centrality by sampling BFS from n_samples sources.

    True betweenness requires all-pairs shortest paths (O(n^2)). We sample
    n_samples random sources and count how often each node appears on shortest
    paths from those sources.
    """
    centrality = np.zeros(n_total, dtype=np.float64)
    sources = RNG.choice(n_total, size=min(n_samples, n_total), replace=False)

    for s in sources:
        # BFS from source
        dist = {s: 0}
        pred = defaultdict(list)  # predecessors on shortest paths
        sigma = defaultdict(float)  # number of shortest paths
        sigma[s] = 1.0
        queue = deque([s])
        order = []

        while queue:
            v = queue.popleft()
            order.append(v)
            for w, _ in adj.get(v, []):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist.get(w) == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Accumulate dependencies (Brandes' algorithm)
        delta = defaultdict(float)
        for v in reversed(order):
            for p in pred[v]:
                delta[p] += (sigma[p] / sigma[v]) * (1.0 + delta[v])
            if v != s:
                centrality[v] += delta[v]

    # Normalize
    if n_samples > 0:
        centrality /= n_samples
    return centrality


def shortest_path(adj: dict, source: int, target: int) -> list[int] | None:
    """Find shortest path between two nodes using BFS. Returns node list or None."""
    if source == target:
        return [source]
    visited = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor, _ in adj.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = node
                if neighbor == target:
                    # Reconstruct path
                    path = [target]
                    current = target
                    while visited[current] is not None:
                        current = visited[current]
                        path.append(current)
                    return list(reversed(path))
                queue.append(neighbor)
    return None


def label_propagation(adj: dict, n_total: int, max_iter: int = 20) -> list[int]:
    """Simple label propagation community detection.

    Each node starts with its own label. Iteratively, each node adopts the
    most frequent label among its neighbors. Converges to communities.
    """
    labels = list(range(n_total))
    nodes = list(range(n_total))

    for iteration in range(max_iter):
        RNG.shuffle(nodes)
        changed = 0
        for node in nodes:
            if node not in adj or not adj[node]:
                continue
            neighbor_labels = [labels[n] for n, _ in adj[node]]
            if not neighbor_labels:
                continue
            # Most frequent label among neighbors
            label_counts = defaultdict(int)
            for nl in neighbor_labels:
                label_counts[nl] += 1
            max_count = max(label_counts.values())
            candidates = [l for l, c in label_counts.items() if c == max_count]
            # Break ties randomly
            new_label = candidates[RNG.integers(len(candidates))]
            if labels[node] != new_label:
                labels[node] = new_label
                changed += 1
        if changed == 0:
            break
    return labels


# --- Comparison Metrics ---

def vector_search_top_k(embeddings: np.ndarray, seed_idx: int, k: int) -> list[int]:
    """Get top-K most similar items by cosine (excluding self)."""
    sims = embeddings @ embeddings[seed_idx]
    sims[seed_idx] = -2.0  # exclude self
    if k >= len(sims):
        return list(np.argsort(-sims))
    top_k = np.argpartition(-sims, k)[:k]
    return list(top_k[np.argsort(-sims[top_k])])


def mean_pairwise_distance(embeddings: np.ndarray, indices: list[int]) -> float:
    """Mean pairwise cosine distance among a set of items. Higher = more diverse."""
    if len(indices) < 2:
        return 0.0
    subset = embeddings[indices]
    sim_matrix = subset @ subset.T
    n = len(indices)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(1.0 - sim_matrix[mask].mean())


def compare_graph_vs_vector(
    adj: dict,
    embeddings: np.ndarray,
    sim_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    seed_indices: list[int],
) -> dict:
    """Compare graph neighborhood (depth 2) vs vector search (top-10, top-50).

    Metrics:
    - diversity: mean pairwise distance of results
    - coverage: number of unique clusters touched
    - serendipity: fraction of graph results NOT in top-50 vector but cosine > 0.4
    """
    results = []
    for seed in seed_indices:
        # Vector search baselines
        vec_top10 = vector_search_top_k(embeddings, seed, 10)
        vec_top50 = vector_search_top_k(embeddings, seed, 50)
        vec_top50_set = set(vec_top50)

        # Graph neighborhood at depth 2
        neighborhood = bfs_neighborhood(adj, seed, max_depth=2)
        graph_nodes = [n for n, d in neighborhood.items() if n != seed]

        if not graph_nodes:
            results.append({
                "seed": seed,
                "graph_size": 0,
                "vec10_size": len(vec_top10),
                "vec50_size": len(vec_top50),
                "graph_diversity": 0.0,
                "vec10_diversity": mean_pairwise_distance(embeddings, vec_top10),
                "graph_coverage": 0,
                "vec10_coverage": len(set(cluster_labels[vec_top10])),
                "serendipity_count": 0,
                "serendipity_frac": 0.0,
                "graph_only_count": 0,
                "overlap_with_top50": 0,
            })
            continue

        # Diversity
        graph_diversity = mean_pairwise_distance(embeddings, graph_nodes[:50])  # cap at 50 for fairness
        vec10_diversity = mean_pairwise_distance(embeddings, vec_top10)

        # Coverage (unique clusters touched)
        graph_coverage = len(set(cluster_labels[graph_nodes]))
        vec10_coverage = len(set(cluster_labels[vec_top10]))

        # Serendipity: graph items NOT in top-50 vector search but cosine > 0.4 to seed
        graph_only = [n for n in graph_nodes if n not in vec_top50_set]
        serendipitous = [n for n in graph_only if sim_matrix[seed, n] > 0.4]

        # Overlap
        overlap = len(set(graph_nodes) & vec_top50_set)

        results.append({
            "seed": seed,
            "graph_size": len(graph_nodes),
            "vec10_size": len(vec_top10),
            "vec50_size": len(vec_top50),
            "graph_diversity": graph_diversity,
            "vec10_diversity": vec10_diversity,
            "graph_coverage": graph_coverage,
            "vec10_coverage": vec10_coverage,
            "serendipity_count": len(serendipitous),
            "serendipity_frac": len(serendipitous) / len(graph_nodes) if graph_nodes else 0,
            "graph_only_count": len(graph_only),
            "overlap_with_top50": overlap,
        })

    # Aggregate
    agg = {}
    for key in results[0]:
        if key == "seed":
            continue
        values = [r[key] for r in results]
        agg[f"mean_{key}"] = float(np.mean(values))
        agg[f"median_{key}"] = float(np.median(values))

    return {"per_seed": results, "aggregate": agg}


# --- Knowledge Trails ---

def find_distant_pairs(sim_matrix: np.ndarray, n_pairs: int = 10) -> list[tuple[int, int, float]]:
    """Find pairs with low cosine similarity (distant topics)."""
    n = sim_matrix.shape[0]
    pairs = []
    # Sample random pairs and pick the most distant ones
    idx_a = RNG.integers(0, n, size=1000)
    idx_b = RNG.integers(0, n, size=1000)
    for a, b in zip(idx_a, idx_b):
        if a != b:
            pairs.append((int(a), int(b), float(sim_matrix[a, b])))
    pairs.sort(key=lambda p: p[2])
    return pairs[:n_pairs]


def evaluate_knowledge_trails(
    adj: dict,
    claims: list[dict],
    sim_matrix: np.ndarray,
) -> list[dict]:
    """Find shortest paths between distant claims and evaluate as 'knowledge trails'."""
    distant_pairs = find_distant_pairs(sim_matrix, n_pairs=10)
    trails = []

    for source, target, direct_sim in distant_pairs:
        path = shortest_path(adj, source, target)
        if path is None:
            trails.append({
                "source_text": claims[source]["text"][:100],
                "target_text": claims[target]["text"][:100],
                "direct_cosine": direct_sim,
                "path_length": None,
                "reachable": False,
            })
            continue

        # Compute consecutive cosines along path
        step_cosines = []
        for i in range(len(path) - 1):
            step_cosines.append(float(sim_matrix[path[i], path[i + 1]]))

        trail = {
            "source_text": claims[source]["text"][:100],
            "target_text": claims[target]["text"][:100],
            "direct_cosine": direct_sim,
            "path_length": len(path) - 1,
            "min_step_cosine": min(step_cosines),
            "mean_step_cosine": float(np.mean(step_cosines)),
            "reachable": True,
            "steps": [
                {"text": claims[path[i]]["text"][:80], "cosine_to_next": step_cosines[i] if i < len(step_cosines) else None}
                for i in range(len(path))
            ],
        }
        trails.append(trail)

    return trails


# --- Main ---

def main():
    t0 = time.time()

    # Load and embed claims
    claims = load_claims(SAMPLE_SIZE)
    texts = [c["text"] for c in claims]
    n = len(claims)

    print(f"\nEmbedding {n} claims...")
    model = EmbeddingModel()
    embeddings = model.embed_batch(texts)
    print(f"  Embedding shape: {embeddings.shape}")

    # Normalize (should already be, but be safe)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    # Pre-compute clustering for coverage metric
    print("\nClustering with greedy_centroid_cluster (threshold=0.75)...")
    clusters = greedy_centroid_cluster(embeddings, threshold=0.75)
    cluster_labels = np.full(n, -1, dtype=int)
    for ci, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_labels[idx] = ci
    # Assign isolated nodes their own cluster ids
    next_label = len(clusters)
    for i in range(n):
        if cluster_labels[i] == -1:
            cluster_labels[i] = next_label
            next_label += 1
    print(f"  {len(clusters)} clusters with 2+ members, {next_label} total labels")

    # Select seed nodes for comparison
    seed_indices = list(RNG.choice(n, size=N_SEEDS, replace=False))

    all_results = {}

    for threshold in THRESHOLDS:
        print(f"\n{'='*80}")
        print(f"THRESHOLD = {threshold}")
        print(f"{'='*80}")

        # Build graph
        t_graph = time.time()
        adj, sim_matrix = build_adjacency(embeddings, threshold)
        build_time = time.time() - t_graph
        print(f"  Graph built in {build_time:.1f}s")

        # Graph stats
        stats = graph_stats(adj, n)
        print(f"  Nodes with edges: {stats['n_nodes_with_edges']} ({stats['pct_connected']:.1f}%)")
        print(f"  Edges: {stats['n_edges']}")
        print(f"  Mean degree: {stats['mean_degree']:.1f}, Max degree: {stats['max_degree']}")
        print(f"  Isolated nodes: {stats['n_isolated']}")

        # Connected components
        components = connected_components(adj, n)
        component_sizes = sorted([len(c) for c in components], reverse=True)
        print(f"  Connected components: {len(components)}")
        if component_sizes:
            print(f"  Largest 5 components: {component_sizes[:5]}")

        # --- Test 1: Knowledge Neighborhoods ---
        print(f"\n  --- Knowledge Neighborhoods (BFS depth 2) ---")
        neighborhood_sizes = []
        for seed in seed_indices[:5]:
            hood = bfs_neighborhood(adj, seed, max_depth=2)
            hood_size = len(hood) - 1  # exclude seed
            vec10 = vector_search_top_k(embeddings, seed, 10)
            neighborhood_sizes.append(hood_size)
            print(f"    Seed '{texts[seed][:60]}...'")
            print(f"      BFS depth-2: {hood_size} items, Vector top-10: {len(vec10)} items")

            # Depth breakdown
            depth_counts = defaultdict(int)
            for _, d in hood.items():
                depth_counts[d] += 1
            print(f"      Depth 0: {depth_counts[0]}, Depth 1: {depth_counts[1]}, Depth 2: {depth_counts[2]}")

        print(f"    Mean neighborhood size (depth 2): {np.mean(neighborhood_sizes):.0f}")

        # --- Test 2: Bridge Detection ---
        print(f"\n  --- Bridge Detection (approximate betweenness) ---")
        t_bridge = time.time()
        centrality = approximate_betweenness(adj, n, n_samples=200)
        bridge_time = time.time() - t_bridge
        print(f"    Betweenness computed in {bridge_time:.1f}s")

        top_bridges = np.argsort(-centrality)[:10]
        bridge_claims = []
        for rank, idx in enumerate(top_bridges[:5]):
            idx = int(idx)
            bc = float(centrality[idx])
            degree = len(adj.get(idx, []))
            print(f"    #{rank+1} (betweenness={bc:.4f}, degree={degree}): {texts[idx][:100]}")
            bridge_claims.append({
                "rank": rank + 1,
                "index": idx,
                "text": texts[idx],
                "betweenness": bc,
                "degree": degree,
            })

        # --- Test 3: Community Detection ---
        print(f"\n  --- Community Detection (label propagation) ---")
        lp_labels = label_propagation(adj, n, max_iter=30)
        lp_communities = defaultdict(list)
        for i, label in enumerate(lp_labels):
            lp_communities[label].append(i)
        # Filter to communities with 2+ members
        lp_communities = {k: v for k, v in lp_communities.items() if len(v) >= 2}
        lp_sizes = sorted([len(v) for v in lp_communities.values()], reverse=True)
        print(f"    Label propagation communities (2+ members): {len(lp_communities)}")
        if lp_sizes:
            print(f"    Largest 5: {lp_sizes[:5]}")

        # Compare to greedy centroid clusters
        # How many greedy clusters are split across LP communities?
        cluster_to_lp = defaultdict(set)
        for i in range(n):
            cl = cluster_labels[i]
            lp = lp_labels[i]
            cluster_to_lp[cl].add(lp)
        multi_lp = sum(1 for lps in cluster_to_lp.values() if len(lps) > 1)
        print(f"    Greedy clusters split across LP communities: {multi_lp}/{len(cluster_to_lp)}")

        # Show a sample LP community
        if lp_communities:
            largest_lp_key = max(lp_communities, key=lambda k: len(lp_communities[k]))
            largest_lp = lp_communities[largest_lp_key]
            print(f"    Largest LP community ({len(largest_lp)} members), sample claims:")
            for idx in largest_lp[:5]:
                print(f"      - {texts[idx][:90]}")

        # --- Test 4: Knowledge Trails ---
        print(f"\n  --- Knowledge Trails (shortest paths between distant claims) ---")
        trails = evaluate_knowledge_trails(adj, claims, sim_matrix)
        reachable_trails = [t for t in trails if t["reachable"]]
        unreachable_trails = [t for t in trails if not t["reachable"]]
        print(f"    Reachable: {len(reachable_trails)}/{len(trails)}")
        for trail in reachable_trails[:3]:
            print(f"    Path ({trail['path_length']} steps, direct cosine={trail['direct_cosine']:.3f}):")
            print(f"      From: {trail['source_text']}")
            print(f"      To:   {trail['target_text']}")
            if "steps" in trail:
                for step in trail["steps"]:
                    cos_str = f" (cos→next: {step['cosine_to_next']:.3f})" if step["cosine_to_next"] is not None else ""
                    print(f"        → {step['text']}{cos_str}")

        # --- Test 5: Graph vs Vector Comparison ---
        print(f"\n  --- Graph vs Vector Search Comparison ---")
        comparison = compare_graph_vs_vector(
            adj, embeddings, sim_matrix, cluster_labels, seed_indices
        )
        agg = comparison["aggregate"]
        print(f"    Mean graph neighborhood size: {agg['mean_graph_size']:.1f}")
        print(f"    Mean vec-10 diversity: {agg['mean_vec10_diversity']:.4f}")
        print(f"    Mean graph diversity:  {agg['mean_graph_diversity']:.4f}")
        print(f"    Mean vec-10 coverage (clusters): {agg['mean_vec10_coverage']:.1f}")
        print(f"    Mean graph coverage (clusters):  {agg['mean_graph_coverage']:.1f}")
        print(f"    Mean serendipity count: {agg['mean_serendipity_count']:.1f}")
        print(f"    Mean serendipity fraction: {agg['mean_serendipity_frac']:.4f}")
        print(f"    Mean overlap with top-50: {agg['mean_overlap_with_top50']:.1f}")
        print(f"    Mean graph-only items: {agg['mean_graph_only_count']:.1f}")

        # Collect threshold results
        all_results[str(threshold)] = {
            "threshold": threshold,
            "graph_stats": stats,
            "build_time_seconds": build_time,
            "n_connected_components": len(components),
            "component_sizes_top5": component_sizes[:5],
            "bridge_claims": bridge_claims,
            "bridge_time_seconds": bridge_time,
            "lp_n_communities": len(lp_communities),
            "lp_community_sizes_top5": lp_sizes[:5],
            "greedy_clusters_split_by_lp": multi_lp,
            "trails": [
                {k: v for k, v in t.items() if k != "steps"}
                for t in trails
            ],
            "trails_reachable": len(reachable_trails),
            "trails_total": len(trails),
            "comparison": comparison["aggregate"],
        }

    # --- Final Analysis ---
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS")
    print(f"{'='*80}")

    print("\nThreshold comparison:")
    print(f"  {'Threshold':<12} {'Connected%':>10} {'Edges':>8} {'MeanDeg':>8} {'Components':>10} "
          f"{'GraphDiv':>9} {'Vec10Div':>9} {'Serendipity':>12} {'Coverage':>9}")
    print(f"  {'-'*100}")
    for th_str, res in all_results.items():
        gs = res["graph_stats"]
        comp = res["comparison"]
        print(f"  {th_str:<12} {gs['pct_connected']:>9.1f}% {gs['n_edges']:>8} {gs['mean_degree']:>8.1f} "
              f"{res['n_connected_components']:>10} "
              f"{comp['mean_graph_diversity']:>9.4f} {comp['mean_vec10_diversity']:>9.4f} "
              f"{comp['mean_serendipity_count']:>12.1f} {comp['mean_graph_coverage']:>9.1f}")

    # Key finding: does graph add value?
    print("\nKey findings:")
    for th_str, res in all_results.items():
        comp = res["comparison"]
        serendipity = comp["mean_serendipity_count"]
        graph_size = comp["mean_graph_size"]
        diversity_delta = comp["mean_graph_diversity"] - comp["mean_vec10_diversity"]
        coverage_delta = comp["mean_graph_coverage"] - comp["mean_vec10_coverage"]
        print(f"\n  Threshold {th_str}:")
        print(f"    Graph neighborhood is {graph_size:.0f} items (vs 10 for vector search)")
        print(f"    Diversity delta (graph - vec10): {diversity_delta:+.4f}")
        print(f"    Coverage delta (graph - vec10): {coverage_delta:+.1f} clusters")
        print(f"    Serendipitous discoveries: {serendipity:.1f} items (relevant but not in top-50)")
        if serendipity > 0 and graph_size > 0:
            print(f"    Serendipity rate: {serendipity/graph_size*100:.1f}% of graph neighborhood")
        trails_reachable = res["trails_reachable"]
        print(f"    Knowledge trails: {trails_reachable}/{res['trails_total']} distant pairs reachable")

    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    output = {
        "experiment": "exp13_similarity_graph",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "sample_size": len(claims),
        "n_seeds": N_SEEDS,
        "thresholds_tested": THRESHOLDS,
        "elapsed_seconds": elapsed,
        "n_greedy_clusters": len(clusters),
        "results_by_threshold": all_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
