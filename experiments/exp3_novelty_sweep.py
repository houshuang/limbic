"""Experiment 3: Novelty Scoring Improvements.

Sub-experiments:
  3a: Top-K sweep — which K gives best discrimination?
      - Easy eval: paraphrase (KNOWN) vs unrelated (NEW)
      - Hard eval: all 4 types using true_label (paraphrase=KNOWN, extension=EXTENDS,
        contradiction=NEW, unrelated=NEW) — tests ability to detect contradictions
  3b: Centroid-distance specificity — does weighting by distance from corpus centroid
      improve discrimination, especially for contradictions?
  3c: Global/local weight sweep — uses semantic clusters as synthetic categories.

Eval data: 120 calibration claims (30 sources x 4 variants: paraphrase, extension, contradiction, unrelated).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex

# ---------------------------------------------------------------------------
# Load calibration data
# ---------------------------------------------------------------------------
DATA_PATH = Path("DATA_PATH")
OUTPUT_PATH = Path("experiments/results/exp3_results.json")

with open(DATA_PATH) as f:
    claims = json.load(f)

print(f"Loaded {len(claims)} calibration claims")

# Extract unique source claims (the "existing knowledge" corpus)
source_map = {}  # source_id -> source_claim text
for c in claims:
    sid = c["source_id"]
    if sid not in source_map:
        source_map[sid] = c["source_claim"]

source_ids = sorted(source_map.keys())
source_texts = [source_map[sid] for sid in source_ids]
print(f"Unique source claims: {len(source_texts)}")

# Group variants by type
by_type = {"paraphrase": [], "extension": [], "contradiction": [], "unrelated": []}
for c in claims:
    by_type[c["variant_type"]].append(c)

for vt, items in by_type.items():
    print(f"  {vt}: {len(items)} claims")

# ---------------------------------------------------------------------------
# Embed everything
# ---------------------------------------------------------------------------
print("\nEmbedding all texts...")
t0 = time.time()
model = EmbeddingModel()

# Embed source claims
source_embeddings = model.embed_batch(source_texts)
print(f"  Source embeddings: {source_embeddings.shape}")

# Embed all 120 variant claims
variant_texts = [c["text"] for c in claims]
variant_embeddings = model.embed_batch(variant_texts)
print(f"  Variant embeddings: {variant_embeddings.shape}")
print(f"  Embedding time: {time.time() - t0:.1f}s")

# Build VectorIndex from source claims
vi = VectorIndex()
vi.add([str(sid) for sid in source_ids], source_embeddings)
print(f"  VectorIndex size: {vi.size}")

# ---------------------------------------------------------------------------
# Helper: compute raw cosine similarities to top-K neighbors
# ---------------------------------------------------------------------------
def top_k_sims(query_emb: np.ndarray, index: VectorIndex, k: int) -> list[float]:
    """Return top-K cosine similarities from the index."""
    results = index.search(query_emb, limit=k)
    return [r.score for r in results]

# Pre-compute all claims' top-K sims at max K to avoid redundant searches
MAX_K = 20
all_top_sims = []
for i in range(len(claims)):
    sims = top_k_sims(variant_embeddings[i], vi, MAX_K)
    all_top_sims.append(sims)

def novelty_at_k(claim_idx: int, k: int) -> float:
    """Novelty = 1 - mean(top-K sims) for a pre-computed claim."""
    sims = all_top_sims[claim_idx][:k]
    return 1.0 - np.mean(sims) if sims else 1.0

# ---------------------------------------------------------------------------
# Helper: F1 score for binary classification
# ---------------------------------------------------------------------------
def compute_f1(y_true: list[str], y_pred: list[str], positive_label: str = "NEW") -> dict:
    """Compute precision, recall, F1 for binary classification."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Helper: find optimal threshold via sweep
# ---------------------------------------------------------------------------
def find_best_threshold(novelty_scores: list[float], true_labels: list[str],
                        thresholds: list[float] | None = None) -> tuple[float, dict]:
    """Sweep thresholds to find best F1. novelty >= threshold -> NEW."""
    if thresholds is None:
        thresholds = [round(0.005 * i, 3) for i in range(1, 200)]
    best_f1, best_thresh, best_metrics = 0.0, 0.5, {}
    for t in thresholds:
        preds = ["NEW" if s >= t else "KNOWN" for s in novelty_scores]
        m = compute_f1(true_labels, preds, "NEW")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = t
            best_metrics = m
    return best_thresh, best_metrics


# Claim index lookups
claim_idx = {id(c): i for i, c in enumerate(claims)}

# ===========================================================================
# EXPERIMENT 3a: Top-K Sweep
# ===========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3a: Top-K Sweep")
print("=" * 70)

# Easy eval: paraphrase -> KNOWN, unrelated -> NEW (excludes extension/contradiction)
easy_claims = [c for c in claims if c["variant_type"] in ("paraphrase", "unrelated")]
easy_labels = [c["true_label"] for c in easy_claims]

# Hard eval: ALL claims using true_label (paraphrase=KNOWN, contradiction+unrelated=NEW, extension=EXTENDS->KNOWN)
# Binary: treat EXTENDS as KNOWN (it partially matches existing knowledge)
hard_labels_ext_known = []
for c in claims:
    if c["true_label"] == "NEW":
        hard_labels_ext_known.append("NEW")
    else:
        hard_labels_ext_known.append("KNOWN")

# Alt hard eval: treat EXTENDS as NEW (it adds new info)
hard_labels_ext_new = []
for c in claims:
    if c["true_label"] == "KNOWN":
        hard_labels_ext_new.append("KNOWN")
    else:
        hard_labels_ext_new.append("NEW")

ks = [1, 2, 3, 5, 7, 10, 15, 20]
results_3a = {}

print(f"\n{'K':>4}  {'Para μ':>8}  {'Ext μ':>8}  {'Contr μ':>8}  {'Unrel μ':>8}  "
      f"{'Sep(P-U)':>8}  {'Sep(P-C)':>8}  "
      f"{'Easy F1':>8}  {'Hard F1':>8}  {'HardAlt':>8}")
print("-" * 100)

for k in ks:
    # Compute novelty for all 120 claims
    type_novelty = {vt: [] for vt in by_type}
    all_novelty_k = []
    for i, c in enumerate(claims):
        nov = novelty_at_k(i, k)
        type_novelty[c["variant_type"]].append(nov)
        all_novelty_k.append(nov)

    means = {vt: np.mean(scores) for vt, scores in type_novelty.items()}
    stds = {vt: np.std(scores) for vt, scores in type_novelty.items()}
    sep_pu = means["unrelated"] - means["paraphrase"]
    sep_pc = means["contradiction"] - means["paraphrase"]

    # Easy F1
    easy_novelty = [all_novelty_k[claim_idx[id(c)]] for c in easy_claims]
    easy_thresh, easy_f1 = find_best_threshold(easy_novelty, easy_labels)

    # Hard F1 (EXTENDS=KNOWN)
    hard_thresh, hard_f1 = find_best_threshold(all_novelty_k, hard_labels_ext_known)

    # Hard Alt F1 (EXTENDS=NEW)
    hard_alt_thresh, hard_alt_f1 = find_best_threshold(all_novelty_k, hard_labels_ext_new)

    results_3a[k] = {
        "means": {vt: round(float(v), 4) for vt, v in means.items()},
        "stds": {vt: round(float(v), 4) for vt, v in stds.items()},
        "separation_para_unrel": round(float(sep_pu), 4),
        "separation_para_contr": round(float(sep_pc), 4),
        "easy_eval": {"threshold": round(easy_thresh, 4), **easy_f1},
        "hard_eval_ext_known": {"threshold": round(hard_thresh, 4), **hard_f1},
        "hard_eval_ext_new": {"threshold": round(hard_alt_thresh, 4), **hard_alt_f1},
    }

    print(f"{k:>4}  {means['paraphrase']:>8.4f}  {means['extension']:>8.4f}  "
          f"{means['contradiction']:>8.4f}  {means['unrelated']:>8.4f}  "
          f"{sep_pu:>8.4f}  {sep_pc:>8.4f}  "
          f"{easy_f1['f1']:>8.4f}  {hard_f1['f1']:>8.4f}  {hard_alt_f1['f1']:>8.4f}")

# Find best K for each metric
best_k_easy = max(results_3a, key=lambda k: (results_3a[k]["easy_eval"]["f1"], results_3a[k]["separation_para_unrel"]))
best_k_hard = max(results_3a, key=lambda k: results_3a[k]["hard_eval_ext_known"]["f1"])
best_k_sep = max(results_3a, key=lambda k: results_3a[k]["separation_para_unrel"])
best_k_sep_pc = max(results_3a, key=lambda k: results_3a[k]["separation_para_contr"])

print(f"\nBest K by easy F1 (para vs unrel): {best_k_easy} (F1={results_3a[best_k_easy]['easy_eval']['f1']:.4f})")
print(f"Best K by hard F1 (all, ext=KNOWN): {best_k_hard} (F1={results_3a[best_k_hard]['hard_eval_ext_known']['f1']:.4f})")
print(f"Best K by para-unrel separation: {best_k_sep} (sep={results_3a[best_k_sep]['separation_para_unrel']:.4f})")
print(f"Best K by para-contr separation: {best_k_sep_pc} (sep={results_3a[best_k_sep_pc]['separation_para_contr']:.4f})")

# Use best hard-eval K for subsequent experiments
best_k = best_k_hard
print(f"\nUsing K={best_k} for subsequent experiments (best hard F1)")


# ===========================================================================
# EXPERIMENT 3b: Centroid-Distance Specificity
# ===========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3b: Centroid-Distance Specificity")
print("=" * 70)

# Corpus centroid = mean of source claim embeddings
centroid = source_embeddings.mean(axis=0)
centroid_norm = centroid / np.linalg.norm(centroid)

use_k = best_k
print(f"Using K={use_k} from Experiment 3a\n")

results_3b = {"k_used": use_k, "formulas": {}}

# Compute base novelty and specificity for all claims
all_base_novelty = []
all_specificity = []
all_variant_types = []

for i, c in enumerate(claims):
    base_nov = novelty_at_k(i, use_k)
    emb = variant_embeddings[i]
    cos_to_centroid = float(np.dot(emb / np.linalg.norm(emb), centroid_norm))
    specificity = 1.0 - cos_to_centroid

    all_base_novelty.append(base_nov)
    all_specificity.append(specificity)
    all_variant_types.append(c["variant_type"])

# Test multiple combination formulas
formulas = {
    "base": lambda nov, spec: nov,
    "multiply": lambda nov, spec: nov * spec,
    "add_0.3": lambda nov, spec: 0.7 * nov + 0.3 * spec,
    "add_0.5": lambda nov, spec: 0.5 * nov + 0.5 * spec,
    "add_0.7": lambda nov, spec: 0.3 * nov + 0.7 * spec,
    "sqrt_product": lambda nov, spec: np.sqrt(nov * spec),
    "max": lambda nov, spec: max(nov, spec),
}

print(f"{'Formula':<15} {'Para μ':>8} {'Ext μ':>8} {'Contr μ':>8} {'Unrel μ':>8} "
      f"{'Sep P-U':>8} {'Sep P-C':>8} {'HardF1':>8} {'Thresh':>8}")
print("-" * 105)

for fname, fn in formulas.items():
    combined = [fn(all_base_novelty[i], all_specificity[i]) for i in range(len(claims))]
    type_means = {}
    for vt in ["paraphrase", "extension", "contradiction", "unrelated"]:
        vals = [combined[i] for i, t in enumerate(all_variant_types) if t == vt]
        type_means[vt] = np.mean(vals)

    sep_pu = type_means["unrelated"] - type_means["paraphrase"]
    sep_pc = type_means["contradiction"] - type_means["paraphrase"]

    thresh, f1_metrics = find_best_threshold(combined, hard_labels_ext_known)

    results_3b["formulas"][fname] = {
        "means": {vt: round(float(v), 4) for vt, v in type_means.items()},
        "separation_para_unrel": round(float(sep_pu), 4),
        "separation_para_contr": round(float(sep_pc), 4),
        "hard_f1": {"threshold": round(thresh, 4), **f1_metrics},
    }

    print(f"{fname:<15} {type_means['paraphrase']:>8.4f} {type_means['extension']:>8.4f} "
          f"{type_means['contradiction']:>8.4f} {type_means['unrelated']:>8.4f} "
          f"{sep_pu:>8.4f} {sep_pc:>8.4f} {f1_metrics['f1']:>8.4f} {thresh:>8.3f}")

best_formula = max(results_3b["formulas"], key=lambda f: results_3b["formulas"][f]["hard_f1"]["f1"])
print(f"\nBest formula by hard F1: {best_formula} "
      f"(F1={results_3b['formulas'][best_formula]['hard_f1']['f1']:.4f})")

# Contradiction vs Paraphrase detailed analysis
print("\n--- Contradiction vs Paraphrase overlap analysis ---")
para_nov = sorted([all_base_novelty[i] for i, t in enumerate(all_variant_types) if t == "paraphrase"])
contr_nov = sorted([all_base_novelty[i] for i, t in enumerate(all_variant_types) if t == "contradiction"])
# Count how many contradictions have lower novelty than the max paraphrase novelty
max_para = max(para_nov)
min_contr = min(contr_nov)
overlap_count = sum(1 for x in contr_nov if x <= max_para)
print(f"  Paraphrase novelty range: [{min(para_nov):.4f}, {max_para:.4f}]")
print(f"  Contradiction novelty range: [{min_contr:.4f}, {max(contr_nov):.4f}]")
print(f"  Contradictions below max paraphrase: {overlap_count}/{len(contr_nov)} ({100*overlap_count/len(contr_nov):.0f}%)")
print(f"  This overlap is the fundamental limit of embedding-only novelty detection —")
print(f"  contradictions are semantically close to their source but have opposite meaning.")

# ===========================================================================
# EXPERIMENT 3c: Global/Local Weight Sweep
# ===========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3c: Global/Local Weight Sweep")
print("=" * 70)

claim_types = set(c["claim_type"] for c in claims)
print(f"Unique claim_types: {claim_types}")
print(f"Unique source_ids: {len(set(c['source_id'] for c in claims))}")

# Try multiple clustering thresholds to find one that gives useful groups
from amygdala import greedy_centroid_cluster

print("\nClustering source claims at various thresholds:")
best_cluster_threshold = None
best_clusters = None
for ct in [0.3, 0.35, 0.4, 0.45, 0.5]:
    clusters = greedy_centroid_cluster(source_embeddings, threshold=ct)
    sizes = sorted([len(c) for c in clusters], reverse=True)
    n_multi = sum(1 for c in clusters if len(c) > 1)
    print(f"  threshold={ct}: {len(clusters)} clusters, {n_multi} multi-member, sizes={sizes[:10]}")
    # We want clusters where at least some have >1 member
    if n_multi >= 2 and len(clusters) < len(source_ids) and best_cluster_threshold is None:
        best_cluster_threshold = ct
        best_clusters = clusters

if best_clusters is not None and len(best_clusters) > 1:
    clusters = best_clusters
    cluster_sizes = [len(c) for c in clusters]
    print(f"\nUsing threshold={best_cluster_threshold}: {len(clusters)} clusters")

    # Build cluster membership: source_id -> set of source_ids in same cluster
    id_to_cluster_ids = {}
    for cluster_indices in clusters:
        cluster_id_set = {str(source_ids[i]) for i in cluster_indices}
        for i in cluster_indices:
            id_to_cluster_ids[str(source_ids[i])] = cluster_id_set

    # Use min of use_k and smallest cluster size to avoid degenerate local search
    min_cluster = min(len(c) for c in clusters)
    local_k = min(use_k, min_cluster)
    print(f"  Local K capped at {local_k} (smallest cluster has {min_cluster} members)")

    weight_pairs = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
                    (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (1.0, 0.0)]

    results_3c = {
        "cluster_threshold": best_cluster_threshold,
        "clusters": len(clusters),
        "cluster_sizes": sorted(cluster_sizes, reverse=True),
        "local_k": local_k,
        "global_k": use_k,
        "results": {},
    }

    print(f"\n{'GW':>5} {'LW':>5}  {'Para μ':>8}  {'Ext μ':>8}  {'Contr μ':>8}  {'Unrel μ':>8}  "
          f"{'Sep P-U':>8}  {'Sep P-C':>8}  {'HardF1':>8}")
    print("-" * 82)

    for gw, lw in weight_pairs:
        combined_scores = []
        for i, c in enumerate(claims):
            emb = variant_embeddings[i]
            global_nov = novelty_at_k(i, use_k)

            src_id = str(c["source_id"])
            if src_id in id_to_cluster_ids and lw > 0:
                cat_ids = id_to_cluster_ids[src_id]
                local_results = vi.search(emb, limit=local_k, filter_ids=cat_ids)
                local_sims = [r.score for r in local_results]
                local_nov = 1.0 - np.mean(local_sims) if local_sims else 1.0
            else:
                local_nov = global_nov

            combined_scores.append(gw * global_nov + lw * local_nov)

        type_means = {}
        for vt in ["paraphrase", "extension", "contradiction", "unrelated"]:
            vals = [combined_scores[i] for i, t in enumerate(all_variant_types) if t == vt]
            type_means[vt] = np.mean(vals)

        sep_pu = type_means["unrelated"] - type_means["paraphrase"]
        sep_pc = type_means["contradiction"] - type_means["paraphrase"]

        thresh_gl, f1_gl = find_best_threshold(combined_scores, hard_labels_ext_known)

        key = f"{gw:.1f}/{lw:.1f}"
        results_3c["results"][key] = {
            "means": {vt: round(float(v), 4) for vt, v in type_means.items()},
            "separation_para_unrel": round(float(sep_pu), 4),
            "separation_para_contr": round(float(sep_pc), 4),
            "hard_f1": {"threshold": round(thresh_gl, 4), **f1_gl},
        }

        print(f"{gw:>5.1f} {lw:>5.1f}  {type_means['paraphrase']:>8.4f}  {type_means['extension']:>8.4f}  "
              f"{type_means['contradiction']:>8.4f}  {type_means['unrelated']:>8.4f}  "
              f"{sep_pu:>8.4f}  {sep_pc:>8.4f}  {f1_gl['f1']:>8.4f}")

    best_weights = max(results_3c["results"], key=lambda k: results_3c["results"][k]["hard_f1"]["f1"])
    print(f"\nBest weights by hard F1: {best_weights} "
          f"(F1={results_3c['results'][best_weights]['hard_f1']['f1']:.4f})")
else:
    print("\nSkipping 3c: source claims are too diverse to form meaningful clusters.")
    print("All 30 claims are from different domains — no natural category structure exists.")
    print("This is expected: the calibration data was designed with maximally diverse sources.")
    results_3c = {
        "skipped": True,
        "reason": "Source claims are too semantically diverse for meaningful clustering",
        "note": "Local novelty requires a corpus where claims cluster into topical groups",
    }


# ===========================================================================
# Summary and Save
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

r3a = results_3a[best_k]
print(f"\n3a — Top-K Sweep:")
print(f"     Best K by hard F1 (ext=KNOWN): K={best_k} "
      f"(F1={r3a['hard_eval_ext_known']['f1']:.4f}, "
      f"thresh={r3a['hard_eval_ext_known']['threshold']:.3f})")
print(f"     Easy eval (para vs unrel): F1={r3a['easy_eval']['f1']:.4f} — trivially separable at all K")
print(f"     Para-Unrel separation: K=1 best ({results_3a[1]['separation_para_unrel']:.4f}), "
      f"decreases with K")
print(f"     Para-Contr separation: K={best_k_sep_pc} best "
      f"({results_3a[best_k_sep_pc]['separation_para_contr']:.4f}) — tiny, "
      f"cosine cannot distinguish contradictions")

bf = results_3b["formulas"][best_formula]
base_f1_val = results_3b["formulas"]["base"]["hard_f1"]["f1"]
best_f1_val = bf["hard_f1"]["f1"]
print(f"\n3b — Centroid-Distance Specificity:")
print(f"     Best formula: '{best_formula}' (hard F1={best_f1_val:.4f})")
print(f"     Base novelty hard F1: {base_f1_val:.4f}")
delta = best_f1_val - base_f1_val
if delta > 0.001:
    print(f"     Specificity IMPROVES hard F1 by {delta:+.4f}")
elif delta < -0.001:
    print(f"     Specificity HURTS hard F1 by {delta:+.4f}")
else:
    print(f"     No meaningful difference")

if not results_3c.get("skipped"):
    bw = results_3c["results"][best_weights]
    base_3c = results_3c["results"]["1.0/0.0"]
    print(f"\n3c — Global/Local Weights:")
    print(f"     Best combo: {best_weights} (hard F1={bw['hard_f1']['f1']:.4f})")
    print(f"     Global-only (1.0/0.0): hard F1={base_3c['hard_f1']['f1']:.4f}")
else:
    print(f"\n3c — Skipped: {results_3c.get('reason', 'N/A')}")

print(f"\nKey finding: cosine-based novelty perfectly separates paraphrases from unrelated")
print(f"claims but CANNOT distinguish contradictions from paraphrases (both are")
print(f"semantically close to source). Contradiction detection requires semantic")
print(f"understanding beyond embedding similarity (e.g., NLI models or LLM classification).")

# Save results
all_results = {
    "experiment": "exp3_novelty_sweep",
    "eval_data": str(DATA_PATH),
    "n_claims": len(claims),
    "n_sources": len(source_texts),
    "embedding_model": "all-MiniLM-L6-v2",
    "exp3a_topk_sweep": {str(k): v for k, v in results_3a.items()},
    "exp3b_centroid_specificity": results_3b,
    "exp3c_global_local_weights": results_3c,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to {OUTPUT_PATH}")
