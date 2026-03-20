"""Experiment 8: The Karpathy Loop — automated parameter optimization for amygdala.

Sweeps across 120 configurations of:
  - Embedding model (2): paraphrase-multilingual-MiniLM-L12-v2, all-MiniLM-L6-v2
  - Genericize (2): False, True
  - Whiten dims (3): None, 128, 256
  - Top-K (5): 1, 3, 5, 7, 10
  - Centroid specificity (2): False, True

Eval data: 120 calibration claims (30 sources x 4 variants: paraphrase, extension,
contradiction, unrelated) from claims database.

Composite score: 0.4 * separation + 0.3 * accuracy + 0.3 * contradiction_detection

Optimization: embeds texts ONCE per (model, genericize, whiten_dims) combo, then sweeps
novelty params (top_k, centroid_spec) without re-embedding.

Target runtime: under 5 minutes.
"""

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex
from amygdala.novelty import novelty_score, corpus_centroid

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path("DATA_PATH")
OUTPUT_PATH = Path("experiments/results/exp8_results.json")

# ---------------------------------------------------------------------------
# Load calibration data
# ---------------------------------------------------------------------------
with open(DATA_PATH) as f:
    claims = json.load(f)

print(f"Loaded {len(claims)} calibration claims")

# Extract unique source claims (the "existing knowledge" corpus)
source_map = {}
for c in claims:
    sid = c["source_id"]
    if sid not in source_map:
        source_map[sid] = c["source_claim"]

source_ids = sorted(source_map.keys())
source_texts = [source_map[sid] for sid in source_ids]
print(f"Unique source claims: {len(source_texts)}")

# Group claims by variant type
by_type = {"paraphrase": [], "extension": [], "contradiction": [], "unrelated": []}
for i, c in enumerate(claims):
    by_type[c["variant_type"]].append(i)

variant_texts = [c["text"] for c in claims]

for vt, indices in by_type.items():
    print(f"  {vt}: {len(indices)} claims")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_accuracy_at_best_threshold(novelty_scores_list, true_labels, thresholds=None):
    """Sweep thresholds, return best accuracy. novelty >= threshold -> NEW."""
    if thresholds is None:
        thresholds = [round(0.005 * i, 3) for i in range(1, 200)]
    best_acc = 0.0
    best_thresh = 0.5
    for t in thresholds:
        preds = ["NEW" if s >= t else "KNOWN" for s in novelty_scores_list]
        correct = sum(1 for p, l in zip(preds, true_labels) if p == l)
        acc = correct / len(true_labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_acc, best_thresh


# True labels: paraphrase + extension = KNOWN, contradiction + unrelated = NEW
true_labels = []
for c in claims:
    if c["variant_type"] in ("paraphrase", "extension"):
        true_labels.append("KNOWN")
    else:
        true_labels.append("NEW")


# ---------------------------------------------------------------------------
# Phase 1: Pre-compute embeddings for each (model, genericize, whiten_dims)
# ---------------------------------------------------------------------------
MODELS = ["paraphrase-multilingual-MiniLM-L12-v2", "all-MiniLM-L6-v2"]
GENERICIZE_OPTS = [False, True]
WHITEN_DIMS_OPTS = [None, 128, 256]
TOP_K_OPTS = [1, 3, 5, 7, 10]
CENTROID_SPEC_OPTS = [False, True]

# Cache embeddings keyed by (model, genericize, whiten_dims)
embedding_cache = {}

print("\n" + "=" * 70)
print("PHASE 1: Embedding (one-time per model+genericize+whiten combo)")
print("=" * 70)

total_embed_combos = len(MODELS) * len(GENERICIZE_OPTS) * len(WHITEN_DIMS_OPTS)
embed_idx = 0
t_start = time.time()

for model_name in MODELS:
    for genericize in GENERICIZE_OPTS:
        for whiten_dims in WHITEN_DIMS_OPTS:
            embed_idx += 1
            key = (model_name, genericize, whiten_dims)
            short_model = model_name.split("/")[-1][:20]
            label = f"{short_model} gen={genericize} wh={whiten_dims}"
            print(f"  [{embed_idx}/{total_embed_combos}] {label} ...", end=" ", flush=True)

            t0 = time.time()
            em = EmbeddingModel(
                model_name=model_name,
                whiten_dims=whiten_dims,
                genericize=genericize,
            )

            # Get raw embeddings first (before whitening)
            source_embs = em._raw_embed_batch(source_texts)
            variant_embs = em._raw_embed_batch(variant_texts)

            # Fit whitening if requested (on source claims)
            if whiten_dims is not None:
                all_texts_for_whitening = source_texts
                em.fit_whitening(all_texts_for_whitening)
                # Apply whitening
                source_embs = em._apply_whitening(source_embs)
                variant_embs = em._apply_whitening(variant_embs)

            embedding_cache[key] = {
                "source_embs": source_embs,
                "variant_embs": variant_embs,
            }

            dt = time.time() - t0
            print(f"dim={source_embs.shape[1]}, {dt:.1f}s")

t_embed = time.time() - t_start
print(f"\nTotal embedding time: {t_embed:.1f}s")


# ---------------------------------------------------------------------------
# Phase 2: Sweep novelty parameters for each embedding combo
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 2: Novelty parameter sweep (120 configurations)")
print("=" * 70)

all_results = []
total_configs = (
    len(MODELS) * len(GENERICIZE_OPTS) * len(WHITEN_DIMS_OPTS)
    * len(TOP_K_OPTS) * len(CENTROID_SPEC_OPTS)
)
config_idx = 0
t_sweep_start = time.time()

for model_name in MODELS:
    for genericize in GENERICIZE_OPTS:
        for whiten_dims in WHITEN_DIMS_OPTS:
            key = (model_name, genericize, whiten_dims)
            source_embs = embedding_cache[key]["source_embs"]
            variant_embs = embedding_cache[key]["variant_embs"]

            # Build VectorIndex from source claims
            vi = VectorIndex()
            vi.add([str(sid) for sid in source_ids], source_embs)

            # Pre-compute top-K similarities at max K for this embedding
            MAX_K = max(TOP_K_OPTS)
            all_top_sims = []
            for i in range(len(claims)):
                results = vi.search(variant_embs[i], limit=MAX_K)
                sims = [r.score for r in results]
                all_top_sims.append(sims)

            # Pre-compute centroid
            centroid = corpus_centroid(vi)

            for top_k in TOP_K_OPTS:
                for use_centroid_spec in CENTROID_SPEC_OPTS:
                    config_idx += 1

                    # Compute novelty scores
                    novelty_scores = []
                    for i in range(len(claims)):
                        score = novelty_score(
                            variant_embs[i],
                            vi,
                            top_k=top_k,
                            centroid=centroid,
                            use_centroid_specificity=use_centroid_spec,
                        )
                        novelty_scores.append(score)

                    # Compute per-type means
                    type_means = {}
                    for vt, indices in by_type.items():
                        type_means[vt] = float(np.mean([novelty_scores[i] for i in indices]))

                    # Metric 1: Paraphrase-unrelated separation (higher = better)
                    separation = type_means["unrelated"] - type_means["paraphrase"]

                    # Metric 2: Classification accuracy at optimal threshold
                    accuracy, best_thresh = compute_accuracy_at_best_threshold(
                        novelty_scores, true_labels
                    )

                    # Metric 3: Contradiction detection
                    # contradictions should have HIGHER novelty than paraphrases
                    contradiction_gap = type_means["contradiction"] - type_means["paraphrase"]

                    # Normalize metrics to [0, 1] for composite scoring
                    # separation: theoretical range ~[-1, 1], practical [0, 0.8]
                    norm_separation = max(0, separation)  # negative separation is useless
                    # accuracy: already in [0, 1]
                    norm_accuracy = accuracy
                    # contradiction_gap: can be negative; normalize
                    norm_contradiction = max(0, contradiction_gap)

                    # Composite score
                    composite = (
                        0.4 * norm_separation
                        + 0.3 * norm_accuracy
                        + 0.3 * norm_contradiction
                    )

                    result = {
                        "model": model_name,
                        "genericize": genericize,
                        "whiten_dims": whiten_dims,
                        "top_k": top_k,
                        "use_centroid_spec": use_centroid_spec,
                        "type_means": {vt: round(v, 4) for vt, v in type_means.items()},
                        "separation": round(separation, 4),
                        "accuracy": round(accuracy, 4),
                        "accuracy_threshold": round(best_thresh, 4),
                        "contradiction_gap": round(contradiction_gap, 4),
                        "composite_score": round(composite, 4),
                    }
                    all_results.append(result)

t_sweep = time.time() - t_sweep_start
print(f"Swept {config_idx} configurations in {t_sweep:.1f}s")


# ---------------------------------------------------------------------------
# Phase 3: Analyze and report
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 3: Results Analysis")
print("=" * 70)

# Sort by composite score
all_results.sort(key=lambda r: r["composite_score"], reverse=True)

# Top 10
print("\n--- TOP 10 CONFIGURATIONS ---")
print(f"{'#':>3}  {'Model':<25} {'Gen':>4} {'Wh':>5} {'K':>3} {'Cent':>5}  "
      f"{'Sep':>6} {'Acc':>6} {'Contr':>6} {'Composite':>9}")
print("-" * 95)
for i, r in enumerate(all_results[:10]):
    short_model = r["model"].replace("paraphrase-multilingual-MiniLM-L12-v2", "multilingual").replace("all-MiniLM-L6-v2", "minilm-l6")
    wh = str(r["whiten_dims"]) if r["whiten_dims"] else "raw"
    print(f"{i+1:>3}  {short_model:<25} {str(r['genericize']):>4} {wh:>5} {r['top_k']:>3} {str(r['use_centroid_spec']):>5}  "
          f"{r['separation']:>6.3f} {r['accuracy']:>6.1%} {r['contradiction_gap']:>6.3f} {r['composite_score']:>9.4f}")

# Bottom 10
print("\n--- BOTTOM 10 CONFIGURATIONS ---")
print(f"{'#':>3}  {'Model':<25} {'Gen':>4} {'Wh':>5} {'K':>3} {'Cent':>5}  "
      f"{'Sep':>6} {'Acc':>6} {'Contr':>6} {'Composite':>9}")
print("-" * 95)
for i, r in enumerate(all_results[-10:]):
    rank = len(all_results) - 9 + i
    short_model = r["model"].replace("paraphrase-multilingual-MiniLM-L12-v2", "multilingual").replace("all-MiniLM-L6-v2", "minilm-l6")
    wh = str(r["whiten_dims"]) if r["whiten_dims"] else "raw"
    print(f"{rank:>3}  {short_model:<25} {str(r['genericize']):>4} {wh:>5} {r['top_k']:>3} {str(r['use_centroid_spec']):>5}  "
          f"{r['separation']:>6.3f} {r['accuracy']:>6.1%} {r['contradiction_gap']:>6.3f} {r['composite_score']:>9.4f}")

# Optimal configuration
best = all_results[0]
print("\n" + "=" * 70)
print("OPTIMAL CONFIGURATION")
print("=" * 70)
print(f"  Model:               {best['model']}")
print(f"  Genericize:          {best['genericize']}")
print(f"  Whiten dims:         {best['whiten_dims']}")
print(f"  Top-K:               {best['top_k']}")
print(f"  Centroid specificity: {best['use_centroid_spec']}")
print(f"  ---")
print(f"  Separation (P-U):    {best['separation']:.4f}")
print(f"  Accuracy:            {best['accuracy']:.1%}")
print(f"  Threshold:           {best['accuracy_threshold']:.3f}")
print(f"  Contradiction gap:   {best['contradiction_gap']:.4f}")
print(f"  Composite score:     {best['composite_score']:.4f}")
print(f"  ---")
print(f"  Per-type novelty means:")
for vt in ["paraphrase", "extension", "contradiction", "unrelated"]:
    print(f"    {vt:<15}: {best['type_means'][vt]:.4f}")

# Factor analysis: which parameter matters most?
print("\n" + "=" * 70)
print("FACTOR ANALYSIS: Mean composite score by parameter value")
print("=" * 70)

factors = {
    "model": lambda r: r["model"].replace("paraphrase-multilingual-MiniLM-L12-v2", "multilingual").replace("all-MiniLM-L6-v2", "minilm-l6"),
    "genericize": lambda r: str(r["genericize"]),
    "whiten_dims": lambda r: str(r["whiten_dims"]) if r["whiten_dims"] else "raw",
    "top_k": lambda r: str(r["top_k"]),
    "centroid_spec": lambda r: str(r["use_centroid_spec"]),
}

for factor_name, key_fn in factors.items():
    groups = {}
    for r in all_results:
        k = key_fn(r)
        groups.setdefault(k, []).append(r["composite_score"])

    print(f"\n  {factor_name}:")
    sorted_groups = sorted(groups.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for k, scores in sorted_groups:
        print(f"    {k:<30} mean={np.mean(scores):.4f}  std={np.std(scores):.4f}  n={len(scores)}")

# Impact analysis: which factor has the largest range of mean composite scores?
print("\n  Factor impact (range of means):")
impacts = []
for factor_name, key_fn in factors.items():
    groups = {}
    for r in all_results:
        k = key_fn(r)
        groups.setdefault(k, []).append(r["composite_score"])
    means = [np.mean(v) for v in groups.values()]
    impact = max(means) - min(means)
    impacts.append((factor_name, impact))
impacts.sort(key=lambda x: x[1], reverse=True)
for fname, imp in impacts:
    print(f"    {fname:<20} {imp:.4f}")

# Current amygdala defaults vs optimal
print("\n" + "=" * 70)
print("CURRENT DEFAULTS vs OPTIMAL")
print("=" * 70)

# Find the result matching current defaults
current_defaults = {
    "model": "paraphrase-multilingual-MiniLM-L12-v2",
    "genericize": False,
    "whiten_dims": None,
    "use_centroid_spec": False,
}
# Current adaptive K for 30 sources = 1 (since <= 50)
for r in all_results:
    if (r["model"] == current_defaults["model"]
        and r["genericize"] == current_defaults["genericize"]
        and r["whiten_dims"] == current_defaults["whiten_dims"]
        and r["top_k"] == 1
        and r["use_centroid_spec"] == current_defaults["use_centroid_spec"]):
        current = r
        break
else:
    current = None

if current:
    print(f"  Current defaults (adaptive K=1 for 30 items):")
    print(f"    Composite: {current['composite_score']:.4f}  "
          f"(rank {all_results.index(current) + 1}/{len(all_results)})")
    print(f"    Sep={current['separation']:.4f}  "
          f"Acc={current['accuracy']:.1%}  "
          f"Contr={current['contradiction_gap']:.4f}")

    delta = best["composite_score"] - current["composite_score"]
    print(f"\n  Optimal configuration:")
    print(f"    Composite: {best['composite_score']:.4f}  (rank 1/{len(all_results)})")
    print(f"    Delta: {delta:+.4f} ({100*delta/current['composite_score']:+.1f}%)")

    # What changed?
    diffs = []
    if best["model"] != current["model"]:
        diffs.append(f"model: {current['model']} -> {best['model']}")
    if best["genericize"] != current["genericize"]:
        diffs.append(f"genericize: {current['genericize']} -> {best['genericize']}")
    if best["whiten_dims"] != current["whiten_dims"]:
        diffs.append(f"whiten_dims: {current['whiten_dims']} -> {best['whiten_dims']}")
    if best["top_k"] != 1:
        diffs.append(f"top_k: 1 -> {best['top_k']}")
    if best["use_centroid_spec"] != current["use_centroid_spec"]:
        diffs.append(f"centroid_spec: {current['use_centroid_spec']} -> {best['use_centroid_spec']}")
    if diffs:
        print(f"\n  Parameter changes needed:")
        for d in diffs:
            print(f"    - {d}")
    else:
        print(f"\n  Current defaults are already optimal!")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
total_time = time.time() - t_start
print(f"\n{'=' * 70}")
print(f"Total runtime: {total_time:.1f}s")
print(f"{'=' * 70}")

output = {
    "experiment": "exp8_karpathy_loop",
    "description": "Automated parameter optimization across 120 configurations",
    "eval_data": str(DATA_PATH),
    "n_claims": len(claims),
    "n_sources": len(source_texts),
    "composite_formula": "0.4 * separation + 0.3 * accuracy + 0.3 * contradiction_gap",
    "parameters_swept": {
        "models": MODELS,
        "genericize": GENERICIZE_OPTS,
        "whiten_dims": WHITEN_DIMS_OPTS,
        "top_k": TOP_K_OPTS,
        "centroid_spec": CENTROID_SPEC_OPTS,
    },
    "total_configs": total_configs,
    "embedding_time_s": round(t_embed, 1),
    "sweep_time_s": round(t_sweep, 1),
    "total_time_s": round(total_time, 1),
    "optimal": best,
    "rankings": all_results,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved to {OUTPUT_PATH}")
