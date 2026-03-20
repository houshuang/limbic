"""Experiment 10: Temporal Decay for Novelty Scoring.

Hypothesis: Weighting neighbors by recency (exp(-lambda * age_days)) improves
novelty ranking for time-ordered data. Older items should contribute less to
the "already known" signal — a topic revisited after a long gap should score
as more novel than the same topic repeated the next day.

Approach:
1. Use real time-ordered data from conversation-search chunks (user queries
   with timestamps spanning ~6 weeks).
2. Create ground truth: for each item at time T, compute its cosine similarity
   to ALL prior items. An item is "truly novel" if it introduces a topic not
   seen recently (high sim to OLD items but low sim to RECENT items should
   still score as novel). An item is "truly known" if it closely matches
   something seen in the last few days.
3. Compare standard novelty (no decay) vs temporal novelty (with decay) at
   various lambda values.
4. Measure: separation between novel/known items, ranking quality (Spearman
   correlation with ground truth novelty ordering).

Lambda sweep: [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
  - 0.001 = very slow decay (half-life ~693 days)
  - 0.01  = moderate decay (half-life ~69 days)
  - 0.05  = fast decay (half-life ~14 days)
  - 0.2   = very fast decay (half-life ~3.5 days)
"""

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

from amygdala import EmbeddingModel, VectorIndex

OUTPUT_PATH = Path("experiments/results/exp10_results.json")

# ---------------------------------------------------------------------------
# 1. Load real time-ordered data from conversation-search
# ---------------------------------------------------------------------------
print("=" * 70)
print("EXPERIMENT 10: Temporal Decay for Novelty Scoring")
print("=" * 70)

DB_PATH = Path.home() / ".conversation-search" / "index.db"
conn = sqlite3.connect(str(DB_PATH))

rows = conn.execute("""
    SELECT timestamp, user_content
    FROM chunks
    WHERE timestamp IS NOT NULL
    AND user_content IS NOT NULL
    AND length(user_content) > 50
    AND length(user_content) < 2000
    AND user_content NOT LIKE '%<command%'
    AND user_content NOT LIKE '%<task-%'
    AND user_content NOT LIKE '%<teammate-%'
    ORDER BY timestamp
""").fetchall()
conn.close()

print(f"Loaded {len(rows)} time-ordered text chunks")
print(f"Date range: {rows[0][0][:10]} to {rows[-1][0][:10]}")

# Parse timestamps and texts
timestamps = []
texts = []
for ts_str, content in rows:
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        timestamps.append(ts)
        texts.append(content.strip())
    except (ValueError, AttributeError):
        continue

print(f"Parsed {len(texts)} entries with valid timestamps")

# Subsample if too many (embedding is expensive) — take every Nth to preserve temporal spread
MAX_ITEMS = 500
if len(texts) > MAX_ITEMS:
    step = len(texts) // MAX_ITEMS
    indices = list(range(0, len(texts), step))[:MAX_ITEMS]
    texts = [texts[i] for i in indices]
    timestamps = [timestamps[i] for i in indices]
    print(f"Subsampled to {len(texts)} items (every {step}th)")

# Compute days-since-epoch for each item (relative to earliest)
t0 = timestamps[0]
days_from_start = [(t - t0).total_seconds() / 86400 for t in timestamps]
print(f"Time span: {days_from_start[-1]:.1f} days")

# ---------------------------------------------------------------------------
# 2. Embed all texts
# ---------------------------------------------------------------------------
print("\nEmbedding texts...")
t_start = time.time()
model = EmbeddingModel()
embeddings = model.embed_batch(texts)
print(f"Embedded {len(texts)} texts in {time.time() - t_start:.1f}s")
print(f"Embedding shape: {embeddings.shape}")

# ---------------------------------------------------------------------------
# 3. Create ground truth labels
# ---------------------------------------------------------------------------
# For each item i, we look at all prior items [0..i-1].
# "Truly novel" = item introduces a topic not seen in the RECENT past
#   (defined as last RECENCY_WINDOW days), even if similar to older items.
# "Truly known" = item closely matches something from the recent past.
#
# Ground truth novelty score:
#   - Find max cosine sim to items within RECENCY_WINDOW days
#   - If max_recent_sim > 0.7: item is KNOWN (repeating recent topic)
#   - If max_recent_sim < 0.4: item is NOVEL (new topic)
#   - In between: AMBIGUOUS (excluded from binary eval)
#
# The KEY insight: standard novelty treats all neighbors equally, so an old
# item with high similarity suppresses novelty even when that topic hasn't
# been mentioned in weeks. Temporal decay should fix this.

RECENCY_WINDOW_DAYS = 7.0
KNOWN_THRESHOLD = 0.70
NOVEL_THRESHOLD = 0.45
MIN_HISTORY = 20  # Need enough prior items for meaningful scoring

print(f"\nCreating ground truth (recency window={RECENCY_WINDOW_DAYS} days)...")

# Normalize embeddings for cosine sim
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
normed = embeddings / norms

ground_truth = []  # (index, label, max_recent_sim, max_old_sim)
all_eval_indices = []

for i in range(MIN_HISTORY, len(texts)):
    current_day = days_from_start[i]

    # Cosine similarities to all prior items
    prior_sims = normed[:i] @ normed[i]
    prior_days = np.array(days_from_start[:i])
    ages = current_day - prior_days  # age in days

    # Split into recent (within window) and old (outside window)
    recent_mask = ages <= RECENCY_WINDOW_DAYS
    old_mask = ages > RECENCY_WINDOW_DAYS

    max_recent_sim = float(prior_sims[recent_mask].max()) if recent_mask.any() else 0.0
    max_old_sim = float(prior_sims[old_mask].max()) if old_mask.any() else 0.0

    if max_recent_sim >= KNOWN_THRESHOLD:
        label = "KNOWN"
    elif max_recent_sim <= NOVEL_THRESHOLD:
        label = "NOVEL"
    else:
        label = "AMBIGUOUS"

    ground_truth.append({
        "index": i,
        "label": label,
        "max_recent_sim": round(max_recent_sim, 4),
        "max_old_sim": round(max_old_sim, 4),
        "day": round(current_day, 2),
    })
    all_eval_indices.append(i)

n_known = sum(1 for g in ground_truth if g["label"] == "KNOWN")
n_novel = sum(1 for g in ground_truth if g["label"] == "NOVEL")
n_ambig = sum(1 for g in ground_truth if g["label"] == "AMBIGUOUS")
print(f"Ground truth: {n_known} KNOWN, {n_novel} NOVEL, {n_ambig} AMBIGUOUS")
print(f"Total evaluable: {n_known + n_novel} (excluding ambiguous)")

# ---------------------------------------------------------------------------
# 4. Compute novelty scores: standard vs temporal decay
# ---------------------------------------------------------------------------
# For each item i, build an index from all prior items [0..i-1] and compute
# novelty. For temporal decay, multiply each neighbor's similarity by
# exp(-lambda * age_days) before the 1 - mean(top-K) calculation.

LAMBDAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

print(f"\nComputing novelty for {len(all_eval_indices)} items x {len(LAMBDAS)} lambda values...")
t_start = time.time()

# Pre-compute all pairwise similarities and ages for eval items
# For each eval item i, we need top-K sims from prior items
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)

# Store results: lambda -> list of novelty scores (one per eval item)
novelty_by_lambda = {lam: [] for lam in LAMBDAS}
# Also store per-K results for the best lambda
novelty_by_k_lambda = {}  # (k, lambda) -> scores

for eval_idx, gt in enumerate(ground_truth):
    i = gt["index"]
    current_day = days_from_start[i]

    # Cosine similarities to all prior items
    prior_sims = normed[:i] @ normed[i]
    prior_days = np.array(days_from_start[:i])
    ages = current_day - prior_days

    for lam in LAMBDAS:
        if lam == 0.0:
            # Standard novelty: no decay
            weighted_sims = prior_sims.copy()
        else:
            # Temporal decay: multiply similarity by recency weight
            decay = np.exp(-lam * ages)
            weighted_sims = prior_sims * decay

        # Top-K: use adaptive K based on prior count
        k = min(MAX_K, len(weighted_sims))
        if k == 0:
            novelty_by_lambda[lam].append(1.0)
            continue

        # Sort by weighted similarity (descending), take top K
        top_k_sims = np.sort(weighted_sims)[-k:]
        novelty = 1.0 - float(np.mean(top_k_sims))
        novelty_by_lambda[lam].append(novelty)

elapsed = time.time() - t_start
print(f"Computed in {elapsed:.1f}s")

# ---------------------------------------------------------------------------
# 5. Evaluate: separation, ranking quality, binary classification
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Filter to non-ambiguous items for binary evaluation
binary_mask = [g["label"] != "AMBIGUOUS" for g in ground_truth]
binary_labels = [g["label"] for g in ground_truth if g["label"] != "AMBIGUOUS"]

# Continuous ground truth: use (1 - max_recent_sim) as "true novelty"
# Higher = more novel
true_novelty = [1.0 - g["max_recent_sim"] for g in ground_truth]

results = {}
print(f"\n{'Lambda':>8}  {'KNOWN_μ':>8}  {'NOVEL_μ':>8}  {'Sep':>8}  {'Spearman':>9}  {'p-value':>10}  {'AUC':>6}  {'BestF1':>7}  {'Thresh':>7}")
print("-" * 95)

for lam in LAMBDAS:
    scores = novelty_by_lambda[lam]

    # Mean novelty for KNOWN vs NOVEL
    known_scores = [s for s, g in zip(scores, ground_truth) if g["label"] == "KNOWN"]
    novel_scores = [s for s, g in zip(scores, ground_truth) if g["label"] == "NOVEL"]

    mean_known = np.mean(known_scores) if known_scores else 0
    mean_novel = np.mean(novel_scores) if novel_scores else 0
    separation = mean_novel - mean_known

    # Spearman rank correlation with true novelty
    spearman_r, spearman_p = stats.spearmanr(scores, true_novelty)

    # Simple AUC: proportion of (novel, known) pairs where novel > known
    n_correct = 0
    n_total = 0
    for ns in novel_scores:
        for ks in known_scores:
            n_total += 1
            if ns > ks:
                n_correct += 1
            elif ns == ks:
                n_correct += 0.5
    auc = n_correct / n_total if n_total > 0 else 0.5

    # Best F1 via threshold sweep
    binary_scores = [s for s, m in zip(scores, binary_mask) if m]
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.005):
        preds = ["NOVEL" if s >= thresh else "KNOWN" for s in binary_scores]
        tp = sum(1 for t, p in zip(binary_labels, preds) if t == "NOVEL" and p == "NOVEL")
        fp = sum(1 for t, p in zip(binary_labels, preds) if t == "KNOWN" and p == "NOVEL")
        fn = sum(1 for t, p in zip(binary_labels, preds) if t == "NOVEL" and p == "KNOWN")
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    results[str(lam)] = {
        "lambda": lam,
        "half_life_days": round(np.log(2) / lam, 1) if lam > 0 else float("inf"),
        "mean_known": round(float(mean_known), 4),
        "mean_novel": round(float(mean_novel), 4),
        "separation": round(float(separation), 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 6),
        "auc": round(float(auc), 4),
        "best_f1": round(float(best_f1), 4),
        "best_threshold": round(float(best_thresh), 4),
        "n_known": len(known_scores),
        "n_novel": len(novel_scores),
    }

    half_life = f"{np.log(2)/lam:.0f}d" if lam > 0 else "inf"
    label = f"{lam} ({half_life})"
    print(f"{label:>15}  {mean_known:>8.4f}  {mean_novel:>8.4f}  {separation:>8.4f}  "
          f"{spearman_r:>9.4f}  {spearman_p:>10.6f}  {auc:>6.4f}  {best_f1:>7.4f}  {best_thresh:>7.3f}")

# ---------------------------------------------------------------------------
# 6. Detailed analysis: where does temporal decay help?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DETAILED ANALYSIS: Where does temporal decay help?")
print("=" * 70)

# Find items where temporal decay changes the classification
# Compare lambda=0 (no decay) vs best lambda
best_lam_key = max(results, key=lambda k: results[k]["auc"])
best_lam = results[best_lam_key]["lambda"]
print(f"\nBest lambda by AUC: {best_lam} (half-life: {results[best_lam_key]['half_life_days']} days)")

baseline_scores = novelty_by_lambda[0.0]
best_scores = novelty_by_lambda[best_lam]

# Find items where decay helps: true NOVEL items ranked higher, true KNOWN ranked lower
improvements = []
degradations = []

for idx, gt in enumerate(ground_truth):
    if gt["label"] == "AMBIGUOUS":
        continue
    baseline = baseline_scores[idx]
    decayed = best_scores[idx]
    delta = decayed - baseline

    if gt["label"] == "NOVEL" and delta > 0.02:
        improvements.append((idx, gt, baseline, decayed, delta))
    elif gt["label"] == "KNOWN" and delta < -0.02:
        improvements.append((idx, gt, baseline, decayed, delta))
    elif gt["label"] == "NOVEL" and delta < -0.02:
        degradations.append((idx, gt, baseline, decayed, delta))
    elif gt["label"] == "KNOWN" and delta > 0.02:
        degradations.append((idx, gt, baseline, decayed, delta))

print(f"\nItems where decay HELPS (correct direction, delta > 0.02): {len(improvements)}")
for idx, gt, base, dec, delta in improvements[:5]:
    i = gt["index"]
    print(f"  [{gt['label']}] day={gt['day']:.0f} base={base:.3f} decay={dec:.3f} delta={delta:+.3f}")
    print(f"    max_recent_sim={gt['max_recent_sim']:.3f} max_old_sim={gt['max_old_sim']:.3f}")
    print(f"    text: {texts[i][:100]}...")

print(f"\nItems where decay HURTS (wrong direction, delta > 0.02): {len(degradations)}")
for idx, gt, base, dec, delta in degradations[:5]:
    i = gt["index"]
    print(f"  [{gt['label']}] day={gt['day']:.0f} base={base:.3f} decay={dec:.3f} delta={delta:+.3f}")
    print(f"    max_recent_sim={gt['max_recent_sim']:.3f} max_old_sim={gt['max_old_sim']:.3f}")
    print(f"    text: {texts[i][:100]}...")

# ---------------------------------------------------------------------------
# 7. Analyze the "topic return" pattern specifically
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TOPIC RETURN ANALYSIS")
print("=" * 70)
print("Items that match something OLD but not RECENT (the decay sweet spot)")

topic_returns = []
for idx, gt in enumerate(ground_truth):
    if gt["max_old_sim"] > 0.6 and gt["max_recent_sim"] < 0.4:
        topic_returns.append((idx, gt))

print(f"\nFound {len(topic_returns)} topic-return items (old_sim>0.6, recent_sim<0.4)")
if topic_returns:
    print("\nEffect of decay on topic-return items:")
    for lam in [0.0, 0.01, 0.05, 0.1]:
        scores = [novelty_by_lambda[lam][idx] for idx, _ in topic_returns]
        print(f"  lambda={lam}: mean novelty = {np.mean(scores):.4f} (should be HIGH — these are returning topics)")

# Items that match something RECENT (should be KNOWN regardless)
recent_repeats = []
for idx, gt in enumerate(ground_truth):
    if gt["max_recent_sim"] > 0.7:
        recent_repeats.append((idx, gt))

print(f"\nFound {len(recent_repeats)} recent-repeat items (recent_sim>0.7)")
if recent_repeats:
    print("Effect of decay on recent-repeat items:")
    for lam in [0.0, 0.01, 0.05, 0.1]:
        scores = [novelty_by_lambda[lam][idx] for idx, _ in recent_repeats]
        print(f"  lambda={lam}: mean novelty = {np.mean(scores):.4f} (should be LOW — these are repeats)")

# ---------------------------------------------------------------------------
# 8. K sensitivity analysis at best lambda
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"K SENSITIVITY AT BEST LAMBDA ({best_lam})")
print("=" * 70)

k_results = {}
print(f"\n{'K':>4}  {'KNOWN_μ':>8}  {'NOVEL_μ':>8}  {'Sep':>8}  {'AUC':>6}  {'F1':>6}")
print("-" * 50)

for k in K_VALUES:
    k_scores = []
    for eval_idx, gt in enumerate(ground_truth):
        i = gt["index"]
        current_day = days_from_start[i]
        prior_sims = normed[:i] @ normed[i]
        prior_days = np.array(days_from_start[:i])
        ages = current_day - prior_days

        if best_lam == 0.0:
            weighted_sims = prior_sims.copy()
        else:
            decay = np.exp(-best_lam * ages)
            weighted_sims = prior_sims * decay

        actual_k = min(k, len(weighted_sims))
        top_k = np.sort(weighted_sims)[-actual_k:]
        k_scores.append(1.0 - float(np.mean(top_k)))

    known_k = [s for s, g in zip(k_scores, ground_truth) if g["label"] == "KNOWN"]
    novel_k = [s for s, g in zip(k_scores, ground_truth) if g["label"] == "NOVEL"]

    sep = np.mean(novel_k) - np.mean(known_k)

    n_correct = sum(1 for ns in novel_k for ks in known_k if ns > ks)
    n_ties = sum(0.5 for ns in novel_k for ks in known_k if ns == ks)
    auc = (n_correct + n_ties) / (len(novel_k) * len(known_k)) if novel_k and known_k else 0.5

    binary_k = [s for s, m in zip(k_scores, binary_mask) if m]
    best_f1_k = 0
    for thresh in np.arange(0.1, 0.9, 0.005):
        preds = ["NOVEL" if s >= thresh else "KNOWN" for s in binary_k]
        tp = sum(1 for t, p in zip(binary_labels, preds) if t == "NOVEL" and p == "NOVEL")
        fp = sum(1 for t, p in zip(binary_labels, preds) if t == "KNOWN" and p == "NOVEL")
        fn = sum(1 for t, p in zip(binary_labels, preds) if t == "NOVEL" and p == "KNOWN")
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        best_f1_k = max(best_f1_k, f1)

    k_results[k] = {
        "separation": round(float(sep), 4),
        "auc": round(float(auc), 4),
        "f1": round(float(best_f1_k), 4),
    }
    print(f"{k:>4}  {np.mean(known_k):>8.4f}  {np.mean(novel_k):>8.4f}  {sep:>8.4f}  {auc:>6.4f}  {best_f1_k:>6.4f}")

# ---------------------------------------------------------------------------
# 9. Summary and save
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

baseline = results["0.0"]
best = results[best_lam_key]

print(f"\nBaseline (no decay):    AUC={baseline['auc']:.4f}  F1={baseline['best_f1']:.4f}  "
      f"Separation={baseline['separation']:.4f}  Spearman={baseline['spearman_r']:.4f}")
print(f"Best (λ={best_lam}):  AUC={best['auc']:.4f}  F1={best['best_f1']:.4f}  "
      f"Separation={best['separation']:.4f}  Spearman={best['spearman_r']:.4f}  "
      f"Half-life={best['half_life_days']}d")

auc_delta = best["auc"] - baseline["auc"]
f1_delta = best["best_f1"] - baseline["best_f1"]
sep_delta = best["separation"] - baseline["separation"]
spearman_delta = best["spearman_r"] - baseline["spearman_r"]

print(f"\nDeltas:  AUC={auc_delta:+.4f}  F1={f1_delta:+.4f}  "
      f"Sep={sep_delta:+.4f}  Spearman={spearman_delta:+.4f}")

if auc_delta > 0.02:
    recommendation = "ADD temporal decay to novelty.py"
    reason = f"AUC improves by {auc_delta:.4f} with λ={best_lam}"
elif auc_delta > 0.005:
    recommendation = "OPTIONAL: add as opt-in parameter"
    reason = f"Small but positive AUC improvement ({auc_delta:+.4f})"
elif auc_delta > -0.005:
    recommendation = "NEUTRAL: no meaningful effect"
    reason = f"AUC change is negligible ({auc_delta:+.4f})"
else:
    recommendation = "DO NOT ADD: temporal decay hurts performance"
    reason = f"AUC decreases by {abs(auc_delta):.4f}"

print(f"\nRecommendation: {recommendation}")
print(f"Reason: {reason}")

# Also note: if topic returns exist, check if decay helps those specifically
if topic_returns:
    baseline_tr = np.mean([novelty_by_lambda[0.0][idx] for idx, _ in topic_returns])
    best_tr = np.mean([novelty_by_lambda[best_lam][idx] for idx, _ in topic_returns])
    print(f"\nTopic-return items ({len(topic_returns)} items):")
    print(f"  Baseline novelty: {baseline_tr:.4f}")
    print(f"  With decay: {best_tr:.4f} (delta={best_tr - baseline_tr:+.4f})")
    if best_tr > baseline_tr + 0.01:
        print(f"  Decay correctly boosts novelty for returning topics")
    else:
        print(f"  Decay does NOT help for returning topics")

# Save full results
output = {
    "experiment": "exp10_temporal_decay",
    "description": "Test temporal decay (exp(-lambda*age_days)) for novelty scoring",
    "data_source": "conversation-search chunks",
    "n_items": len(texts),
    "n_eval_items": len(ground_truth),
    "n_known": n_known,
    "n_novel": n_novel,
    "n_ambiguous": n_ambig,
    "time_span_days": round(days_from_start[-1], 1),
    "recency_window_days": RECENCY_WINDOW_DAYS,
    "known_threshold": KNOWN_THRESHOLD,
    "novel_threshold": NOVEL_THRESHOLD,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "lambda_sweep": results,
    "best_lambda": best_lam,
    "best_lambda_key": best_lam_key,
    "k_sensitivity": k_results,
    "topic_return_count": len(topic_returns),
    "recommendation": recommendation,
    "reason": reason,
    "deltas": {
        "auc": round(float(auc_delta), 4),
        "f1": round(float(f1_delta), 4),
        "separation": round(float(sep_delta), 4),
        "spearman": round(float(spearman_delta), 4),
    },
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_PATH}")
