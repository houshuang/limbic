import os
"""Experiment 15: Novelty Scoring at Real Scale (~27K Claims).

Previous experiments (1-14) used small datasets (30-500 items). This experiment
validates novelty scoring on a full ~27K education claims corpus:

1. How do novelty score distributions look at real scale?
2. Does centroid specificity help or hurt on a large homogeneous corpus?
3. Does temporal decay (lambda=0.02) shift the distribution meaningfully?
4. What's the per-call and batch performance at this scale?
5. Do the lowest/highest novelty claims make intuitive sense?
"""

import json
import random
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex, novelty_score, batch_novelty, corpus_centroid

OTAK_DB = Path(os.environ.get("AMYGDALA_EVAL_DB", "eval_claims.db"))
OUTPUT_PATH = Path("experiments/results/exp15_results.json")
SEED = 42
SAMPLE_N = 100  # claims for per-config scoring
BATCH_SIZES = [100, 1000]  # batch_novelty perf test sizes

print("=" * 70)
print("EXPERIMENT 15: Novelty Scoring at Real Scale (~27K Claims)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Load all claims from claims database
# ---------------------------------------------------------------------------
print("\n1. Loading claims from claims database...")
conn = sqlite3.connect(str(OTAK_DB))
rows = conn.execute("""
    SELECT n.id, n.name, n.created_at
    FROM nodes n
    JOIN idx_knowledge_item_item_type t ON n.id = t.node_id
    WHERE t.value = 'claim'
      AND n.deleted_at IS NULL
      AND n.version = (SELECT MAX(n2.version) FROM nodes n2 WHERE n2.id = n.id)
      AND length(n.name) > 10
      AND length(n.name) < 1000
    ORDER BY n.created_at
""").fetchall()
conn.close()

claim_ids = [r[0] for r in rows]
claim_texts = [r[1] for r in rows]
claim_dates = [r[2] for r in rows]

print(f"   Loaded {len(claim_ids)} claims")
print(f"   Date range: {claim_dates[0][:10]} to {claim_dates[-1][:10]}")
print(f"   Text length: min={min(len(t) for t in claim_texts)}, "
      f"max={max(len(t) for t in claim_texts)}, "
      f"mean={np.mean([len(t) for t in claim_texts]):.0f}")

# ---------------------------------------------------------------------------
# 2. Embed all claims in batches
# ---------------------------------------------------------------------------
print("\n2. Embedding all claims...")
model = EmbeddingModel()
BATCH_SIZE = 512
all_embeddings = []
t_embed_start = time.time()

for i in range(0, len(claim_texts), BATCH_SIZE):
    batch = claim_texts[i:i + BATCH_SIZE]
    emb = model.embed_batch(batch)
    all_embeddings.append(emb)
    done = min(i + BATCH_SIZE, len(claim_texts))
    elapsed = time.time() - t_embed_start
    rate = done / elapsed if elapsed > 0 else 0
    print(f"   {done}/{len(claim_texts)} ({done*100/len(claim_texts):.0f}%) - "
          f"{rate:.0f} claims/sec", end="\r")

embeddings = np.vstack(all_embeddings)
t_embed_total = time.time() - t_embed_start
print(f"\n   Embedded {len(claim_texts)} claims in {t_embed_total:.1f}s "
      f"({len(claim_texts)/t_embed_total:.0f} claims/sec)")
print(f"   Embedding shape: {embeddings.shape}")

# ---------------------------------------------------------------------------
# 3. Build VectorIndex with all embeddings
# ---------------------------------------------------------------------------
print("\n3. Building VectorIndex...")
t_index_start = time.time()
index = VectorIndex()
index.add(claim_ids, embeddings)
t_index = time.time() - t_index_start
print(f"   Index size: {index.size} items, built in {t_index:.2f}s")

# ---------------------------------------------------------------------------
# 4. Compute corpus centroid
# ---------------------------------------------------------------------------
print("\n4. Computing corpus centroid...")
t_centroid_start = time.time()
centroid = corpus_centroid(index)
t_centroid = time.time() - t_centroid_start
print(f"   Centroid computed in {t_centroid*1000:.1f}ms")
print(f"   Centroid norm: {np.linalg.norm(centroid):.4f}")

# ---------------------------------------------------------------------------
# 5. Compute timestamps (age in days from created_at)
# ---------------------------------------------------------------------------
print("\n5. Computing timestamps...")
now = datetime.fromisoformat("2026-03-19T00:00:00")
ages = {}
for cid, dt_str in zip(claim_ids, claim_dates):
    try:
        dt = datetime.fromisoformat(dt_str)
        age_days = (now - dt).total_seconds() / 86400.0
        ages[cid] = max(age_days, 0.0)
    except (ValueError, TypeError):
        ages[cid] = 0.0

age_values = list(ages.values())
print(f"   Age range: {min(age_values):.1f} to {max(age_values):.1f} days")
print(f"   Mean age: {np.mean(age_values):.1f} days, median: {np.median(age_values):.1f} days")

# ---------------------------------------------------------------------------
# 6. Select 100 random claims for detailed scoring
# ---------------------------------------------------------------------------
rng = random.Random(SEED)
sample_indices = rng.sample(range(len(claim_ids)), SAMPLE_N)
sample_ids = [claim_ids[i] for i in sample_indices]
sample_embeddings = embeddings[sample_indices]
sample_texts = [claim_texts[i] for i in sample_indices]

print(f"\n6. Selected {SAMPLE_N} random claims for per-config scoring")

# ---------------------------------------------------------------------------
# 7. Test novelty scoring: 4 configurations
# ---------------------------------------------------------------------------
configs = {
    "baseline": {"use_centroid_specificity": False, "timestamps": None, "decay_lambda": 0.0},
    "centroid": {"use_centroid_specificity": True, "timestamps": None, "decay_lambda": 0.0},
    "temporal": {"use_centroid_specificity": False, "timestamps": ages, "decay_lambda": 0.02},
    "combined": {"use_centroid_specificity": True, "timestamps": ages, "decay_lambda": 0.02},
}

results = {}

for config_name, kwargs in configs.items():
    print(f"\n7{chr(ord('a') + list(configs).index(config_name))}. Scoring: {config_name}...")
    scores = []
    t_start = time.time()
    for i, (emb, sid) in enumerate(zip(sample_embeddings, sample_ids)):
        s = novelty_score(
            emb, index,
            centroid=centroid,
            use_centroid_specificity=kwargs["use_centroid_specificity"],
            timestamps=kwargs["timestamps"],
            decay_lambda=kwargs["decay_lambda"],
        )
        scores.append(s)
    t_elapsed = time.time() - t_start
    ms_per_call = (t_elapsed / len(scores)) * 1000

    arr = np.array(scores)
    quartiles = np.percentile(arr, [25, 50, 75])
    stats = {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "q25": round(float(quartiles[0]), 4),
        "median": round(float(quartiles[1]), 4),
        "q75": round(float(quartiles[2]), 4),
        "ms_per_call": round(ms_per_call, 2),
        "total_time_s": round(t_elapsed, 2),
    }
    results[config_name] = {"stats": stats, "scores": [round(s, 4) for s in scores]}

    print(f"   Mean={stats['mean']:.4f}  Std={stats['std']:.4f}  "
          f"Min={stats['min']:.4f}  Max={stats['max']:.4f}")
    print(f"   Q25={stats['q25']:.4f}  Median={stats['median']:.4f}  Q75={stats['q75']:.4f}")
    print(f"   Time: {ms_per_call:.2f}ms/call ({t_elapsed:.2f}s total)")

# ---------------------------------------------------------------------------
# 8. Distribution comparison table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DISTRIBUTION COMPARISON")
print("=" * 70)
print(f"\n{'Config':<12} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} "
      f"{'Q25':>7} {'Med':>7} {'Q75':>7} {'ms/call':>8}")
print("-" * 82)
for name, data in results.items():
    s = data["stats"]
    print(f"{name:<12} {s['mean']:>7.4f} {s['std']:>7.4f} {s['min']:>7.4f} {s['max']:>7.4f} "
          f"{s['q25']:>7.4f} {s['median']:>7.4f} {s['q75']:>7.4f} {s['ms_per_call']:>8.2f}")

# ---------------------------------------------------------------------------
# 9. batch_novelty performance
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BATCH NOVELTY PERFORMANCE")
print("=" * 70)

batch_perf = {}
for n in BATCH_SIZES:
    batch_embs = embeddings[:n]
    batch_ids_arg = claim_ids[:n]
    print(f"\n   batch_novelty({n} claims)...", end=" ", flush=True)
    t_start = time.time()
    batch_scores = batch_novelty(batch_embs, index)
    t_elapsed = time.time() - t_start
    ms_per = (t_elapsed / n) * 1000
    print(f"{t_elapsed:.2f}s ({ms_per:.2f}ms/claim)")
    batch_perf[str(n)] = {
        "n": n,
        "total_s": round(t_elapsed, 2),
        "ms_per_claim": round(ms_per, 2),
        "mean_score": round(float(np.mean(batch_scores)), 4),
    }

# Full corpus batch (all claims) — if feasible
print(f"\n   batch_novelty(ALL {len(claim_ids)} claims)...", end=" ", flush=True)
t_start = time.time()
all_scores = batch_novelty(embeddings, index)
t_all = time.time() - t_start
ms_per_all = (t_all / len(claim_ids)) * 1000
print(f"{t_all:.1f}s ({ms_per_all:.2f}ms/claim)")
batch_perf["all"] = {
    "n": len(claim_ids),
    "total_s": round(t_all, 2),
    "ms_per_claim": round(ms_per_all, 2),
    "mean_score": round(float(np.mean(all_scores)), 4),
}

all_scores_arr = np.array(all_scores)
all_q = np.percentile(all_scores_arr, [5, 25, 50, 75, 95])
print(f"\n   Full corpus novelty distribution (N={len(all_scores)}):")
print(f"   Mean={all_scores_arr.mean():.4f}  Std={all_scores_arr.std():.4f}")
print(f"   P5={all_q[0]:.4f}  Q25={all_q[1]:.4f}  Median={all_q[2]:.4f}  "
      f"Q75={all_q[3]:.4f}  P95={all_q[4]:.4f}")

# ---------------------------------------------------------------------------
# 10. Sanity check: 5 lowest + 5 highest novelty claims
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SANITY CHECK: Most Redundant vs Most Novel Claims")
print("=" * 70)

sorted_indices = np.argsort(all_scores_arr)

print("\n--- 5 LOWEST NOVELTY (most redundant) ---")
lowest_claims = []
for rank, idx in enumerate(sorted_indices[:5]):
    score = all_scores[idx]
    text = claim_texts[idx]
    cid = claim_ids[idx]
    print(f"  {rank+1}. [{score:.4f}] {text[:120]}")
    lowest_claims.append({"rank": rank+1, "id": cid, "score": round(score, 4), "text": text[:200]})

print("\n--- 5 HIGHEST NOVELTY (most unique) ---")
highest_claims = []
for rank, idx in enumerate(sorted_indices[-5:][::-1]):
    score = all_scores[idx]
    text = claim_texts[idx]
    cid = claim_ids[idx]
    print(f"  {rank+1}. [{score:.4f}] {text[:120]}")
    highest_claims.append({"rank": rank+1, "id": cid, "score": round(score, 4), "text": text[:200]})

# ---------------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)

baseline_stats = results["baseline"]["stats"]
centroid_stats = results["centroid"]["stats"]
temporal_stats = results["temporal"]["stats"]
combined_stats = results["combined"]["stats"]

print(f"""
Corpus: {len(claim_ids)} education claims from claims database
Embedding: paraphrase-multilingual-MiniLM-L12-v2 (384d)
Index: VectorIndex with {index.size} items

1. BASELINE DISTRIBUTION (no centroid, no temporal):
   Mean={baseline_stats['mean']:.4f}, Std={baseline_stats['std']:.4f}
   Range [{baseline_stats['min']:.4f}, {baseline_stats['max']:.4f}]
   IQR [{baseline_stats['q25']:.4f}, {baseline_stats['q75']:.4f}]

2. CENTROID SPECIFICITY EFFECT:
   Mean shift: {centroid_stats['mean'] - baseline_stats['mean']:+.4f}
   Std change: {centroid_stats['std'] - baseline_stats['std']:+.4f}
   -> {'Spreads' if centroid_stats['std'] > baseline_stats['std'] else 'Compresses'} the distribution

3. TEMPORAL DECAY EFFECT (lambda=0.02, half-life ~35 days):
   Mean shift: {temporal_stats['mean'] - baseline_stats['mean']:+.4f}
   Std change: {temporal_stats['std'] - baseline_stats['std']:+.4f}
   -> {'Increases' if temporal_stats['mean'] > baseline_stats['mean'] else 'Decreases'} overall novelty

4. COMBINED (centroid + temporal):
   Mean shift: {combined_stats['mean'] - baseline_stats['mean']:+.4f}
   Std change: {combined_stats['std'] - baseline_stats['std']:+.4f}

5. PERFORMANCE at ~27K scale:
   Single novelty_score call: {baseline_stats['ms_per_call']:.2f}ms
   batch_novelty (100):  {batch_perf['100']['ms_per_claim']:.2f}ms/claim
   batch_novelty (1000): {batch_perf['1000']['ms_per_claim']:.2f}ms/claim
   batch_novelty (ALL):  {batch_perf['all']['ms_per_claim']:.2f}ms/claim ({batch_perf['all']['total_s']:.1f}s total)

6. FULL CORPUS NOVELTY (batch_novelty over all {len(claim_ids)} claims):
   Mean={all_scores_arr.mean():.4f}, Std={all_scores_arr.std():.4f}
   P5={all_q[0]:.4f}, Median={all_q[2]:.4f}, P95={all_q[4]:.4f}
""")

# ---------------------------------------------------------------------------
# 12. Save results
# ---------------------------------------------------------------------------
output = {
    "experiment": "exp15_large_corpus_novelty",
    "description": "Novelty scoring at real scale (~27K education claims from claims database)",
    "corpus_size": len(claim_ids),
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_dim": int(embeddings.shape[1]),
    "embedding_time_s": round(t_embed_total, 1),
    "index_build_time_s": round(t_index, 2),
    "sample_size": SAMPLE_N,
    "age_stats": {
        "min_days": round(min(age_values), 1),
        "max_days": round(max(age_values), 1),
        "mean_days": round(float(np.mean(age_values)), 1),
        "median_days": round(float(np.median(age_values)), 1),
    },
    "configs": {name: data["stats"] for name, data in results.items()},
    "batch_performance": batch_perf,
    "full_corpus_novelty": {
        "mean": round(float(all_scores_arr.mean()), 4),
        "std": round(float(all_scores_arr.std()), 4),
        "p5": round(float(all_q[0]), 4),
        "q25": round(float(all_q[1]), 4),
        "median": round(float(all_q[2]), 4),
        "q75": round(float(all_q[3]), 4),
        "p95": round(float(all_q[4]), 4),
    },
    "sanity_check": {
        "lowest_novelty": lowest_claims,
        "highest_novelty": highest_claims,
    },
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {OUTPUT_PATH}")
