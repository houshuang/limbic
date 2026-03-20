"""Eval: Amygdala novelty scoring on Quora Question Pairs (QQP).

Can novelty_score distinguish duplicate questions from non-duplicates?
  - duplicate pair -> low novelty (question2 is "known" given question1)
  - non-duplicate pair -> high novelty (question2 is "new")

Two test scenarios:
  1. Pairwise: 1-item index per pair (10K pairs, balanced 5K dup + 5K non-dup)
  2. Corpus: 1K-item index, score 1K queries against it (500 dup + 500 non-dup)

Models tested:
  - paraphrase-multilingual-MiniLM-L12-v2 (current default)
  - all-MiniLM-L6-v2 (old default)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex
from amygdala.novelty import novelty_score, corpus_centroid

OUTPUT_PATH = Path("experiments/results/eval_qqp_results.json")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auc_roc(scores: list[float], labels: list[int]) -> float:
    """Compute AUC-ROC. label=1 means 'novel' (non-duplicate)."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp, fp, prev_score = 0, 0, None
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    auc = 0.0
    prev_tp, prev_fp = 0, 0
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp, prev_fp = tp, fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (total_pos * total_neg)


def best_f1(scores: list[float], labels: list[int], n_thresholds: int = 200) -> dict:
    """Find optimal F1 threshold. Positive class = novel (label=1)."""
    lo, hi = min(scores), max(scores)
    thresholds = [lo + (hi - lo) * i / n_thresholds for i in range(n_thresholds + 1)]
    best = {"f1": 0, "threshold": 0, "precision": 0, "recall": 0}
    for t in thresholds:
        tp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
        fn = sum(1 for s, l in zip(scores, labels) if s < t and l == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best["f1"]:
            best = {"f1": round(f1, 4), "threshold": round(t, 4),
                    "precision": round(prec, 4), "recall": round(rec, 4)}
    return best


# ---------------------------------------------------------------------------
# Load QQP dataset — vectorized column access (Arrow-backed, fast)
# ---------------------------------------------------------------------------
print("Loading QQP dataset...")
t0 = time.time()
from datasets import load_dataset
ds = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")
print(f"  Loaded {len(ds)} pairs in {time.time() - t0:.1f}s")

# Vectorized column access — entire columns at once, no per-row overhead
print("  Extracting columns...")
t0 = time.time()
all_labels = np.array(ds["label"])
all_q1 = ds["sentence1"]  # list of strings
all_q2 = ds["sentence2"]
print(f"  Extracted in {time.time() - t0:.1f}s")

PAIRWISE_N = 5000  # per class
CORPUS_INDEX_SIZE = 1000
CORPUS_QUERY_SIZE = 500  # per class

np.random.seed(42)

dup_indices = np.where(all_labels == 1)[0]
nondup_indices = np.where(all_labels == 0)[0]
print(f"  Total duplicates: {len(dup_indices)}, non-duplicates: {len(nondup_indices)}")

# Sample balanced pairwise data
dup_sample = np.random.choice(dup_indices, size=PAIRWISE_N, replace=False)
nondup_sample = np.random.choice(nondup_indices, size=PAIRWISE_N, replace=False)
pairwise_indices = np.concatenate([dup_sample, nondup_sample])
np.random.shuffle(pairwise_indices)

pairwise_data = [
    {"q1": all_q1[int(i)], "q2": all_q2[int(i)], "is_duplicate": int(all_labels[i])}
    for i in pairwise_indices
]

print(f"  Pairwise sample: {len(pairwise_data)} pairs "
      f"({sum(p['is_duplicate'] for p in pairwise_data)} dup, "
      f"{sum(1 - p['is_duplicate'] for p in pairwise_data)} non-dup)")


# ---------------------------------------------------------------------------
# Build corpus test data
# ---------------------------------------------------------------------------
print("\nBuilding corpus test data...")

corpus_pool_size = 50000
corpus_pool = np.random.choice(len(ds), size=min(corpus_pool_size, len(ds)), replace=False)

index_questions = {}  # text -> id
index_q_set = set()
dup_queries = []
nondup_queries = []

for idx in corpus_pool:
    idx = int(idx)
    q1, q2, label = all_q1[idx], all_q2[idx], int(all_labels[idx])

    if len(index_questions) < CORPUS_INDEX_SIZE:
        if q1 not in index_q_set:
            index_questions[q1] = f"q{len(index_questions)}"
            index_q_set.add(q1)

    if len(index_questions) >= CORPUS_INDEX_SIZE:
        if label == 1 and q1 in index_q_set and q2 not in index_q_set:
            if len(dup_queries) < CORPUS_QUERY_SIZE:
                dup_queries.append(q2)
        elif label == 0 and q1 in index_q_set and q2 not in index_q_set:
            if len(nondup_queries) < CORPUS_QUERY_SIZE:
                nondup_queries.append(q2)

    if len(dup_queries) >= CORPUS_QUERY_SIZE and len(nondup_queries) >= CORPUS_QUERY_SIZE:
        break

# Fallback: scan full dataset if pool wasn't enough
if len(dup_queries) < CORPUS_QUERY_SIZE or len(nondup_queries) < CORPUS_QUERY_SIZE:
    print("  Scanning full dataset for more corpus queries...")
    pool_set = set(corpus_pool.tolist())
    for idx in range(len(ds)):
        if idx in pool_set:
            continue
        q1, q2, label = all_q1[idx], all_q2[idx], int(all_labels[idx])
        if label == 1 and q1 in index_q_set and q2 not in index_q_set:
            if len(dup_queries) < CORPUS_QUERY_SIZE:
                dup_queries.append(q2)
        elif label == 0 and q1 in index_q_set and q2 not in index_q_set:
            if len(nondup_queries) < CORPUS_QUERY_SIZE:
                nondup_queries.append(q2)
        if len(dup_queries) >= CORPUS_QUERY_SIZE and len(nondup_queries) >= CORPUS_QUERY_SIZE:
            break

print(f"  Corpus index: {len(index_questions)} questions")
print(f"  Corpus queries: {len(dup_queries)} dup + {len(nondup_queries)} non-dup")


# ---------------------------------------------------------------------------
# Models to test
# ---------------------------------------------------------------------------
MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", "multilingual (new default)"),
    ("all-MiniLM-L6-v2", "MiniLM-L6 (old default)"),
]

all_results = {
    "experiment": "eval_qqp_novelty",
    "dataset": "sentence-transformers/quora-duplicates (pair-class)",
    "pairwise_n": len(pairwise_data),
    "corpus_index_size": len(index_questions),
    "corpus_query_dup": len(dup_queries),
    "corpus_query_nondup": len(nondup_queries),
    "models": {},
}


for model_name, model_desc in MODELS:
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name} ({model_desc})")
    print(f"{'=' * 70}")

    model = EmbeddingModel(model_name=model_name)

    # ==================================================================
    # PAIRWISE TEST
    # ==================================================================
    print(f"\n--- Pairwise Test ({len(pairwise_data)} pairs) ---")

    q1_texts = [p["q1"] for p in pairwise_data]
    q2_texts = [p["q2"] for p in pairwise_data]

    print("  Embedding q1...")
    t0 = time.time()
    q1_embs = model.embed_batch(q1_texts)
    print(f"  Embedded {len(q1_texts)} in {time.time() - t0:.1f}s")

    print("  Embedding q2...")
    t0 = time.time()
    q2_embs = model.embed_batch(q2_texts)
    print(f"  Embedded {len(q2_texts)} in {time.time() - t0:.1f}s")

    # Pairwise novelty: for 1-item index, novelty_score = 1 - cosine_sim
    # Vectorize: just compute dot product directly (both are L2-normalized)
    print("  Computing pairwise novelty scores...")
    t0 = time.time()
    pairwise_cosines = np.sum(q1_embs * q2_embs, axis=1)
    pairwise_scores = (1.0 - pairwise_cosines).tolist()
    pairwise_labels = [1 - p["is_duplicate"] for p in pairwise_data]
    pw_time = time.time() - t0
    print(f"  Scored {len(pairwise_scores)} pairs in {pw_time:.1f}s")

    dup_novelty = [s for s, p in zip(pairwise_scores, pairwise_data) if p["is_duplicate"] == 1]
    nondup_novelty = [s for s, p in zip(pairwise_scores, pairwise_data) if p["is_duplicate"] == 0]

    pw_auc = auc_roc(pairwise_scores, pairwise_labels)
    pw_f1 = best_f1(pairwise_scores, pairwise_labels)

    print(f"\n  Pairwise Results:")
    print(f"    Mean novelty (duplicates):     {np.mean(dup_novelty):.4f} +/- {np.std(dup_novelty):.4f}")
    print(f"    Mean novelty (non-duplicates): {np.mean(nondup_novelty):.4f} +/- {np.std(nondup_novelty):.4f}")
    print(f"    Separation:                    {np.mean(nondup_novelty) - np.mean(dup_novelty):.4f}")
    print(f"    AUC-ROC:                       {pw_auc:.4f}")
    print(f"    Best F1:                       {pw_f1['f1']:.4f} (threshold={pw_f1['threshold']:.4f})")
    print(f"    Precision/Recall at best F1:   {pw_f1['precision']:.4f} / {pw_f1['recall']:.4f}")

    # ==================================================================
    # CORPUS TEST
    # ==================================================================
    print(f"\n--- Corpus Test ({len(index_questions)} index, "
          f"{len(dup_queries)}+{len(nondup_queries)} queries) ---")

    index_texts = list(index_questions.keys())
    index_ids = list(index_questions.values())

    print("  Embedding index...")
    t0 = time.time()
    index_embs = model.embed_batch(index_texts)
    print(f"  Embedded {len(index_texts)} in {time.time() - t0:.1f}s")

    vi_corpus = VectorIndex()
    vi_corpus.add(index_ids, index_embs)

    all_query_texts = dup_queries + nondup_queries
    corpus_labels = [0] * len(dup_queries) + [1] * len(nondup_queries)

    print("  Embedding queries...")
    t0 = time.time()
    query_embs = model.embed_batch(all_query_texts)
    print(f"  Embedded {len(all_query_texts)} in {time.time() - t0:.1f}s")

    # Score WITHOUT centroid specificity
    print("  Computing corpus novelty (no centroid specificity)...")
    t0 = time.time()
    corpus_scores_no_cs = []
    for i in range(len(all_query_texts)):
        score = novelty_score(query_embs[i], vi_corpus, use_centroid_specificity=False)
        corpus_scores_no_cs.append(score)
    print(f"  Scored in {time.time() - t0:.1f}s")

    # Score WITH centroid specificity
    print("  Computing corpus novelty (with centroid specificity)...")
    t0 = time.time()
    centroid = corpus_centroid(vi_corpus)
    corpus_scores_cs = []
    for i in range(len(all_query_texts)):
        score = novelty_score(query_embs[i], vi_corpus, centroid=centroid, use_centroid_specificity=True)
        corpus_scores_cs.append(score)
    print(f"  Scored in {time.time() - t0:.1f}s")

    # Print metrics for both
    for label_name, scores, cs_flag in [
        ("no_centroid_specificity", corpus_scores_no_cs, False),
        ("with_centroid_specificity", corpus_scores_cs, True),
    ]:
        dup_nov = [s for s, l in zip(scores, corpus_labels) if l == 0]
        nondup_nov = [s for s, l in zip(scores, corpus_labels) if l == 1]

        c_auc = auc_roc(scores, corpus_labels)
        c_f1 = best_f1(scores, corpus_labels)

        cs_desc = "WITH" if cs_flag else "WITHOUT"
        print(f"\n  Corpus Results ({cs_desc} centroid specificity):")
        print(f"    Mean novelty (duplicates):     {np.mean(dup_nov):.4f} +/- {np.std(dup_nov):.4f}")
        print(f"    Mean novelty (non-duplicates): {np.mean(nondup_nov):.4f} +/- {np.std(nondup_nov):.4f}")
        print(f"    Separation:                    {np.mean(nondup_nov) - np.mean(dup_nov):.4f}")
        print(f"    AUC-ROC:                       {c_auc:.4f}")
        print(f"    Best F1:                       {c_f1['f1']:.4f} (threshold={c_f1['threshold']:.4f})")
        print(f"    Precision/Recall at best F1:   {c_f1['precision']:.4f} / {c_f1['recall']:.4f}")

    # Store results
    all_results["models"][model_name] = {
        "description": model_desc,
        "pairwise": {
            "n_pairs": len(pairwise_data),
            "mean_novelty_duplicates": round(float(np.mean(dup_novelty)), 4),
            "std_novelty_duplicates": round(float(np.std(dup_novelty)), 4),
            "mean_novelty_nondups": round(float(np.mean(nondup_novelty)), 4),
            "std_novelty_nondups": round(float(np.std(nondup_novelty)), 4),
            "separation": round(float(np.mean(nondup_novelty) - np.mean(dup_novelty)), 4),
            "auc_roc": round(pw_auc, 4),
            "best_f1": pw_f1,
        },
        "corpus_no_centroid": {
            "index_size": len(index_questions),
            "n_queries": len(all_query_texts),
            "mean_novelty_duplicates": round(float(np.mean([s for s, l in zip(corpus_scores_no_cs, corpus_labels) if l == 0])), 4),
            "mean_novelty_nondups": round(float(np.mean([s for s, l in zip(corpus_scores_no_cs, corpus_labels) if l == 1])), 4),
            "separation": round(float(np.mean([s for s, l in zip(corpus_scores_no_cs, corpus_labels) if l == 1]) - np.mean([s for s, l in zip(corpus_scores_no_cs, corpus_labels) if l == 0])), 4),
            "auc_roc": round(auc_roc(corpus_scores_no_cs, corpus_labels), 4),
            "best_f1": best_f1(corpus_scores_no_cs, corpus_labels),
        },
        "corpus_with_centroid": {
            "index_size": len(index_questions),
            "n_queries": len(all_query_texts),
            "mean_novelty_duplicates": round(float(np.mean([s for s, l in zip(corpus_scores_cs, corpus_labels) if l == 0])), 4),
            "mean_novelty_nondups": round(float(np.mean([s for s, l in zip(corpus_scores_cs, corpus_labels) if l == 1])), 4),
            "separation": round(float(np.mean([s for s, l in zip(corpus_scores_cs, corpus_labels) if l == 1]) - np.mean([s for s, l in zip(corpus_scores_cs, corpus_labels) if l == 0])), 4),
            "auc_roc": round(auc_roc(corpus_scores_cs, corpus_labels), 4),
            "best_f1": best_f1(corpus_scores_cs, corpus_labels),
        },
    }


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")

print(f"\n{'Model':<42} {'Test':<28} {'AUC':>6} {'F1':>6} {'Sep':>7}")
print("-" * 91)

for mname in all_results["models"]:
    m = all_results["models"][mname]
    short = mname.split("/")[-1][:38]
    for test, key in [("Pairwise (1-item index)", "pairwise"),
                      ("Corpus (no centroid spec)", "corpus_no_centroid"),
                      ("Corpus (with centroid spec)", "corpus_with_centroid")]:
        d = m[key]
        print(f"{short:<42} {test:<28} {d['auc_roc']:>6.4f} {d['best_f1']['f1']:>6.4f} {d['separation']:>7.4f}")
    print()

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to {OUTPUT_PATH}")
