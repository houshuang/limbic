import os
"""Experiment 2: Whitening Dimension Sweep.

Measures the effect of PCA whitening dimension on embedding quality using:
- Mean pairwise cosine (lower = more spread, better discrimination)
- Discrimination gap (cosine(paraphrase,source) - cosine(unrelated,source))
- Classification accuracy (paraphrase+extension=RELATED vs contradiction+unrelated=UNRELATED)

Data sources:
- Whitening corpus: 5K random claims from claims database (nodes table)
- Calibration set: 120 claims from exp_l_calibration/calibration_claims.json
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel

OTAK_DB = Path(os.environ.get("AMYGDALA_EVAL_DB", "eval_claims.db"))
CALIBRATION_PATH = Path(os.environ.get("AMYGDALA_CALIBRATION_DATA", "experiments/eval_data/calibration_claims.json"))
RESULTS_PATH = Path("experiments/results/exp2_results.json")
WHITEN_DIMS = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]
CORPUS_SIZE = 5000


def load_whitening_corpus():
    """Load 5K random claims from claims database."""
    conn = sqlite3.connect(str(OTAK_DB))
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM nodes WHERE length(name) > 20 AND length(name) < 500 "
        "AND deleted_at IS NULL ORDER BY RANDOM() LIMIT ?",
        (CORPUS_SIZE,),
    )
    texts = [row[0] for row in cur.fetchall()]
    conn.close()
    print(f"Loaded {len(texts)} whitening corpus texts from claims database")
    return texts


def load_calibration_data():
    """Load calibration claims and group by variant type."""
    with open(CALIBRATION_PATH) as f:
        claims = json.load(f)
    print(f"Loaded {len(claims)} calibration claims")
    return claims


def compute_mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean cosine similarity across all pairs (already L2-normalized)."""
    sim_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim_matrix[mask].mean())


def compute_discrimination_metrics(
    cal_claims: list[dict],
    cal_embeddings: np.ndarray,
    source_embeddings: dict[int, np.ndarray],
):
    """Compute discrimination gap and classification accuracy.

    Discrimination gap: mean cos(paraphrase, source) - mean cos(unrelated, source)
    Classification: paraphrase+extension = RELATED, contradiction+unrelated = UNRELATED
    """
    related_cosines = []  # paraphrase + extension
    unrelated_cosines = []  # unrelated
    contradiction_cosines = []  # contradiction

    all_cosines = []  # (cosine, is_related) for classification

    for i, claim in enumerate(cal_claims):
        vtype = claim["variant_type"]
        source_id = claim["source_id"]
        if source_id not in source_embeddings:
            continue

        emb = cal_embeddings[i]
        src_emb = source_embeddings[source_id]
        cosine = float(emb @ src_emb)

        if vtype == "paraphrase":
            related_cosines.append(cosine)
            all_cosines.append((cosine, True))
        elif vtype == "extension":
            related_cosines.append(cosine)
            all_cosines.append((cosine, True))
        elif vtype == "contradiction":
            contradiction_cosines.append(cosine)
            all_cosines.append((cosine, False))
        elif vtype == "unrelated":
            unrelated_cosines.append(cosine)
            all_cosines.append((cosine, False))

    mean_related = np.mean(related_cosines) if related_cosines else 0
    mean_unrelated = np.mean(unrelated_cosines) if unrelated_cosines else 0
    mean_contradiction = np.mean(contradiction_cosines) if contradiction_cosines else 0
    gap = float(mean_related - mean_unrelated)

    # Classification accuracy at optimal threshold
    best_acc = 0.0
    best_threshold = 0.0
    thresholds = sorted(set(c for c, _ in all_cosines))
    for t in thresholds:
        correct = sum(
            1
            for cosine, is_related in all_cosines
            if (cosine >= t) == is_related
        )
        acc = correct / len(all_cosines)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return {
        "mean_related_cosine": float(mean_related),
        "mean_unrelated_cosine": float(mean_unrelated),
        "mean_contradiction_cosine": float(mean_contradiction),
        "discrimination_gap": gap,
        "classification_accuracy": float(best_acc),
        "optimal_threshold": float(best_threshold),
        "n_related": len(related_cosines),
        "n_unrelated": len(unrelated_cosines),
        "n_contradiction": len(contradiction_cosines),
    }


def apply_whitening(raw_embeddings, mean, U, S, k):
    """Apply PCA whitening with k dimensions."""
    W = U[:, :k] @ np.diag(1.0 / np.sqrt(S[:k] + 1e-8))
    centered = raw_embeddings - mean
    whitened = centered @ W
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    return whitened / np.maximum(norms, 1e-8)


def main():
    t0 = time.time()

    # Load data
    corpus_texts = load_whitening_corpus()
    cal_claims = load_calibration_data()

    # Collect unique source claims and their IDs
    source_map = {}  # source_id -> source_claim_text
    for claim in cal_claims:
        sid = claim["source_id"]
        if sid not in source_map:
            source_map[sid] = claim["source_claim"]

    cal_texts = [c["text"] for c in cal_claims]
    source_ids = list(source_map.keys())
    source_texts = [source_map[sid] for sid in source_ids]

    print(f"\nUnique source claims: {len(source_texts)}")
    print(f"Calibration claims: {len(cal_texts)}")
    print(f"Whitening corpus: {len(corpus_texts)}")

    # Initialize model (no whitening yet — we'll do raw embedding first)
    model = EmbeddingModel(whiten_dims=384)

    # Embed everything raw (no whitening)
    print("\nEmbedding whitening corpus...")
    raw_corpus = model._raw_embed_batch(corpus_texts)
    print(f"  Corpus shape: {raw_corpus.shape}")

    print("Embedding calibration claims...")
    raw_cal = model._raw_embed_batch(cal_texts)
    print(f"  Calibration shape: {raw_cal.shape}")

    print("Embedding source claims...")
    raw_sources = model._raw_embed_batch(source_texts)
    print(f"  Sources shape: {raw_sources.shape}")

    # Compute PCA on corpus (once)
    print("\nComputing PCA on whitening corpus...")
    mean = raw_corpus.mean(axis=0)
    centered_corpus = raw_corpus - mean
    cov = centered_corpus.T @ centered_corpus / len(corpus_texts)
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    print(f"  SVD complete. Top singular values: {S[:5]}")

    # Baseline: raw embeddings (no whitening)
    print("\n--- Computing baseline (raw, no whitening) ---")
    raw_mean_cos = compute_mean_pairwise_cosine(raw_cal)
    source_emb_map_raw = {sid: raw_sources[i] for i, sid in enumerate(source_ids)}
    raw_metrics = compute_discrimination_metrics(cal_claims, raw_cal, source_emb_map_raw)
    print(f"  Mean pairwise cosine: {raw_mean_cos:.4f}")
    print(f"  Discrimination gap:   {raw_metrics['discrimination_gap']:.4f}")
    print(f"  Classification acc:   {raw_metrics['classification_accuracy']:.4f}")

    # Sweep whitening dimensions
    results = []
    results.append(
        {
            "whiten_dims": 0,
            "label": "raw (no whitening)",
            "mean_pairwise_cosine": raw_mean_cos,
            **raw_metrics,
        }
    )

    print(f"\n{'='*90}")
    print(
        f"{'dims':>6} | {'mean_cos':>10} | {'gap':>8} | {'acc':>6} | "
        f"{'related':>8} | {'unrelated':>10} | {'contradict':>11} | {'threshold':>10}"
    )
    print(f"{'-'*90}")
    print(
        f"{'raw':>6} | {raw_mean_cos:>10.4f} | {raw_metrics['discrimination_gap']:>8.4f} | "
        f"{raw_metrics['classification_accuracy']:>6.3f} | "
        f"{raw_metrics['mean_related_cosine']:>8.4f} | "
        f"{raw_metrics['mean_unrelated_cosine']:>10.4f} | "
        f"{raw_metrics['mean_contradiction_cosine']:>11.4f} | "
        f"{raw_metrics['optimal_threshold']:>10.4f}"
    )

    for k in WHITEN_DIMS:
        # Apply whitening to calibration and source embeddings
        w_cal = apply_whitening(raw_cal, mean, U, S, k)
        w_sources = apply_whitening(raw_sources, mean, U, S, k)

        # Mean pairwise cosine
        mpc = compute_mean_pairwise_cosine(w_cal)

        # Source embedding map
        source_emb_map = {sid: w_sources[i] for i, sid in enumerate(source_ids)}

        # Discrimination metrics
        metrics = compute_discrimination_metrics(cal_claims, w_cal, source_emb_map)

        row = {
            "whiten_dims": k,
            "label": f"whitened-{k}",
            "mean_pairwise_cosine": mpc,
            **metrics,
        }
        results.append(row)

        print(
            f"{k:>6} | {mpc:>10.4f} | {metrics['discrimination_gap']:>8.4f} | "
            f"{metrics['classification_accuracy']:>6.3f} | "
            f"{metrics['mean_related_cosine']:>8.4f} | "
            f"{metrics['mean_unrelated_cosine']:>10.4f} | "
            f"{metrics['mean_contradiction_cosine']:>11.4f} | "
            f"{metrics['optimal_threshold']:>10.4f}"
        )

    print(f"{'='*90}")

    # Summary: find best configuration
    # Best by discrimination gap
    best_gap = max(results, key=lambda r: r["discrimination_gap"])
    best_acc = max(results, key=lambda r: r["classification_accuracy"])
    best_spread = min(results[1:], key=lambda r: r["mean_pairwise_cosine"])  # skip raw

    print(f"\n--- Summary ---")
    print(f"Best discrimination gap: dims={best_gap['whiten_dims']} ({best_gap['discrimination_gap']:.4f})")
    print(f"Best classification acc: dims={best_acc['whiten_dims']} ({best_acc['classification_accuracy']:.3f})")
    print(f"Best spread (lowest MPC): dims={best_spread['whiten_dims']} ({best_spread['mean_pairwise_cosine']:.4f})")

    # Also compute explained variance ratio
    total_var = S.sum()
    print(f"\n--- Explained Variance by Dimension ---")
    for k in WHITEN_DIMS:
        explained = S[:k].sum() / total_var
        print(f"  dims={k:>3}: {explained:.4f} ({explained*100:.1f}%)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    output = {
        "experiment": "exp2_whitening_sweep",
        "model": "all-MiniLM-L6-v2",
        "corpus_size": len(corpus_texts),
        "calibration_size": len(cal_claims),
        "n_source_claims": len(source_texts),
        "elapsed_seconds": elapsed,
        "explained_variance": {
            str(k): float(S[:k].sum() / total_var) for k in WHITEN_DIMS
        },
        "results": results,
        "best": {
            "by_discrimination_gap": best_gap["whiten_dims"],
            "by_classification_accuracy": best_acc["whiten_dims"],
            "by_spread": best_spread["whiten_dims"],
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
