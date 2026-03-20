import os
"""Experiment 12: Soft-ZCA Whitening vs PCA Whitening vs All-but-the-Top.

Compares three embedding post-processing methods:
  a) PCA whitening (current): W = U[:,:k] @ diag(1/sqrt(S[:k] + eps)), truncates to k dims
  b) Soft-ZCA (ESANN 2025): W = U @ diag(1/sqrt(S + eps)) @ U^T, full-rank regularized
  c) All-but-the-top (Mu & Viswanath, ICLR 2018): subtract projection onto top D PCs

Evaluated on two datasets:
  1) Calibration claims (120 diverse claims with ground truth labels)
  2) Otak education claims (~2K domain-homogeneous, nearest-neighbor vs random pairs)

Sweeps:
  - PCA whitening: dims in [128, 256, 384]
  - Soft-ZCA: epsilon in [0.001, 0.01, 0.05, 0.1, 0.5]
  - All-but-top: D (components removed) in [1, 2, 3, 5]
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
RESULTS_PATH = Path("experiments/results/exp12_results.json")

# Hyperparameter grids
PCA_DIMS = [128, 256, 384]
ZCA_EPSILONS = [0.001, 0.01, 0.05, 0.1, 0.5]
ABT_COMPONENTS = [1, 2, 3, 5]

# Corpus sizes
WHITENING_CORPUS_SIZE = 5000
OTAK_EVAL_SIZE = 2000


def load_whitening_corpus():
    """Load random claims from claims database for fitting whitening transforms."""
    conn = sqlite3.connect(str(OTAK_DB))
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT n.name FROM idx_knowledge_item_claim_type k "
        "JOIN nodes n ON n.id = k.node_id "
        "WHERE n.deleted_at IS NULL AND n.name IS NOT NULL AND length(n.name) > 20 AND length(n.name) < 500 "
        "ORDER BY RANDOM() LIMIT ?",
        (WHITENING_CORPUS_SIZE,),
    )
    texts = [row[0] for row in cur.fetchall()]
    conn.close()
    print(f"Loaded {len(texts)} whitening corpus texts from claims database")
    return texts


def load_domain_eval_claims():
    """Load a separate set of claims for domain-homogeneous evaluation."""
    conn = sqlite3.connect(str(OTAK_DB))
    cur = conn.cursor()
    # Get claims NOT in the whitening corpus (different random sample)
    cur.execute(
        "SELECT DISTINCT n.name FROM idx_knowledge_item_claim_type k "
        "JOIN nodes n ON n.id = k.node_id "
        "WHERE n.deleted_at IS NULL AND n.name IS NOT NULL AND length(n.name) > 20 AND length(n.name) < 500 "
        "ORDER BY RANDOM() LIMIT ?",
        (OTAK_EVAL_SIZE,),
    )
    texts = [row[0] for row in cur.fetchall()]
    conn.close()
    print(f"Loaded {len(texts)} domain eval claims")
    return texts


def load_calibration_data():
    """Load calibration claims with ground truth labels."""
    with open(CALIBRATION_PATH) as f:
        claims = json.load(f)
    print(f"Loaded {len(claims)} calibration claims")
    return claims


# --- Whitening Methods ---

def apply_pca_whitening(embeddings, mean, U, S, k):
    """PCA whitening (current amygdala method): truncate to k dims."""
    W = U[:, :k] @ np.diag(1.0 / np.sqrt(S[:k] + 1e-8))
    centered = embeddings - mean
    whitened = centered @ W
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    return whitened / np.maximum(norms, 1e-8)


def apply_soft_zca(embeddings, mean, U, S, epsilon):
    """Soft-ZCA (ESANN 2025): full-rank regularized whitening.

    W = U @ diag(1/sqrt(S + epsilon)) @ U^T
    Preserves all dimensions, rotates back to original basis.
    """
    D_inv_sqrt = np.diag(1.0 / np.sqrt(S + epsilon))
    W = U @ D_inv_sqrt @ U.T
    centered = embeddings - mean
    whitened = centered @ W
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    return whitened / np.maximum(norms, 1e-8)


def apply_all_but_top(embeddings, mean, U, d):
    """All-but-the-top (Mu & Viswanath, ICLR 2018): remove top D PCs.

    1. Center embeddings
    2. Subtract projection onto top D principal components
    3. Re-normalize
    """
    centered = embeddings - mean
    # Project onto top-d components and subtract
    top_components = U[:, :d]  # shape: (dim, d)
    projection = centered @ top_components @ top_components.T
    cleaned = centered - projection
    norms = np.linalg.norm(cleaned, axis=1, keepdims=True)
    return cleaned / np.maximum(norms, 1e-8)


# --- Evaluation Metrics ---

def compute_mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean cosine similarity across all pairs (already L2-normalized)."""
    sim_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim_matrix[mask].mean())


def compute_calibration_metrics(cal_claims, cal_embeddings, source_embeddings):
    """Compute discrimination gap and classification accuracy on calibration set.

    Returns metrics dict with gap, accuracy, per-type cosines.
    """
    related_cosines = []
    unrelated_cosines = []
    contradiction_cosines = []

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

    mean_related = float(np.mean(related_cosines)) if related_cosines else 0
    mean_unrelated = float(np.mean(unrelated_cosines)) if unrelated_cosines else 0
    mean_contradiction = float(np.mean(contradiction_cosines)) if contradiction_cosines else 0
    gap = mean_related - mean_unrelated

    # Classification accuracy at optimal threshold
    best_acc = 0.0
    best_threshold = 0.0
    thresholds = sorted(set(c for c, _ in all_cosines))
    for t in thresholds:
        correct = sum(1 for cosine, is_related in all_cosines if (cosine >= t) == is_related)
        acc = correct / len(all_cosines)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return {
        "mean_related_cosine": mean_related,
        "mean_unrelated_cosine": mean_unrelated,
        "mean_contradiction_cosine": mean_contradiction,
        "discrimination_gap": gap,
        "classification_accuracy": float(best_acc),
        "optimal_threshold": float(best_threshold),
    }


def compute_domain_homogeneous_metrics(embeddings: np.ndarray, n_pairs=500):
    """Evaluate on domain-homogeneous data using nearest-neighbor vs random separation.

    For domain-homogeneous corpora (all education claims), a good embedding space
    should still distinguish between genuinely similar vs random pairs.

    Returns:
    - mean_nn_cosine: average cosine of nearest neighbor pairs
    - mean_random_cosine: average cosine of random pairs
    - nn_random_gap: separation between NN and random pairs
    - mean_pairwise_cosine: overall isotropy measure
    """
    n = len(embeddings)
    sim_matrix = embeddings @ embeddings.T

    # Nearest neighbors (excluding self)
    np.fill_diagonal(sim_matrix, -1)
    nn_indices = np.argmax(sim_matrix, axis=1)
    nn_cosines = sim_matrix[np.arange(n), nn_indices]
    mean_nn = float(nn_cosines.mean())

    # Random pairs
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    # Ensure no self-pairs
    mask = idx_a == idx_b
    idx_b[mask] = (idx_b[mask] + 1) % n
    np.fill_diagonal(sim_matrix, 1)  # restore diagonal
    random_cosines = np.array([float(embeddings[a] @ embeddings[b]) for a, b in zip(idx_a, idx_b)])
    mean_random = float(random_cosines.mean())

    # Mean pairwise cosine (isotropy)
    mpc = compute_mean_pairwise_cosine(embeddings)

    return {
        "mean_nn_cosine": mean_nn,
        "mean_random_cosine": mean_random,
        "nn_random_gap": mean_nn - mean_random,
        "mean_pairwise_cosine": mpc,
    }


def format_table_row(method, cal_metrics, dom_metrics):
    """Format a result as a table row."""
    return (
        f"{method:<25} | "
        f"{cal_metrics['discrimination_gap']:>6.4f} | "
        f"{cal_metrics['classification_accuracy']:>5.3f} | "
        f"{cal_metrics['mean_related_cosine']:>6.4f} | "
        f"{cal_metrics['mean_unrelated_cosine']:>6.4f} | "
        f"{cal_metrics['mean_contradiction_cosine']:>6.4f} | "
        f"{dom_metrics['nn_random_gap']:>6.4f} | "
        f"{dom_metrics['mean_pairwise_cosine']:>6.4f}"
    )


def main():
    t0 = time.time()

    # Load data
    corpus_texts = load_whitening_corpus()
    cal_claims = load_calibration_data()
    domain_eval_texts = load_domain_eval_claims()

    # Collect unique source claims
    source_map = {}
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
    print(f"Domain eval claims: {len(domain_eval_texts)}")

    # Initialize model (raw, no whitening)
    model = EmbeddingModel()

    # Embed everything raw
    print("\nEmbedding whitening corpus...")
    raw_corpus = model._raw_embed_batch(corpus_texts)
    print(f"  Corpus shape: {raw_corpus.shape}")

    print("Embedding calibration claims...")
    raw_cal = model._raw_embed_batch(cal_texts)

    print("Embedding source claims...")
    raw_sources = model._raw_embed_batch(source_texts)

    print("Embedding domain eval claims...")
    raw_domain = model._raw_embed_batch(domain_eval_texts)
    print(f"  Domain eval shape: {raw_domain.shape}")

    # Compute PCA on whitening corpus (once)
    print("\nComputing PCA on whitening corpus...")
    mean = raw_corpus.mean(axis=0)
    centered_corpus = raw_corpus - mean
    cov = centered_corpus.T @ centered_corpus / len(corpus_texts)
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    print(f"  SVD complete. Top 10 eigenvalues: {S[:10].round(4)}")
    print(f"  Eigenvalue range: {S[0]:.4f} to {S[-1]:.6f} (ratio: {S[0]/S[-1]:.0f}x)")

    # Print eigenvalue spectrum analysis
    total_var = S.sum()
    for k in [10, 50, 100, 200, 300]:
        pct = S[:k].sum() / total_var * 100
        print(f"  Top {k:>3} components explain {pct:.1f}% variance")

    # Table header
    header = (
        f"{'Method':<25} | "
        f"{'Gap':>6} | {'Acc':>5} | "
        f"{'Rel':>6} | {'Unr':>6} | {'Con':>6} | "
        f"{'NNGap':>6} | {'MPC':>6}"
    )
    separator = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("CALIBRATION CLAIMS (diverse, 120 claims) + OTAK CLAIMS (domain-homogeneous)")
    print(f"{'='*len(header)}")
    print(header)
    print(separator)

    all_results = []

    # --- Baseline: raw embeddings ---
    source_emb_map = {sid: raw_sources[i] for i, sid in enumerate(source_ids)}
    cal_metrics = compute_calibration_metrics(cal_claims, raw_cal, source_emb_map)
    dom_metrics = compute_domain_homogeneous_metrics(raw_domain)
    print(format_table_row("raw (baseline)", cal_metrics, dom_metrics))
    all_results.append({
        "method": "raw",
        "params": {},
        "calibration": cal_metrics,
        "domain_homogeneous": dom_metrics,
    })

    # --- PCA Whitening sweep ---
    for k in PCA_DIMS:
        w_cal = apply_pca_whitening(raw_cal, mean, U, S, k)
        w_sources = apply_pca_whitening(raw_sources, mean, U, S, k)
        w_domain = apply_pca_whitening(raw_domain, mean, U, S, k)

        src_map = {sid: w_sources[i] for i, sid in enumerate(source_ids)}
        c_met = compute_calibration_metrics(cal_claims, w_cal, src_map)
        d_met = compute_domain_homogeneous_metrics(w_domain)
        label = f"PCA k={k}"
        print(format_table_row(label, c_met, d_met))
        all_results.append({
            "method": "pca",
            "params": {"dims": k},
            "calibration": c_met,
            "domain_homogeneous": d_met,
        })

    # --- Soft-ZCA sweep ---
    for eps in ZCA_EPSILONS:
        w_cal = apply_soft_zca(raw_cal, mean, U, S, eps)
        w_sources = apply_soft_zca(raw_sources, mean, U, S, eps)
        w_domain = apply_soft_zca(raw_domain, mean, U, S, eps)

        src_map = {sid: w_sources[i] for i, sid in enumerate(source_ids)}
        c_met = compute_calibration_metrics(cal_claims, w_cal, src_map)
        d_met = compute_domain_homogeneous_metrics(w_domain)
        label = f"Soft-ZCA eps={eps}"
        print(format_table_row(label, c_met, d_met))
        all_results.append({
            "method": "soft_zca",
            "params": {"epsilon": eps},
            "calibration": c_met,
            "domain_homogeneous": d_met,
        })

    # --- All-but-the-top sweep ---
    for d in ABT_COMPONENTS:
        w_cal = apply_all_but_top(raw_cal, mean, U, d)
        w_sources = apply_all_but_top(raw_sources, mean, U, d)
        w_domain = apply_all_but_top(raw_domain, mean, U, d)

        src_map = {sid: w_sources[i] for i, sid in enumerate(source_ids)}
        c_met = compute_calibration_metrics(cal_claims, w_cal, src_map)
        d_met = compute_domain_homogeneous_metrics(w_domain)
        label = f"All-but-top D={d}"
        print(format_table_row(label, c_met, d_met))
        all_results.append({
            "method": "all_but_top",
            "params": {"D": d},
            "calibration": c_met,
            "domain_homogeneous": d_met,
        })

    print(separator)

    # --- Analysis: find best methods ---
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    # Best for diverse data (calibration)
    best_cal_gap = max(all_results, key=lambda r: r["calibration"]["discrimination_gap"])
    best_cal_acc = max(all_results, key=lambda r: r["calibration"]["classification_accuracy"])
    print(f"\nBest calibration discrimination gap: {best_cal_gap['method']} "
          f"({best_cal_gap['params']}) = {best_cal_gap['calibration']['discrimination_gap']:.4f}")
    print(f"Best calibration accuracy: {best_cal_acc['method']} "
          f"({best_cal_acc['params']}) = {best_cal_acc['calibration']['classification_accuracy']:.3f}")

    # Best for domain-homogeneous data
    best_dom_gap = max(all_results, key=lambda r: r["domain_homogeneous"]["nn_random_gap"])
    best_dom_mpc = min(all_results, key=lambda r: r["domain_homogeneous"]["mean_pairwise_cosine"])
    print(f"\nBest domain-homogeneous NN-random gap: {best_dom_gap['method']} "
          f"({best_dom_gap['params']}) = {best_dom_gap['domain_homogeneous']['nn_random_gap']:.4f}")
    print(f"Best domain-homogeneous isotropy (lowest MPC): {best_dom_mpc['method']} "
          f"({best_dom_mpc['params']}) = {best_dom_mpc['domain_homogeneous']['mean_pairwise_cosine']:.4f}")

    # Compare methods head-to-head
    print(f"\n--- Method Comparison Summary ---")
    for method_name in ["raw", "pca", "soft_zca", "all_but_top"]:
        method_results = [r for r in all_results if r["method"] == method_name]
        if not method_results:
            continue
        best = max(method_results, key=lambda r: r["calibration"]["discrimination_gap"])
        print(f"\n{method_name.upper()}:")
        print(f"  Best config: {best['params']}")
        print(f"  Cal gap: {best['calibration']['discrimination_gap']:.4f}  "
              f"Cal acc: {best['calibration']['classification_accuracy']:.3f}")
        print(f"  Dom NN-gap: {best['domain_homogeneous']['nn_random_gap']:.4f}  "
              f"Dom MPC: {best['domain_homogeneous']['mean_pairwise_cosine']:.4f}")

    # Verdict
    raw_result = [r for r in all_results if r["method"] == "raw"][0]
    raw_cal_gap = raw_result["calibration"]["discrimination_gap"]
    raw_dom_gap = raw_result["domain_homogeneous"]["nn_random_gap"]

    print(f"\n--- Verdict ---")
    print(f"Raw baseline: cal_gap={raw_cal_gap:.4f}, dom_nn_gap={raw_dom_gap:.4f}")
    for r in all_results:
        if r["method"] == "raw":
            continue
        cal_delta = r["calibration"]["discrimination_gap"] - raw_cal_gap
        dom_delta = r["domain_homogeneous"]["nn_random_gap"] - raw_dom_gap
        label = f"{r['method']}({r['params']})"
        print(f"  {label:<35} cal_gap: {'+' if cal_delta >= 0 else ''}{cal_delta:.4f}  "
              f"dom_gap: {'+' if dom_delta >= 0 else ''}{dom_delta:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    output = {
        "experiment": "exp12_soft_zca",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "corpus_size": len(corpus_texts),
        "calibration_size": len(cal_claims),
        "domain_eval_size": len(domain_eval_texts),
        "elapsed_seconds": elapsed,
        "eigenvalue_spectrum": {
            "top_10": S[:10].tolist(),
            "range_ratio": float(S[0] / S[-1]),
            "explained_variance": {
                str(k): float(S[:k].sum() / total_var)
                for k in [10, 50, 100, 200, 300, 384]
            },
        },
        "results": all_results,
        "best": {
            "calibration_gap": {
                "method": best_cal_gap["method"],
                "params": best_cal_gap["params"],
                "value": best_cal_gap["calibration"]["discrimination_gap"],
            },
            "calibration_accuracy": {
                "method": best_cal_acc["method"],
                "params": best_cal_acc["params"],
                "value": best_cal_acc["calibration"]["classification_accuracy"],
            },
            "domain_nn_gap": {
                "method": best_dom_gap["method"],
                "params": best_dom_gap["params"],
                "value": best_dom_gap["domain_homogeneous"]["nn_random_gap"],
            },
            "domain_isotropy": {
                "method": best_dom_mpc["method"],
                "params": best_dom_mpc["params"],
                "value": best_dom_mpc["domain_homogeneous"]["mean_pairwise_cosine"],
            },
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
