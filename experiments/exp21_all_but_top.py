import os
"""Experiment 21: All-but-the-Top — Dedicated Deep Analysis.

Evaluates "All-but-the-top" (Mu & Viswanath, ICLR 2018) as a simpler alternative
to Soft-ZCA whitening. The method removes top D principal components from embeddings,
which eliminates the dominant direction(s) that inflate cosine similarities.

Key question: does All-but-the-top match Soft-ZCA quality (+32% NN-gap)? If so, at what D?

Comparisons:
  - Raw embeddings (baseline)
  - All-but-the-top D=1, 2, 3, 5
  - Soft-ZCA eps=0.1 (current recommended)
  - PCA whitening k=128 (legacy)

Datasets:
  1) 120 calibration claims (diverse, with ground truth labels)
  2) ~5K domain claims (domain-homogeneous, for fitting + NN-gap eval)
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
RESULTS_PATH = Path("experiments/results/exp21_results.json")

WHITENING_CORPUS_SIZE = 5000
OTAK_EVAL_SIZE = 2000


# --- Data Loading ---

def load_whitening_corpus():
    """Load random claims from claims database for fitting whitening transforms."""
    conn = sqlite3.connect(str(OTAK_DB))
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT n.name FROM idx_knowledge_item_claim_type k "
        "JOIN nodes n ON n.id = k.node_id "
        "WHERE n.deleted_at IS NULL AND n.name IS NOT NULL "
        "AND length(n.name) > 20 AND length(n.name) < 500 "
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
    cur.execute(
        "SELECT DISTINCT n.name FROM idx_knowledge_item_claim_type k "
        "JOIN nodes n ON n.id = k.node_id "
        "WHERE n.deleted_at IS NULL AND n.name IS NOT NULL "
        "AND length(n.name) > 20 AND length(n.name) < 500 "
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


# --- All-but-the-Top Implementation ---

def all_but_the_top(embeddings, D=1):
    """Remove top D principal components from embeddings.

    Mu & Viswanath, ICLR 2018: "All-but-the-Top: Simple and Effective
    Postprocessing for Word Representations."

    Steps:
      1. Subtract mean
      2. Compute SVD of centered embeddings
      3. Remove projection onto top D right-singular vectors
      4. Re-normalize to unit length

    Returns (result, mean, top_components) so the transform can be applied
    to new embeddings.
    """
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    top_components = Vt[:D]  # (D, dim)
    result = centered - centered @ top_components.T @ top_components
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(norms, 1e-8)
    return result, mean, top_components


def apply_all_but_top(embedding, mean, top_components):
    """Apply fitted all-but-the-top to new embedding(s).

    embedding: (dim,) or (N, dim) array
    """
    centered = embedding - mean
    result = centered - centered @ top_components.T @ top_components
    if result.ndim == 1:
        norm = np.linalg.norm(result)
        return result / max(norm, 1e-8)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(norms, 1e-8)


# --- Comparison Whitening Methods ---

def apply_pca_whitening(embeddings, mean, U, S, k):
    """PCA whitening: truncate to k dims."""
    W = U[:, :k] @ np.diag(1.0 / np.sqrt(S[:k] + 1e-8))
    centered = embeddings - mean
    whitened = centered @ W
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    return whitened / np.maximum(norms, 1e-8)


def apply_soft_zca(embeddings, mean, U, S, epsilon):
    """Soft-ZCA: full-rank regularized whitening. W = U diag(1/sqrt(S+eps)) U^T."""
    D_inv_sqrt = np.diag(1.0 / np.sqrt(S + epsilon))
    W = U @ D_inv_sqrt @ U.T
    centered = embeddings - mean
    whitened = centered @ W
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    return whitened / np.maximum(norms, 1e-8)


# --- Evaluation Metrics ---

def compute_calibration_metrics(cal_claims, cal_embeddings, source_embeddings):
    """Compute discrimination gap and classification accuracy on calibration set."""
    related_cosines = []
    unrelated_cosines = []
    contradiction_cosines = []
    all_cosines = []

    for i, claim in enumerate(cal_claims):
        vtype = claim["variant_type"]
        source_id = claim["source_id"]
        if source_id not in source_embeddings:
            continue

        emb = cal_embeddings[i]
        src_emb = source_embeddings[source_id]
        cosine = float(emb @ src_emb)

        if vtype in ("paraphrase", "extension"):
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


def compute_domain_metrics(embeddings, n_pairs=500):
    """Evaluate on domain-homogeneous data: NN-gap and mean pairwise cosine."""
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
    mask = idx_a == idx_b
    idx_b[mask] = (idx_b[mask] + 1) % n
    np.fill_diagonal(sim_matrix, 1)
    random_cosines = np.array([float(embeddings[a] @ embeddings[b]) for a, b in zip(idx_a, idx_b)])
    mean_random = float(random_cosines.mean())

    # Mean pairwise cosine (isotropy)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    sim_matrix_full = embeddings @ embeddings.T
    mpc = float(sim_matrix_full[upper].mean())

    return {
        "mean_nn_cosine": mean_nn,
        "mean_random_cosine": mean_random,
        "nn_random_gap": mean_nn - mean_random,
        "mean_pairwise_cosine": mpc,
    }


def format_row(method, cal, dom):
    """Format a result as a table row."""
    return (
        f"{method:<25} | "
        f"{cal['discrimination_gap']:>6.4f} | "
        f"{cal['classification_accuracy']:>5.3f} | "
        f"{cal['mean_related_cosine']:>6.4f} | "
        f"{cal['mean_unrelated_cosine']:>6.4f} | "
        f"{cal['mean_contradiction_cosine']:>6.4f} | "
        f"{dom['nn_random_gap']:>6.4f} | "
        f"{dom['mean_pairwise_cosine']:>6.4f}"
    )


# --- Principal Component Analysis ---

def analyze_principal_components(raw_corpus, corpus_texts):
    """Analyze what the top principal components represent."""
    mean = raw_corpus.mean(axis=0)
    centered = raw_corpus - mean
    U, S_vals, Vt = np.linalg.svd(centered, full_matrices=False)

    total_var = (S_vals ** 2).sum()
    print("\n--- Principal Component Analysis ---")
    print(f"Total variance (sum of squared singular values): {total_var:.2f}")
    print(f"\nVariance explained by each of the top 10 components:")
    for i in range(min(10, len(S_vals))):
        var_i = S_vals[i] ** 2
        pct = var_i / total_var * 100
        print(f"  PC{i+1}: singular_value={S_vals[i]:.4f}  variance={var_i:.4f}  "
              f"explains={pct:.2f}%  cumulative={sum(S_vals[:i+1]**2)/total_var*100:.2f}%")

    print(f"\nCumulative variance explained:")
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        if k <= len(S_vals):
            cum = sum(S_vals[:k] ** 2) / total_var * 100
            print(f"  Top {k:>3} PCs: {cum:.2f}%")

    # Characterize top PCs by finding texts with highest/lowest projections
    print(f"\nCharacterizing top PCs (texts with extreme projections):")
    projections = centered @ Vt.T  # (N, dim) — projection onto each PC
    for pc_idx in range(min(3, len(S_vals))):
        proj = projections[:, pc_idx]
        top_pos = np.argsort(proj)[-3:][::-1]
        top_neg = np.argsort(proj)[:3]

        print(f"\n  PC{pc_idx+1} (explains {S_vals[pc_idx]**2/total_var*100:.2f}% variance):")
        print(f"    Most positive projections:")
        for idx in top_pos:
            print(f"      [{proj[idx]:+.3f}] {corpus_texts[idx][:100]}...")
        print(f"    Most negative projections:")
        for idx in top_neg:
            print(f"      [{proj[idx]:+.3f}] {corpus_texts[idx][:100]}...")

    # Eigenvalue spectrum shape analysis
    print(f"\n--- Eigenvalue Spectrum Shape ---")
    ratio_1_2 = S_vals[0] / S_vals[1] if len(S_vals) > 1 else float('inf')
    ratio_1_10 = S_vals[0] / S_vals[9] if len(S_vals) > 9 else float('inf')
    ratio_1_last = S_vals[0] / S_vals[-1]
    print(f"  S1/S2 ratio: {ratio_1_2:.2f}")
    print(f"  S1/S10 ratio: {ratio_1_10:.2f}")
    print(f"  S1/S_last ratio: {ratio_1_last:.2f}")
    print(f"  Effective rank (nuclear norm / spectral norm): "
          f"{S_vals.sum() / S_vals[0]:.1f} out of {len(S_vals)}")

    return mean, centered, S_vals, Vt


def main():
    t0 = time.time()

    # Load data
    corpus_texts = load_whitening_corpus()
    cal_claims = load_calibration_data()
    domain_eval_texts = load_domain_eval_claims()

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

    # Initialize model and embed everything raw
    model = EmbeddingModel()

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

    # --- Part 4: Analyze principal components ---
    mean_corpus, centered_corpus, S_spectrum, Vt_corpus = analyze_principal_components(
        raw_corpus, corpus_texts
    )

    # Compute covariance-based SVD for PCA/Soft-ZCA (same as embed.py)
    mean = raw_corpus.mean(axis=0)
    centered = raw_corpus - mean
    cov = centered.T @ centered / len(corpus_texts)
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    print(f"\nCovariance SVD complete. Top 5 eigenvalues: {S[:5].round(4)}")

    # --- Part 1 & 2 & 3: Evaluate all methods ---
    header = (
        f"{'Method':<25} | "
        f"{'Gap':>6} | {'Acc':>5} | "
        f"{'Rel':>6} | {'Unr':>6} | {'Con':>6} | "
        f"{'NNGap':>6} | {'MPC':>6}"
    )
    separator = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("RESULTS: CALIBRATION (diverse) + OTAK DOMAIN (homogeneous)")
    print(f"Whitening fitted on {len(corpus_texts)} domain claims, evaluated on calibration + separate domain evalk eval")
    print(f"{'='*len(header)}")
    print(header)
    print(separator)

    all_results = []

    # --- Baseline: raw ---
    source_emb_map = {sid: raw_sources[i] for i, sid in enumerate(source_ids)}
    cal_met = compute_calibration_metrics(cal_claims, raw_cal, source_emb_map)
    dom_met = compute_domain_metrics(raw_domain)
    print(format_row("raw (baseline)", cal_met, dom_met))
    all_results.append({
        "method": "raw", "params": {},
        "calibration": cal_met, "domain_homogeneous": dom_met,
    })

    # --- All-but-the-top sweep (D=1,2,3,5) ---
    # ABT is fit on the whitening corpus, then applied to cal/source/domain embeddings
    for D in [1, 2, 3, 5]:
        # Fit on corpus
        _, abt_mean, abt_components = all_but_the_top(raw_corpus, D=D)
        # Apply to eval data
        abt_cal = apply_all_but_top(raw_cal, abt_mean, abt_components)
        abt_sources = apply_all_but_top(raw_sources, abt_mean, abt_components)
        abt_domain = apply_all_but_top(raw_domain, abt_mean, abt_components)

        src_map = {sid: abt_sources[i] for i, sid in enumerate(source_ids)}
        c = compute_calibration_metrics(cal_claims, abt_cal, src_map)
        d = compute_domain_metrics(abt_domain)
        label = f"All-but-top D={D}"
        print(format_row(label, c, d))
        all_results.append({
            "method": "all_but_top", "params": {"D": D},
            "calibration": c, "domain_homogeneous": d,
        })

    # --- Soft-ZCA eps=0.1 (comparison) ---
    w_cal = apply_soft_zca(raw_cal, mean, U, S, 0.1)
    w_sources = apply_soft_zca(raw_sources, mean, U, S, 0.1)
    w_domain = apply_soft_zca(raw_domain, mean, U, S, 0.1)
    src_map = {sid: w_sources[i] for i, sid in enumerate(source_ids)}
    c = compute_calibration_metrics(cal_claims, w_cal, src_map)
    d = compute_domain_metrics(w_domain)
    print(format_row("Soft-ZCA eps=0.1", c, d))
    all_results.append({
        "method": "soft_zca", "params": {"epsilon": 0.1},
        "calibration": c, "domain_homogeneous": d,
    })

    # --- PCA whitening k=128 (comparison) ---
    w_cal = apply_pca_whitening(raw_cal, mean, U, S, 128)
    w_sources = apply_pca_whitening(raw_sources, mean, U, S, 128)
    w_domain = apply_pca_whitening(raw_domain, mean, U, S, 128)
    src_map = {sid: w_sources[i] for i, sid in enumerate(source_ids)}
    c = compute_calibration_metrics(cal_claims, w_cal, src_map)
    d = compute_domain_metrics(w_domain)
    print(format_row("PCA k=128", c, d))
    all_results.append({
        "method": "pca", "params": {"dims": 128},
        "calibration": c, "domain_homogeneous": d,
    })

    print(separator)

    # --- Summary Analysis ---
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    raw_result = all_results[0]
    raw_cal_gap = raw_result["calibration"]["discrimination_gap"]
    raw_dom_gap = raw_result["domain_homogeneous"]["nn_random_gap"]
    raw_cal_acc = raw_result["calibration"]["classification_accuracy"]
    raw_mpc = raw_result["domain_homogeneous"]["mean_pairwise_cosine"]

    print(f"\nRaw baseline:")
    print(f"  Cal gap: {raw_cal_gap:.4f}  Cal acc: {raw_cal_acc:.3f}")
    print(f"  Dom NN-gap: {raw_dom_gap:.4f}  Dom MPC: {raw_mpc:.4f}")

    print(f"\nDelta from baseline:")
    print(f"  {'Method':<25} | {'Cal gap':>10} | {'Cal acc':>10} | {'NN-gap':>10} | {'MPC':>10}")
    print(f"  {'-'*75}")
    for r in all_results[1:]:
        cal_gap_d = r["calibration"]["discrimination_gap"] - raw_cal_gap
        cal_acc_d = r["calibration"]["classification_accuracy"] - raw_cal_acc
        nn_gap_d = r["domain_homogeneous"]["nn_random_gap"] - raw_dom_gap
        mpc_d = r["domain_homogeneous"]["mean_pairwise_cosine"] - raw_mpc
        label = f"{r['method']}({r['params']})"
        print(f"  {label:<25} | {cal_gap_d:>+10.4f} | {cal_acc_d:>+10.3f} | "
              f"{nn_gap_d:>+10.4f} | {mpc_d:>+10.4f}")

    # Percentage improvements
    print(f"\nPercentage improvement over raw baseline:")
    print(f"  {'Method':<25} | {'Cal gap %':>10} | {'NN-gap %':>10}")
    print(f"  {'-'*50}")
    for r in all_results[1:]:
        cal_pct = (r["calibration"]["discrimination_gap"] - raw_cal_gap) / abs(raw_cal_gap) * 100 if raw_cal_gap != 0 else 0
        nn_pct = (r["domain_homogeneous"]["nn_random_gap"] - raw_dom_gap) / abs(raw_dom_gap) * 100 if raw_dom_gap != 0 else 0
        label = f"{r['method']}({r['params']})"
        print(f"  {label:<25} | {cal_pct:>+9.1f}% | {nn_pct:>+9.1f}%")

    # Head-to-head: best ABT vs Soft-ZCA
    abt_results = [r for r in all_results if r["method"] == "all_but_top"]
    zca_result = [r for r in all_results if r["method"] == "soft_zca"][0]

    best_abt_nn = max(abt_results, key=lambda r: r["domain_homogeneous"]["nn_random_gap"])
    best_abt_cal = max(abt_results, key=lambda r: r["calibration"]["discrimination_gap"])

    print(f"\n--- Head-to-Head: Best ABT vs Soft-ZCA ---")
    print(f"\nOn domain-homogeneous data (NN-gap, higher is better):")
    print(f"  Best ABT: D={best_abt_nn['params']['D']}  "
          f"NN-gap={best_abt_nn['domain_homogeneous']['nn_random_gap']:.4f}")
    print(f"  Soft-ZCA: eps=0.1  "
          f"NN-gap={zca_result['domain_homogeneous']['nn_random_gap']:.4f}")
    nn_diff = best_abt_nn['domain_homogeneous']['nn_random_gap'] - zca_result['domain_homogeneous']['nn_random_gap']
    print(f"  Difference: {nn_diff:+.4f} ({'ABT wins' if nn_diff > 0 else 'Soft-ZCA wins'})")

    print(f"\nOn calibration data (discrimination gap, higher is better):")
    print(f"  Best ABT: D={best_abt_cal['params']['D']}  "
          f"gap={best_abt_cal['calibration']['discrimination_gap']:.4f}")
    print(f"  Soft-ZCA: eps=0.1  "
          f"gap={zca_result['calibration']['discrimination_gap']:.4f}")
    cal_diff = best_abt_cal['calibration']['discrimination_gap'] - zca_result['calibration']['discrimination_gap']
    print(f"  Difference: {cal_diff:+.4f} ({'ABT wins' if cal_diff > 0 else 'Soft-ZCA wins'})")

    print(f"\nOn calibration accuracy:")
    best_abt_acc = max(abt_results, key=lambda r: r["calibration"]["classification_accuracy"])
    print(f"  Best ABT: D={best_abt_acc['params']['D']}  "
          f"acc={best_abt_acc['calibration']['classification_accuracy']:.3f}")
    print(f"  Soft-ZCA: eps=0.1  "
          f"acc={zca_result['calibration']['classification_accuracy']:.3f}")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    # Determine if ABT is competitive
    zca_nn_gap = zca_result["domain_homogeneous"]["nn_random_gap"]
    zca_cal_gap = zca_result["calibration"]["discrimination_gap"]

    competitive_abt = [
        r for r in abt_results
        if r["domain_homogeneous"]["nn_random_gap"] >= zca_nn_gap * 0.9
    ]

    if competitive_abt:
        best_comp = max(competitive_abt, key=lambda r: r["domain_homogeneous"]["nn_random_gap"])
        print(f"\nAll-but-the-top D={best_comp['params']['D']} is competitive with Soft-ZCA:")
        print(f"  ABT NN-gap: {best_comp['domain_homogeneous']['nn_random_gap']:.4f} "
              f"vs ZCA: {zca_nn_gap:.4f} "
              f"({(best_comp['domain_homogeneous']['nn_random_gap']/zca_nn_gap - 1)*100:+.1f}%)")
        print(f"  ABT is simpler: no epsilon tuning, no matrix inversion, just remove top PCs.")

        # Check if it MATCHES (within 5%)
        if best_comp['domain_homogeneous']['nn_random_gap'] >= zca_nn_gap * 0.95:
            print(f"\n  MATCHES Soft-ZCA quality (within 5%) at D={best_comp['params']['D']}.")
            print(f"  Worth offering as a simpler alternative.")
        else:
            print(f"\n  Close but does not fully match Soft-ZCA.")
            print(f"  Soft-ZCA still recommended for maximum quality.")
    else:
        print(f"\nAll-but-the-top does NOT match Soft-ZCA quality.")
        print(f"  Best ABT NN-gap: {best_abt_nn['domain_homogeneous']['nn_random_gap']:.4f}")
        print(f"  Soft-ZCA NN-gap: {zca_nn_gap:.4f}")
        print(f"  Gap is too large ({(best_abt_nn['domain_homogeneous']['nn_random_gap']/zca_nn_gap - 1)*100:+.1f}%).")
        print(f"  Soft-ZCA remains the recommended method.")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # --- Save results ---
    total_var = (S_spectrum ** 2).sum()
    output = {
        "experiment": "exp21_all_but_top",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "corpus_size": len(corpus_texts),
        "calibration_size": len(cal_claims),
        "domain_eval_size": len(domain_eval_texts),
        "elapsed_seconds": elapsed,
        "eigenvalue_spectrum": {
            "top_10_singular_values": S_spectrum[:10].tolist(),
            "top_10_variance_explained_pct": [
                float(S_spectrum[i] ** 2 / total_var * 100) for i in range(10)
            ],
            "cumulative_variance": {
                str(k): float(sum(S_spectrum[:k] ** 2) / total_var * 100)
                for k in [1, 2, 3, 5, 10, 20, 50, 100]
            },
            "effective_rank": float(S_spectrum.sum() / S_spectrum[0]),
        },
        "results": all_results,
        "head_to_head": {
            "best_abt_for_nn_gap": {
                "D": best_abt_nn["params"]["D"],
                "nn_gap": best_abt_nn["domain_homogeneous"]["nn_random_gap"],
            },
            "soft_zca": {
                "epsilon": 0.1,
                "nn_gap": zca_result["domain_homogeneous"]["nn_random_gap"],
            },
            "abt_vs_zca_nn_gap_pct": float(
                (best_abt_nn["domain_homogeneous"]["nn_random_gap"] / zca_nn_gap - 1) * 100
            ),
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
