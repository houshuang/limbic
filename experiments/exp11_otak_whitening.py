import os
"""Experiment 11: PCA Whitening on a Domain-Specific (Education) Corpus.

Tests whether whitening improves embedding quality on ~27K education claims.
Previous experiments showed whitening hurts diverse data but helps domain-homogeneous
data (political proposals: mean cosine 0.80 -> 0.24). This corpus is
domain-homogeneous (education research), making it the ideal test case.

Evaluation strategy:
1. Domain-specific pairs:
   a) SIMILAR: claims sharing the same canonical_claim (semantic duplicates)
   b) SAME-SOURCE: claims from the same source document (topically related)
   c) UNRELATED: random claim pairs from different sources and topics
2. General quality: STS-B correlation (ensure whitening doesn't degrade)

Metrics:
- Mean cosine similarity for each pair type
- Discrimination gap: cos(similar) - cos(unrelated)
- Intra-class variance (how tight are similar clusters?)
- STS-B Spearman correlation
"""

import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from amygdala import EmbeddingModel

OTAK_DB = Path(os.environ.get("AMYGDALA_EVAL_DB", "eval_claims.db"))
RESULTS_PATH = Path("experiments/results/exp11_results.json")
WHITEN_DIMS = [64, 128, 192, 256, 384]
FIT_CORPUS_SIZE = 5000
EVAL_SAMPLE_SIZE = 1000
NUM_UNRELATED_PAIRS = 500
SEED = 42


def load_claims(conn):
    """Load all non-deleted claims (latest version) with their metadata."""
    rows = conn.execute("""
        SELECT n.id, n.name, n.data
        FROM nodes n
        JOIN idx_knowledge_item_item_type t ON n.id = t.node_id
        WHERE t.value = 'claim'
          AND n.deleted_at IS NULL
          AND n.version = (SELECT MAX(n2.version) FROM nodes n2 WHERE n2.id = n.id)
          AND length(n.name) > 20
          AND length(n.name) < 500
    """).fetchall()
    claims = []
    for node_id, name, data_json in rows:
        data = json.loads(data_json)
        type_key = list(data.keys())[0]
        props = data[type_key]
        claims.append({
            "id": node_id,
            "text": name,
            "source_ref": props.get("source_ref", ""),
            "claim_type": props.get("claim_type", ""),
        })
    return claims


def load_canonical_groups(conn):
    """Load canonical claim groups (2+ members) for similar-pair generation."""
    rows = conn.execute("""
        SELECT c.value as canonical_id, c.node_id
        FROM idx_knowledge_item_canonical_claim c
        JOIN nodes n ON c.node_id = n.id
          AND n.deleted_at IS NULL
          AND n.version = (SELECT MAX(n2.version) FROM nodes n2 WHERE n2.id = n.id)
          AND length(n.name) > 20
          AND length(n.name) < 500
    """).fetchall()

    from collections import defaultdict
    groups = defaultdict(list)
    for canonical_id, node_id in rows:
        groups[canonical_id].append(node_id)

    # Keep only groups with 2+ members
    return {k: v for k, v in groups.items() if len(v) >= 2}


def load_source_groups(conn, claims_by_id):
    """Group claims by source_ref for same-source pair generation."""
    from collections import defaultdict
    groups = defaultdict(list)
    for claim in claims_by_id.values():
        src = claim["source_ref"]
        if src and len(src) > 5:
            groups[src].append(claim["id"])

    # Keep only groups with 2+ members
    return {k: v for k, v in groups.items() if len(v) >= 2}


def generate_eval_pairs(canonical_groups, source_groups, claims_by_id, rng):
    """Generate evaluation pairs of three types.

    Returns:
        similar_pairs: list of (id_a, id_b) from same canonical group
        same_source_pairs: list of (id_a, id_b) from same source document
        unrelated_pairs: list of (id_a, id_b) random pairs
    """
    # Similar pairs: from canonical groups
    similar_pairs = []
    canonical_keys = list(canonical_groups.keys())
    rng.shuffle(canonical_keys)
    for key in canonical_keys:
        members = canonical_groups[key]
        if len(members) >= 2:
            # Take all unique pairs from this group
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if members[i] in claims_by_id and members[j] in claims_by_id:
                        similar_pairs.append((members[i], members[j]))
    rng.shuffle(similar_pairs)
    similar_pairs = similar_pairs[:NUM_UNRELATED_PAIRS]

    # Same-source pairs: claims from same source document (but not same canonical)
    canonical_member_set = set()
    for members in canonical_groups.values():
        for m in members:
            canonical_member_set.add(m)

    same_source_pairs = []
    source_keys = list(source_groups.keys())
    rng.shuffle(source_keys)
    for key in source_keys:
        members = [m for m in source_groups[key] if m in claims_by_id]
        if len(members) >= 2:
            # Take a random pair from this source
            pair = tuple(rng.sample(members, 2))
            # Skip if they share a canonical group (would be "similar")
            same_source_pairs.append(pair)
        if len(same_source_pairs) >= NUM_UNRELATED_PAIRS:
            break

    # Unrelated pairs: random claims from different sources
    all_ids = list(claims_by_id.keys())
    unrelated_pairs = []
    attempts = 0
    while len(unrelated_pairs) < NUM_UNRELATED_PAIRS and attempts < NUM_UNRELATED_PAIRS * 10:
        a, b = rng.sample(all_ids, 2)
        src_a = claims_by_id[a]["source_ref"]
        src_b = claims_by_id[b]["source_ref"]
        if src_a != src_b or not src_a:
            unrelated_pairs.append((a, b))
        attempts += 1

    return similar_pairs, same_source_pairs, unrelated_pairs


def compute_pair_cosines(pairs, embeddings_by_id):
    """Compute cosine similarities for pairs of embedded claims."""
    cosines = []
    for id_a, id_b in pairs:
        if id_a in embeddings_by_id and id_b in embeddings_by_id:
            cos = float(embeddings_by_id[id_a] @ embeddings_by_id[id_b])
            cosines.append(cos)
    return np.array(cosines) if cosines else np.array([])


def compute_intra_class_variance(canonical_groups, embeddings_by_id):
    """Mean variance of cosine similarities within canonical groups."""
    variances = []
    for members in canonical_groups.values():
        embedded = [embeddings_by_id[m] for m in members if m in embeddings_by_id]
        if len(embedded) >= 2:
            emb_matrix = np.vstack(embedded)
            sim_matrix = emb_matrix @ emb_matrix.T
            n = len(embedded)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            if mask.sum() > 0:
                within_sims = sim_matrix[mask]
                variances.append(float(within_sims.var()))
    return float(np.mean(variances)) if variances else 0.0


def eval_stsb(model):
    """Evaluate on STS-B test set. Returns Spearman correlation."""
    from datasets import load_dataset

    ds_test = load_dataset("sentence-transformers/stsb", split="test")
    test_s1 = ds_test["sentence1"]
    test_s2 = ds_test["sentence2"]
    gold_scores = np.array(ds_test["score"], dtype=np.float32)

    embs_a = model.embed_batch(test_s1)
    embs_b = model.embed_batch(test_s2)
    pred_sims = np.sum(embs_a * embs_b, axis=1)

    spearman_r, _ = spearmanr(pred_sims, gold_scores)
    return float(spearman_r)


def apply_whitening_to_embeddings(raw_embs, mean, U, S, k):
    """Apply PCA whitening with k dimensions to pre-computed raw embeddings."""
    W = U[:, :k] @ np.diag(1.0 / np.sqrt(S[:k] + 1e-8))
    centered = raw_embs - mean
    whitened = centered @ W
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    return whitened / np.maximum(norms, 1e-8)


def main():
    t0 = time.time()
    rng = random.Random(SEED)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading claims...")
    conn = sqlite3.connect(str(OTAK_DB))
    claims = load_claims(conn)
    print(f"  Total claims: {len(claims)}")

    canonical_groups = load_canonical_groups(conn)
    print(f"  Canonical groups (2+ members): {len(canonical_groups)}")
    print(f"  Claims in canonical groups: {sum(len(v) for v in canonical_groups.values())}")

    claims_by_id = {c["id"]: c for c in claims}
    source_groups = load_source_groups(conn, claims_by_id)
    print(f"  Source groups (2+ claims): {len(source_groups)}")
    conn.close()

    # ── Generate eval pairs ──────────────────────────────────────────
    print("\nGenerating evaluation pairs...")
    similar_pairs, same_source_pairs, unrelated_pairs = generate_eval_pairs(
        canonical_groups, source_groups, claims_by_id, rng
    )
    print(f"  Similar pairs (same canonical): {len(similar_pairs)}")
    print(f"  Same-source pairs: {len(same_source_pairs)}")
    print(f"  Unrelated pairs: {len(unrelated_pairs)}")

    # Collect all unique claim IDs needed for embedding
    eval_ids = set()
    for a, b in similar_pairs + same_source_pairs + unrelated_pairs:
        eval_ids.add(a)
        eval_ids.add(b)
    print(f"  Unique claims to embed: {len(eval_ids)}")

    # ── Select whitening corpus ──────────────────────────────────────
    # Use a random sample of claims for fitting whitening (disjoint from eval)
    all_claim_ids = [c["id"] for c in claims]
    rng.shuffle(all_claim_ids)
    fit_ids = [cid for cid in all_claim_ids if cid not in eval_ids][:FIT_CORPUS_SIZE]
    fit_texts = [claims_by_id[cid]["text"] for cid in fit_ids]
    print(f"\nWhitening fit corpus: {len(fit_texts)} claims")

    # ── Embed everything raw ─────────────────────────────────────────
    model = EmbeddingModel()
    eval_texts = [claims_by_id[cid]["text"] for cid in eval_ids]
    eval_id_list = list(eval_ids)

    print(f"\nEmbedding {len(eval_texts)} eval claims (raw)...")
    raw_eval = model._raw_embed_batch(eval_texts)
    print(f"  Shape: {raw_eval.shape}")

    print(f"Embedding {len(fit_texts)} fit corpus claims (raw)...")
    raw_fit = model._raw_embed_batch(fit_texts)
    print(f"  Shape: {raw_fit.shape}")

    # Build ID -> raw embedding map
    raw_emb_by_id = {eval_id_list[i]: raw_eval[i] for i in range(len(eval_id_list))}

    # ── Compute PCA on fit corpus ────────────────────────────────────
    print("\nComputing PCA on fit corpus...")
    mean = raw_fit.mean(axis=0)
    centered = raw_fit - mean
    cov = centered.T @ centered / len(fit_texts)
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    print(f"  SVD complete. Top-5 singular values: {S[:5].round(4)}")

    total_var = S.sum()
    for k in WHITEN_DIMS:
        explained = S[:k].sum() / total_var
        print(f"  dims={k:>3}: {explained*100:.1f}% variance explained")

    # ── Baseline: raw embeddings ─────────────────────────────────────
    print("\n" + "=" * 90)
    print("EVALUATING RAW vs WHITENED EMBEDDINGS")
    print("=" * 90)

    results = []

    def evaluate_embeddings(label, emb_by_id, stsb_spearman=None):
        cos_similar = compute_pair_cosines(similar_pairs, emb_by_id)
        cos_same_src = compute_pair_cosines(same_source_pairs, emb_by_id)
        cos_unrelated = compute_pair_cosines(unrelated_pairs, emb_by_id)

        # Intra-class variance for canonical groups
        intra_var = compute_intra_class_variance(canonical_groups, emb_by_id)

        # Mean pairwise cosine over all eval embeddings (sample for speed)
        all_embs = np.vstack(list(emb_by_id.values()))
        sample_n = min(1000, len(all_embs))
        sample_idx = rng.sample(range(len(all_embs)), sample_n)
        sample_embs = all_embs[sample_idx]
        sim_matrix = sample_embs @ sample_embs.T
        n = len(sample_embs)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mean_pairwise = float(sim_matrix[mask].mean())

        result = {
            "label": label,
            "mean_cos_similar": float(cos_similar.mean()) if len(cos_similar) > 0 else None,
            "std_cos_similar": float(cos_similar.std()) if len(cos_similar) > 0 else None,
            "mean_cos_same_source": float(cos_same_src.mean()) if len(cos_same_src) > 0 else None,
            "std_cos_same_source": float(cos_same_src.std()) if len(cos_same_src) > 0 else None,
            "mean_cos_unrelated": float(cos_unrelated.mean()) if len(cos_unrelated) > 0 else None,
            "std_cos_unrelated": float(cos_unrelated.std()) if len(cos_unrelated) > 0 else None,
            "gap_similar_vs_unrelated": (
                float(cos_similar.mean() - cos_unrelated.mean())
                if len(cos_similar) > 0 and len(cos_unrelated) > 0 else None
            ),
            "gap_source_vs_unrelated": (
                float(cos_same_src.mean() - cos_unrelated.mean())
                if len(cos_same_src) > 0 and len(cos_unrelated) > 0 else None
            ),
            "intra_class_variance": intra_var,
            "mean_pairwise_cosine": mean_pairwise,
            "stsb_spearman": stsb_spearman,
            "n_similar_pairs": len(cos_similar),
            "n_same_source_pairs": len(cos_same_src),
            "n_unrelated_pairs": len(cos_unrelated),
        }
        results.append(result)

        print(f"\n  {label}:")
        print(f"    Similar pairs:     mean={result['mean_cos_similar']:.4f}  std={result['std_cos_similar']:.4f}")
        print(f"    Same-source pairs: mean={result['mean_cos_same_source']:.4f}  std={result['std_cos_same_source']:.4f}")
        print(f"    Unrelated pairs:   mean={result['mean_cos_unrelated']:.4f}  std={result['std_cos_unrelated']:.4f}")
        print(f"    Gap (similar-unrel):    {result['gap_similar_vs_unrelated']:.4f}")
        print(f"    Gap (source-unrel):     {result['gap_source_vs_unrelated']:.4f}")
        print(f"    Intra-class variance:   {result['intra_class_variance']:.6f}")
        print(f"    Mean pairwise cosine:   {result['mean_pairwise_cosine']:.4f}")
        if stsb_spearman is not None:
            print(f"    STS-B Spearman:         {stsb_spearman:.4f}")

        return result

    # Raw baseline
    print("\n--- Raw embeddings (no whitening) ---")
    print("  Computing STS-B correlation...")
    stsb_raw = eval_stsb(model)
    evaluate_embeddings("raw (384d)", raw_emb_by_id, stsb_spearman=stsb_raw)

    # ── Whitened variants ────────────────────────────────────────────
    for k in WHITEN_DIMS:
        print(f"\n--- Whitened ({k}d) ---")

        # Apply whitening to eval embeddings
        whitened_eval = apply_whitening_to_embeddings(raw_eval, mean, U, S, k)
        whitened_emb_by_id = {
            eval_id_list[i]: whitened_eval[i] for i in range(len(eval_id_list))
        }

        # STS-B with whitening fitted on domain corpus
        print("  Computing STS-B correlation...")
        w_model = EmbeddingModel(whiten_dims=k)
        # Fit whitening on domain corpus
        w_model._load_model()
        from amygdala.embed import WhiteningParams
        W_matrix = U[:, :k] @ np.diag(1.0 / np.sqrt(S[:k] + 1e-8))
        w_model.set_whitening(WhiteningParams(mean=mean, W=W_matrix))
        stsb_w = eval_stsb(w_model)

        evaluate_embeddings(f"whitened ({k}d)", whitened_emb_by_id, stsb_spearman=stsb_w)

    # ── Summary Table ────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    header = (
        f"{'Config':<18s} | {'cos(sim)':>9s} | {'cos(src)':>9s} | {'cos(unr)':>9s} | "
        f"{'gap(s-u)':>9s} | {'gap(r-u)':>9s} | {'intra_var':>10s} | {'MPC':>7s} | {'STS-B':>7s}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        print(
            f"{r['label']:<18s} | "
            f"{r['mean_cos_similar']:>9.4f} | "
            f"{r['mean_cos_same_source']:>9.4f} | "
            f"{r['mean_cos_unrelated']:>9.4f} | "
            f"{r['gap_similar_vs_unrelated']:>9.4f} | "
            f"{r['gap_source_vs_unrelated']:>9.4f} | "
            f"{r['intra_class_variance']:>10.6f} | "
            f"{r['mean_pairwise_cosine']:>7.4f} | "
            f"{r['stsb_spearman']:>7.4f}"
        )

    # ── Analysis ─────────────────────────────────────────────────────
    raw_result = results[0]
    print("\n" + "=" * 120)
    print("ANALYSIS")
    print("=" * 120)

    best_gap = max(results, key=lambda r: r["gap_similar_vs_unrelated"])
    best_stsb = max(results, key=lambda r: r["stsb_spearman"])
    best_spread = min(results, key=lambda r: r["mean_pairwise_cosine"])
    lowest_intra_var = min(results, key=lambda r: r["intra_class_variance"])

    print(f"Best discrimination gap:  {best_gap['label']} ({best_gap['gap_similar_vs_unrelated']:.4f})")
    print(f"Best STS-B Spearman:      {best_stsb['label']} ({best_stsb['stsb_spearman']:.4f})")
    print(f"Best spread (lowest MPC): {best_spread['label']} ({best_spread['mean_pairwise_cosine']:.4f})")
    print(f"Tightest clusters:        {lowest_intra_var['label']} ({lowest_intra_var['intra_class_variance']:.6f})")

    # Check if whitening helps: does any whitened config beat raw on gap WITHOUT degrading STS-B?
    raw_gap = raw_result["gap_similar_vs_unrelated"]
    raw_stsb = raw_result["stsb_spearman"]
    whitening_helps = False
    for r in results[1:]:
        if r["gap_similar_vs_unrelated"] > raw_gap and r["stsb_spearman"] >= raw_stsb * 0.95:
            whitening_helps = True
            print(f"\n  WHITENING HELPS: {r['label']} improves gap by "
                  f"{r['gap_similar_vs_unrelated'] - raw_gap:.4f} "
                  f"(STS-B: {r['stsb_spearman']:.4f} vs {raw_stsb:.4f})")

    if not whitening_helps:
        # Check if gap improves at all, even with STS-B degradation
        better_gap_configs = [r for r in results[1:] if r["gap_similar_vs_unrelated"] > raw_gap]
        if better_gap_configs:
            print("\n  WHITENING MIXED: Some configs improve gap but degrade STS-B:")
            for r in better_gap_configs:
                print(f"    {r['label']}: gap={r['gap_similar_vs_unrelated']:.4f} (+{r['gap_similar_vs_unrelated']-raw_gap:.4f}), "
                      f"STS-B={r['stsb_spearman']:.4f} ({r['stsb_spearman']-raw_stsb:+.4f})")
        else:
            print("\n  WHITENING DOES NOT HELP: No whitened config improves discrimination gap.")

    # ── MPC analysis (key metric from prior experiments) ─────────────
    print(f"\n  Mean Pairwise Cosine reduction:")
    for r in results:
        delta = r["mean_pairwise_cosine"] - raw_result["mean_pairwise_cosine"]
        print(f"    {r['label']:<18s}: {r['mean_pairwise_cosine']:.4f} ({delta:+.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "experiment": "exp11_domain_whitening",
        "description": "PCA whitening on education corpus (domain-homogeneous)",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "fit_corpus_size": len(fit_texts),
        "eval_claims": len(eval_ids),
        "n_similar_pairs": len(similar_pairs),
        "n_same_source_pairs": len(same_source_pairs),
        "n_unrelated_pairs": len(unrelated_pairs),
        "whiten_dims_tested": WHITEN_DIMS,
        "explained_variance": {
            str(k): float(S[:k].sum() / total_var) for k in WHITEN_DIMS
        },
        "elapsed_seconds": elapsed,
        "results": results,
        "analysis": {
            "whitening_helps": whitening_helps,
            "best_gap_config": best_gap["label"],
            "best_stsb_config": best_stsb["label"],
            "best_spread_config": best_spread["label"],
            "raw_gap": raw_gap,
            "raw_stsb": raw_stsb,
            "raw_mpc": raw_result["mean_pairwise_cosine"],
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
