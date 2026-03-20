import os
"""Experiment 1: Embedding Model Comparison

Compares embedding models on discrimination, threshold accuracy, latency,
and cross-lingual (Norwegian-English) similarity using calibration claims.
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────
CALIBRATION_PATH = Path(os.environ.get("AMYGDALA_CALIBRATION_DATA", "experiments/eval_data/calibration_claims.json"))
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ─────────────────────────────────────────────────────────────
MODELS = {
    "MiniLM-L6 (384d)": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "trust_remote_code": False,
        "prefix_query": None,
        "prefix_doc": None,
        "matryoshka_dims": None,
    },
    "Nomic-v1.5 (768d)": {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "trust_remote_code": True,
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
        "matryoshka_dims": [768, 384, 256],
    },
    "MPNet-base (768d)": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "trust_remote_code": False,
        "prefix_query": None,
        "prefix_doc": None,
        "matryoshka_dims": None,
    },
    "Multilingual-MiniLM (384d)": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "trust_remote_code": False,
        "prefix_query": None,
        "prefix_doc": None,
        "matryoshka_dims": None,
    },
}

# ── Norwegian test pairs ───────────────────────────────────────────────
NORWEGIAN_PAIRS = [
    (
        "Utdanning er viktig for demokratiet",
        "Education is important for democracy",
    ),
    (
        "Fagfornyelsen vektlegger dybdelæring",
        "The curriculum reform emphasizes deep learning",
    ),
    (
        "Kommunen bør satse på kollektivtransport",
        "The municipality should invest in public transport",
    ),
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalized vectors (= dot product)."""
    return float(np.dot(a, b))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize vectors (handles both 1-D and 2-D)."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / max(norm, 1e-8)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-8)


def truncate_matryoshka(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """Truncate embeddings to `dim` dimensions and re-normalize."""
    truncated = embeddings[:, :dim]
    return l2_normalize(truncated)


def embed_texts(model: SentenceTransformer, texts: list[str],
                prefix: str | None = None) -> np.ndarray:
    """Embed a list of texts, optionally prepending a prefix."""
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(
        texts, batch_size=64, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    ).astype(np.float32)


def optimal_threshold_accuracy(similarities: list[float],
                               labels: list[bool]) -> tuple[float, float]:
    """Find threshold that maximizes accuracy. Returns (threshold, accuracy)."""
    best_acc, best_thresh = 0.0, 0.0
    thresholds = sorted(set(similarities))
    for t in thresholds:
        preds = [s >= t for s in similarities]
        correct = sum(p == l for p, l in zip(preds, labels))
        acc = correct / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc


def evaluate_model(model_key: str, config: dict,
                   claims: list[dict], source_texts: list[str],
                   source_ids: list[int]) -> list[dict]:
    """Evaluate a single model (possibly at multiple Matryoshka dims).
    Returns list of result dicts (one per dimension variant)."""

    print(f"\n{'='*60}")
    print(f"Loading {model_key}: {config['name']}")
    print(f"{'='*60}")

    model = SentenceTransformer(
        config["name"],
        trust_remote_code=config["trust_remote_code"],
    )

    claim_texts = [c["text"] for c in claims]
    variant_types = [c["variant_type"] for c in claims]
    claim_source_ids = [c["source_id"] for c in claims]

    # Build source_id -> index mapping
    sid_to_idx = {sid: i for i, sid in enumerate(source_ids)}

    # Embed with timing
    t0 = time.perf_counter()
    claim_embs = embed_texts(model, claim_texts, config["prefix_query"])
    source_embs = embed_texts(model, source_texts, config["prefix_doc"])
    latency = time.perf_counter() - t0

    # Norwegian pairs
    no_texts = [p[0] for p in NORWEGIAN_PAIRS]
    en_texts = [p[1] for p in NORWEGIAN_PAIRS]
    no_embs = embed_texts(model, no_texts, config["prefix_query"])
    en_embs = embed_texts(model, en_texts, config["prefix_doc"])

    # Determine which dimensions to evaluate
    dims_to_eval = config["matryoshka_dims"] or [claim_embs.shape[1]]

    results = []
    for dim in dims_to_eval:
        if dim == claim_embs.shape[1]:
            c_embs = claim_embs
            s_embs = source_embs
            n_embs = no_embs
            e_embs = en_embs
            dim_label = f"{model_key}"
        else:
            c_embs = truncate_matryoshka(claim_embs, dim)
            s_embs = truncate_matryoshka(source_embs, dim)
            n_embs = truncate_matryoshka(no_embs, dim)
            e_embs = truncate_matryoshka(en_embs, dim)
            dim_label = f"Nomic-v1.5 ({dim}d)"

        # Compute per-claim cosine to its source
        sims_by_type: dict[str, list[float]] = {
            "paraphrase": [], "extension": [], "contradiction": [], "unrelated": []
        }
        all_sims = []
        all_labels = []  # True = should be above threshold (paraphrase/extension)

        for i, claim in enumerate(claims):
            src_idx = sid_to_idx[claim["source_id"]]
            sim = cosine_sim(c_embs[i], s_embs[src_idx])
            sims_by_type[claim["variant_type"]].append(sim)
            all_sims.append(sim)
            all_labels.append(claim["variant_type"] in ("paraphrase", "extension"))

        # Per-type mean cosine
        type_means = {t: float(np.mean(v)) for t, v in sims_by_type.items()}

        # Discrimination gap
        disc_gap = type_means["paraphrase"] - type_means["unrelated"]

        # Threshold accuracy
        thresh, acc = optimal_threshold_accuracy(all_sims, all_labels)

        # Norwegian cross-lingual similarity
        no_sims = [cosine_sim(n_embs[i], e_embs[i]) for i in range(len(NORWEGIAN_PAIRS))]

        result = {
            "model": dim_label,
            "dim": dim,
            "per_type_mean_cosine": type_means,
            "discrimination_gap": disc_gap,
            "optimal_threshold": thresh,
            "threshold_accuracy": acc,
            "latency_s": latency,
            "norwegian_pairs": [
                {"no": p[0], "en": p[1], "cosine": s}
                for p, s in zip(NORWEGIAN_PAIRS, no_sims)
            ],
            "norwegian_mean": float(np.mean(no_sims)),
        }
        results.append(result)

        # Print summary
        print(f"\n  --- {dim_label} ---")
        print(f"  Per-type mean cosine:")
        for t in ["paraphrase", "extension", "contradiction", "unrelated"]:
            print(f"    {t:15s}: {type_means[t]:.4f}")
        print(f"  Discrimination gap (para-unrel): {disc_gap:.4f}")
        print(f"  Optimal threshold: {thresh:.4f} → accuracy: {acc:.1%}")
        print(f"  Embedding latency (150 texts): {latency:.3f}s")
        print(f"  Norwegian cross-lingual:")
        for pair, sim in zip(NORWEGIAN_PAIRS, no_sims):
            print(f"    {pair[0][:40]:40s} ↔ {pair[1][:40]:40s} = {sim:.4f}")
        print(f"  Norwegian mean: {float(np.mean(no_sims)):.4f}")

    del model  # free GPU/CPU memory
    return results


def main():
    # Load calibration data
    with open(CALIBRATION_PATH) as f:
        claims = json.load(f)

    print(f"Loaded {len(claims)} claims")

    # Extract unique source claims
    source_map = {}
    for c in claims:
        source_map[c["source_id"]] = c["source_claim"]
    source_ids = sorted(source_map.keys())
    source_texts = [source_map[sid] for sid in source_ids]
    print(f"Found {len(source_ids)} unique source claims")

    all_results = []
    for model_key, config in MODELS.items():
        results = evaluate_model(model_key, config, claims, source_texts, source_ids)
        all_results.extend(results)

    # ── Summary table ──────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    header = (
        f"{'Model':<30s} {'Dim':>4s} "
        f"{'Para':>6s} {'Ext':>6s} {'Contr':>6s} {'Unrel':>6s} "
        f"{'Gap':>6s} {'Thresh':>6s} {'Acc':>6s} "
        f"{'Lat(s)':>7s} {'NO↔EN':>6s}"
    )
    print(header)
    print("-" * 120)
    for r in all_results:
        tm = r["per_type_mean_cosine"]
        print(
            f"{r['model']:<30s} {r['dim']:>4d} "
            f"{tm['paraphrase']:>6.4f} {tm['extension']:>6.4f} "
            f"{tm['contradiction']:>6.4f} {tm['unrelated']:>6.4f} "
            f"{r['discrimination_gap']:>6.4f} "
            f"{r['optimal_threshold']:>6.4f} {r['threshold_accuracy']:>6.1%} "
            f"{r['latency_s']:>7.3f} {r['norwegian_mean']:>6.4f}"
        )

    # ── Norwegian detail ───────────────────────────────────────────
    print(f"\n{'='*120}")
    print("NORWEGIAN CROSS-LINGUAL DETAIL")
    print(f"{'='*120}")
    for r in all_results:
        print(f"\n  {r['model']}:")
        for p in r["norwegian_pairs"]:
            print(f"    {p['no'][:45]:45s} ↔ {p['en'][:45]:45s} = {p['cosine']:.4f}")

    # ── Save JSON ──────────────────────────────────────────────────
    out_path = RESULTS_DIR / "exp1_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
