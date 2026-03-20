"""STS Benchmark Evaluation: Embedding Models + Whitening

Evaluates embedding models on the STS-B test set (1,379 human-annotated
sentence pairs, similarity 0-5 scaled to 0-1). Compares raw models and
whitening variants (fit on train split, eval on test split).

Metrics: Spearman correlation, Pearson correlation, mean absolute error.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

from amygdala import EmbeddingModel

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", 384, "Multilingual-MiniLM (384d)"),
    ("all-MiniLM-L6-v2", 384, "MiniLM-L6 (384d)"),
    ("all-mpnet-base-v2", 768, "MPNet-base (768d)"),
]

WHITEN_DIMS = [256, 128]


def cosine_similarities(embs_a: np.ndarray, embs_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two L2-normalized matrices."""
    return np.sum(embs_a * embs_b, axis=1)


def evaluate(predicted_sims: np.ndarray, gold_scores: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    spearman_r, spearman_p = spearmanr(predicted_sims, gold_scores)
    pearson_r, pearson_p = pearsonr(predicted_sims, gold_scores)

    # Scale predicted cosine [-1, 1] -> [0, 1] for MAE comparison
    pred_scaled = (predicted_sims + 1) / 2
    mae = float(np.mean(np.abs(pred_scaled - gold_scores)))

    return {
        "spearman": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mae": mae,
    }


def main():
    from datasets import load_dataset

    print("Loading STS-B dataset...")
    ds_train = load_dataset("sentence-transformers/stsb", split="train")
    ds_test = load_dataset("sentence-transformers/stsb", split="test")
    print(f"  Train: {len(ds_train)} pairs (used for whitening fit)")
    print(f"  Test:  {len(ds_test)} pairs (used for evaluation)")

    # Extract test data
    test_s1 = ds_test["sentence1"]
    test_s2 = ds_test["sentence2"]
    gold_scores = np.array(ds_test["score"], dtype=np.float32)

    # Extract train texts for whitening (unique sentences)
    train_texts = list(set(list(ds_train["sentence1"]) + list(ds_train["sentence2"])))
    print(f"  Unique train sentences for whitening: {len(train_texts)}")

    all_results = []

    for model_name, native_dim, label in MODELS:
        print(f"\n{'='*70}")
        print(f"Model: {label} ({model_name})")
        print(f"{'='*70}")

        # ── Raw model (no whitening) ──────────────────────────────────
        model = EmbeddingModel(model_name=model_name)

        t0 = time.perf_counter()
        embs_a = model.embed_batch(test_s1)
        embs_b = model.embed_batch(test_s2)
        embed_time = time.perf_counter() - t0

        pred_sims = cosine_similarities(embs_a, embs_b)
        metrics = evaluate(pred_sims, gold_scores)

        result = {
            "model": label,
            "model_name": model_name,
            "variant": "raw",
            "dim": native_dim,
            "embed_time_s": round(embed_time, 2),
            **metrics,
        }
        all_results.append(result)

        print(f"\n  Raw ({native_dim}d):")
        print(f"    Spearman: {metrics['spearman']:.4f}")
        print(f"    Pearson:  {metrics['pearson']:.4f}")
        print(f"    MAE:      {metrics['mae']:.4f}")
        print(f"    Time:     {embed_time:.2f}s")

        # Cosine similarity distribution for diagnostics
        print(f"    Cosine dist: mean={pred_sims.mean():.3f}, "
              f"std={pred_sims.std():.3f}, "
              f"min={pred_sims.min():.3f}, max={pred_sims.max():.3f}")

        # ── Whitened variants ─────────────────────────────────────────
        for whiten_dim in WHITEN_DIMS:
            if whiten_dim >= native_dim:
                continue  # skip if whitening dim >= native dim

            w_model = EmbeddingModel(
                model_name=model_name, whiten_dims=whiten_dim
            )

            print(f"\n  Fitting whitening ({whiten_dim}d) on {len(train_texts)} train sentences...")
            t0 = time.perf_counter()
            w_model.fit_whitening(train_texts)
            fit_time = time.perf_counter() - t0
            print(f"    Whitening fit: {fit_time:.2f}s")

            t0 = time.perf_counter()
            w_embs_a = w_model.embed_batch(test_s1)
            w_embs_b = w_model.embed_batch(test_s2)
            w_embed_time = time.perf_counter() - t0

            w_pred_sims = cosine_similarities(w_embs_a, w_embs_b)
            w_metrics = evaluate(w_pred_sims, gold_scores)

            w_result = {
                "model": label,
                "model_name": model_name,
                "variant": f"whitened-{whiten_dim}d",
                "dim": whiten_dim,
                "embed_time_s": round(w_embed_time, 2),
                "whiten_fit_time_s": round(fit_time, 2),
                **w_metrics,
            }
            all_results.append(w_result)

            print(f"  Whitened ({whiten_dim}d):")
            print(f"    Spearman: {w_metrics['spearman']:.4f}")
            print(f"    Pearson:  {w_metrics['pearson']:.4f}")
            print(f"    MAE:      {w_metrics['mae']:.4f}")
            print(f"    Time:     {w_embed_time:.2f}s")
            print(f"    Cosine dist: mean={w_pred_sims.mean():.3f}, "
                  f"std={w_pred_sims.std():.3f}, "
                  f"min={w_pred_sims.min():.3f}, max={w_pred_sims.max():.3f}")

        del model  # free memory before loading next model

    # ── Summary Table ─────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("STS-B EVALUATION RESULTS")
    print(f"{'='*100}")
    header = (
        f"{'Model':<35s} {'Variant':<15s} {'Dim':>4s} "
        f"{'Spearman':>9s} {'Pearson':>9s} {'MAE':>7s} {'Time(s)':>8s}"
    )
    print(header)
    print("-" * 100)

    for r in all_results:
        print(
            f"{r['model']:<35s} {r['variant']:<15s} {r['dim']:>4d} "
            f"{r['spearman']:>9.4f} {r['pearson']:>9.4f} {r['mae']:>7.4f} "
            f"{r['embed_time_s']:>8.2f}"
        )

    # ── Best result ───────────────────────────────────────────────────
    best = max(all_results, key=lambda x: x["spearman"])
    print(f"\nBest Spearman: {best['model']} ({best['variant']}, {best['dim']}d) "
          f"= {best['spearman']:.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "eval_stsb_results.json"
    output = {
        "benchmark": "STS-B",
        "test_pairs": len(ds_test),
        "train_sentences_for_whitening": len(train_texts),
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
