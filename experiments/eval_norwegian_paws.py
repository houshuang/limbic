"""Norwegian PAWS-X Evaluation: Paraphrase Detection via Cosine Similarity

Tests embedding models on NbAiLab/norwegian-paws-x — an adversarial paraphrase
dataset where pairs have high lexical overlap but different semantics. This tests
whether models capture meaning beyond surface-level word matching.

Both Bokmal (nb) and Nynorsk (nn) splits are evaluated. We compare the
multilingual default model against the English-only all-MiniLM-L6-v2 to
quantify the multilingual advantage on Norwegian text.

Metrics: AUC-ROC, best F1, mean cosine for paraphrases vs non-paraphrases.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/tmp/paws-no/x-final")

MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", "Multilingual-MiniLM (default)"),
    ("all-MiniLM-L6-v2", "MiniLM-L6 (English-only)"),
]

SAMPLE_SIZE = 2000  # max per variant (dataset has exactly 2000)


def load_paws_jsonl(path: Path) -> list[dict]:
    """Load PAWS-X JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def balanced_sample(data: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """Sample n items with balanced labels (n/2 per class)."""
    pos = [d for d in data if d["label"] == 1]
    neg = [d for d in data if d["label"] == 0]
    half = n // 2
    if len(pos) < half or len(neg) < half:
        half = min(len(pos), len(neg))
    sampled_pos = rng.choice(pos, size=min(half, len(pos)), replace=False).tolist()
    sampled_neg = rng.choice(neg, size=min(half, len(neg)), replace=False).tolist()
    combined = sampled_pos + sampled_neg
    rng.shuffle(combined)
    return combined


def cosine_similarities(embs_a: np.ndarray, embs_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between L2-normalized matrices."""
    return np.sum(embs_a * embs_b, axis=1)


def compute_auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-ROC using the trapezoidal rule (no sklearn dependency)."""
    desc_idx = np.argsort(-scores)
    labels_sorted = labels[desc_idx]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return float(auc)


def best_f1_at_threshold(labels: np.ndarray, scores: np.ndarray, n_thresholds: int = 200):
    """Find the threshold that maximizes F1 score."""
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_f1 = 0.0
    best_thresh = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for t in thresholds:
        predicted = (scores >= t).astype(int)
        tp = ((predicted == 1) & (labels == 1)).sum()
        fp = ((predicted == 1) & (labels == 0)).sum()
        fn = ((predicted == 0) & (labels == 1)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            best_prec = float(prec)
            best_rec = float(rec)

    return best_f1, best_thresh, best_prec, best_rec


def evaluate_variant(
    model: EmbeddingModel,
    data: list[dict],
    model_label: str,
    variant_label: str,
) -> dict:
    """Run evaluation on a single model + data variant."""
    s1 = [d["sentence1"] for d in data]
    s2 = [d["sentence2"] for d in data]
    labels = np.array([d["label"] for d in data], dtype=int)

    t0 = time.perf_counter()
    embs_a = model.embed_batch(s1)
    embs_b = model.embed_batch(s2)
    embed_time = time.perf_counter() - t0

    sims = cosine_similarities(embs_a, embs_b)

    # Metrics
    auc = compute_auc_roc(labels, sims)
    f1, threshold, precision, recall = best_f1_at_threshold(labels, sims)

    para_mask = labels == 1
    non_para_mask = labels == 0
    mean_para = float(sims[para_mask].mean())
    mean_non_para = float(sims[non_para_mask].mean())
    separation = mean_para - mean_non_para

    result = {
        "model": model_label,
        "variant": variant_label,
        "n_pairs": len(data),
        "n_paraphrase": int(para_mask.sum()),
        "n_non_paraphrase": int(non_para_mask.sum()),
        "auc_roc": round(auc, 4),
        "best_f1": round(f1, 4),
        "best_threshold": round(threshold, 4),
        "precision_at_best_f1": round(precision, 4),
        "recall_at_best_f1": round(recall, 4),
        "mean_cosine_paraphrase": round(mean_para, 4),
        "mean_cosine_non_paraphrase": round(mean_non_para, 4),
        "cosine_separation": round(separation, 4),
        "cosine_std_paraphrase": round(float(sims[para_mask].std()), 4),
        "cosine_std_non_paraphrase": round(float(sims[non_para_mask].std()), 4),
        "embed_time_s": round(embed_time, 2),
    }

    return result


def print_result(r: dict):
    """Print a single result row."""
    print(f"  AUC-ROC:           {r['auc_roc']:.4f}")
    print(f"  Best F1:           {r['best_f1']:.4f}  (threshold={r['best_threshold']:.4f})")
    print(f"    Precision:       {r['precision_at_best_f1']:.4f}")
    print(f"    Recall:          {r['recall_at_best_f1']:.4f}")
    print(f"  Mean cos (para):   {r['mean_cosine_paraphrase']:.4f} +/- {r['cosine_std_paraphrase']:.4f}")
    print(f"  Mean cos (non):    {r['mean_cosine_non_paraphrase']:.4f} +/- {r['cosine_std_non_paraphrase']:.4f}")
    print(f"  Separation:        {r['cosine_separation']:.4f}")
    print(f"  Embed time:        {r['embed_time_s']:.2f}s")


def main():
    rng = np.random.default_rng(42)

    # Load data
    print("Loading Norwegian PAWS-X data...")
    variants = {}
    for lang_code, lang_name in [("nb", "Bokmal"), ("nn", "Nynorsk")]:
        test_path = DATA_DIR / lang_code / "translated_test_2k.json"
        data = load_paws_jsonl(test_path)
        sampled = balanced_sample(data, SAMPLE_SIZE, rng)
        variants[lang_code] = sampled
        n_pos = sum(1 for d in sampled if d["label"] == 1)
        n_neg = len(sampled) - n_pos
        print(f"  {lang_name} ({lang_code}): {len(sampled)} pairs ({n_pos} para, {n_neg} non-para)")

    all_results = []

    for model_name, model_label in MODELS:
        print(f"\n{'='*70}")
        print(f"Model: {model_label} ({model_name})")
        print(f"{'='*70}")

        model = EmbeddingModel(model_name=model_name)

        for lang_code, lang_name in [("nb", "Bokmal"), ("nn", "Nynorsk")]:
            print(f"\n  --- {lang_name} ({lang_code}) ---")
            result = evaluate_variant(
                model, variants[lang_code], model_label, lang_name,
            )
            all_results.append(result)
            print_result(result)

        del model

    # Summary table
    print(f"\n{'='*110}")
    print("NORWEGIAN PAWS-X: PARAPHRASE DETECTION RESULTS")
    print(f"{'='*110}")
    header = (
        f"{'Model':<35s} {'Variant':<10s} {'AUC':>6s} {'F1':>6s} "
        f"{'Thresh':>7s} {'CosPara':>8s} {'CosNon':>8s} {'Sep':>6s} {'Time':>6s}"
    )
    print(header)
    print("-" * 110)

    for r in all_results:
        print(
            f"{r['model']:<35s} {r['variant']:<10s} "
            f"{r['auc_roc']:>6.4f} {r['best_f1']:>6.4f} "
            f"{r['best_threshold']:>7.4f} "
            f"{r['mean_cosine_paraphrase']:>8.4f} "
            f"{r['mean_cosine_non_paraphrase']:>8.4f} "
            f"{r['cosine_separation']:>6.4f} "
            f"{r['embed_time_s']:>6.2f}"
        )

    # Multilingual advantage
    print(f"\n{'='*110}")
    print("MULTILINGUAL ADVANTAGE")
    print(f"{'='*110}")
    for lang_code, lang_name in [("nb", "Bokmal"), ("nn", "Nynorsk")]:
        multi = next(r for r in all_results if r["model"] == MODELS[0][1] and r["variant"] == lang_name)
        english = next(r for r in all_results if r["model"] == MODELS[1][1] and r["variant"] == lang_name)
        auc_diff = multi["auc_roc"] - english["auc_roc"]
        f1_diff = multi["best_f1"] - english["best_f1"]
        sep_diff = multi["cosine_separation"] - english["cosine_separation"]
        print(f"  {lang_name}: AUC +{auc_diff:+.4f}, F1 +{f1_diff:+.4f}, Separation +{sep_diff:+.4f}")

    # Save JSON
    out_path = RESULTS_DIR / "eval_norwegian_results.json"
    output = {
        "benchmark": "Norwegian PAWS-X (NbAiLab/norwegian-paws-x)",
        "description": "Adversarial paraphrase detection on Norwegian text. "
                       "High lexical overlap pairs testing semantic understanding.",
        "variants": ["Bokmal (nb)", "Nynorsk (nn)"],
        "sample_size": SAMPLE_SIZE,
        "balanced": True,
        "seed": 42,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
