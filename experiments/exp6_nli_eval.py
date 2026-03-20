"""Experiment 6: NLI Cross-Encoder Evaluation on SICK Dataset

Tests whether amygdala's nli_classify can improve novelty detection by
distinguishing entailment from contradiction — something cosine similarity
fundamentally cannot do.

Dataset: SICK (Sentences Involving Compositional Knowledge), test split.
~4900 sentence pairs with human labels: entailment, neutral, contradiction.

Models tested:
  1. cross-encoder/nli-deberta-v3-base (small, fast)
  2. MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (large, accurate)

Compared against cosine similarity from EmbeddingModel (paraphrase-multilingual-MiniLM-L12-v2).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np


from amygdala import EmbeddingModel

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# SICK label mapping: 0=entailment, 1=neutral, 2=contradiction
SICK_ID_TO_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

# Model configs: name -> label order (from model config id2label)
NLI_MODELS = {
    "cross-encoder/nli-deberta-v3-base": {
        "short_name": "nli-deberta-base",
        # id2label: {0: contradiction, 1: entailment, 2: neutral}
        "label_order": ["contradiction", "entailment", "neutral"],
    },
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": {
        "short_name": "deberta-large-mnli",
        # id2label: {0: entailment, 1: neutral, 2: contradiction}
        "label_order": ["entailment", "neutral", "contradiction"],
    },
}


def load_sick_test():
    """Load SICK test split. Returns list of dicts with text1, text2, label."""
    from datasets import load_dataset
    ds = load_dataset("yangwang825/sick")
    test = ds["test"]
    pairs = []
    for i in range(len(test)):
        ex = test[i]
        pairs.append({
            "text1": ex["text1"],
            "text2": ex["text2"],
            "label": SICK_ID_TO_LABEL[ex["label"]],
        })
    return pairs


def compute_cosine_similarities(pairs, model):
    """Embed all sentences and compute pairwise cosine similarity."""
    texts_a = [p["text1"] for p in pairs]
    texts_b = [p["text2"] for p in pairs]

    print(f"  Embedding {len(texts_a)} sentence A's...")
    t0 = time.perf_counter()
    embs_a = model.embed_batch(texts_a)
    t1 = time.perf_counter()
    print(f"  Embedding {len(texts_b)} sentence B's...")
    embs_b = model.embed_batch(texts_b)
    t2 = time.perf_counter()
    print(f"  Embedding time: {t1-t0:.1f}s + {t2-t1:.1f}s = {t2-t0:.1f}s")

    # Cosine similarity (embeddings are already L2-normalized)
    cosines = np.sum(embs_a * embs_b, axis=1)
    return cosines.tolist()


def run_nli_model(pairs, model_name, label_order, batch_size=64):
    """Run NLI cross-encoder on all pairs. Returns list of predicted labels."""
    from sentence_transformers import CrossEncoder

    print(f"  Loading {model_name}...")
    ce = CrossEncoder(model_name)

    input_pairs = [(p["text1"], p["text2"]) for p in pairs]
    n = len(input_pairs)

    print(f"  Running inference on {n} pairs (batch_size={batch_size})...")
    t0 = time.perf_counter()
    all_scores = ce.predict(input_pairs, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    print(f"  Inference time: {elapsed:.1f}s ({elapsed/n*1000:.1f}ms/pair)")

    results = []
    for scores in all_scores:
        label_scores = {l: float(s) for l, s in zip(label_order, scores)}
        best = max(label_scores, key=label_scores.get)
        results.append({"label": best, "scores": label_scores})

    return results, elapsed


def compute_metrics(gold_labels, pred_labels, class_names):
    """Compute accuracy, per-class precision/recall/F1."""
    n = len(gold_labels)
    correct = sum(g == p for g, p in zip(gold_labels, pred_labels))
    accuracy = correct / n

    metrics = {"accuracy": accuracy, "n": n, "correct": correct}
    per_class = {}

    for cls in class_names:
        tp = sum(g == cls and p == cls for g, p in zip(gold_labels, pred_labels))
        fp = sum(g != cls and p == cls for g, p in zip(gold_labels, pred_labels))
        fn = sum(g == cls and p != cls for g, p in zip(gold_labels, pred_labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
        }

    metrics["per_class"] = per_class

    # Macro F1
    metrics["macro_f1"] = np.mean([v["f1"] for v in per_class.values()])

    return metrics


def cosine_distribution_by_label(cosines, gold_labels, class_names):
    """Compute cosine similarity stats grouped by human label."""
    stats = {}
    for cls in class_names:
        cls_cosines = [c for c, l in zip(cosines, gold_labels) if l == cls]
        if cls_cosines:
            stats[cls] = {
                "mean": float(np.mean(cls_cosines)),
                "std": float(np.std(cls_cosines)),
                "median": float(np.median(cls_cosines)),
                "min": float(np.min(cls_cosines)),
                "max": float(np.max(cls_cosines)),
                "n": len(cls_cosines),
            }
    return stats


def cosine_novelty_analysis(cosines, gold_labels):
    """Analyze how well 'novelty = 1 - cosine' separates the three classes.

    For novelty detection:
    - entailment -> should have LOW novelty (high cosine) -> KNOWN
    - contradiction -> should have HIGH novelty (low cosine) -> NEW
    - neutral -> should be somewhere in between -> EXTENDS

    The key question: can cosine distinguish contradiction from entailment?
    """
    ent_cosines = [c for c, l in zip(cosines, gold_labels) if l == "entailment"]
    con_cosines = [c for c, l in zip(cosines, gold_labels) if l == "contradiction"]
    neu_cosines = [c for c, l in zip(cosines, gold_labels) if l == "neutral"]

    # Overlap analysis: what fraction of contradictions have higher cosine than
    # mean entailment? (These would be misclassified by cosine-only novelty)
    ent_mean = np.mean(ent_cosines)
    con_above_ent_mean = sum(c >= ent_mean for c in con_cosines) / len(con_cosines)

    # AUC-like: can cosine separate entailment from contradiction?
    # For each entailment-contradiction pair, does entailment have higher cosine?
    n_correct = 0
    n_total = 0
    # Sample to keep runtime reasonable
    rng = np.random.RandomState(42)
    ent_sample = rng.choice(ent_cosines, min(500, len(ent_cosines)), replace=False)
    con_sample = rng.choice(con_cosines, min(500, len(con_cosines)), replace=False)
    for e in ent_sample:
        for c in con_sample:
            n_total += 1
            if e > c:
                n_correct += 1
            elif e == c:
                n_correct += 0.5
    cosine_auc = n_correct / n_total if n_total > 0 else 0.5

    return {
        "entailment_mean_cosine": float(np.mean(ent_cosines)),
        "contradiction_mean_cosine": float(np.mean(con_cosines)),
        "neutral_mean_cosine": float(np.mean(neu_cosines)),
        "cosine_gap_ent_vs_con": float(np.mean(ent_cosines) - np.mean(con_cosines)),
        "contradiction_above_ent_mean_frac": float(con_above_ent_mean),
        "cosine_auc_ent_vs_con": float(cosine_auc),
    }


def nli_informed_novelty_analysis(nli_results, cosines, gold_labels):
    """Compare cosine-only novelty vs NLI-informed novelty.

    Cosine-only: novelty = 1 - cosine
    NLI-informed: if NLI says entailment, novelty = 0 (known);
                  if contradiction, novelty = 1 (new);
                  if neutral, novelty = 1 - cosine (fallback to cosine)

    Evaluate: for pairs where cosine is misleading (high cosine but contradiction),
    does NLI correctly identify the relationship?
    """
    # Find "cosine failures": contradictions with high cosine (top quartile of all cosines)
    all_cos = np.array(cosines)
    q75 = np.percentile(all_cos, 75)

    high_cos_contradictions = []
    nli_catches = 0
    for i, (cos, gold) in enumerate(zip(cosines, gold_labels)):
        if gold == "contradiction" and cos >= q75:
            high_cos_contradictions.append(i)
            if nli_results[i]["label"] == "contradiction":
                nli_catches += 1

    # Also: entailments with low cosine
    q25 = np.percentile(all_cos, 25)
    low_cos_entailments = []
    nli_catches_ent = 0
    for i, (cos, gold) in enumerate(zip(cosines, gold_labels)):
        if gold == "entailment" and cos <= q25:
            low_cos_entailments.append(i)
            if nli_results[i]["label"] == "entailment":
                nli_catches_ent += 1

    return {
        "high_cosine_contradictions": {
            "count": len(high_cos_contradictions),
            "cosine_threshold": float(q75),
            "nli_correctly_identified": nli_catches,
            "nli_accuracy_on_these": nli_catches / len(high_cos_contradictions) if high_cos_contradictions else 0,
        },
        "low_cosine_entailments": {
            "count": len(low_cos_entailments),
            "cosine_threshold": float(q25),
            "nli_correctly_identified": nli_catches_ent,
            "nli_accuracy_on_these": nli_catches_ent / len(low_cos_entailments) if low_cos_entailments else 0,
        },
    }


def build_confusion_matrix(gold_labels, pred_labels, class_names):
    """Build confusion matrix as dict of dicts."""
    matrix = {g: {p: 0 for p in class_names} for g in class_names}
    for g, p in zip(gold_labels, pred_labels):
        matrix[g][p] += 1
    return matrix


def print_results_table(all_model_results, cosine_dist, cosine_analysis):
    """Print formatted results."""
    class_names = ["entailment", "neutral", "contradiction"]

    print("\n" + "=" * 100)
    print("EXPERIMENT 6: NLI CROSS-ENCODER EVALUATION ON SICK DATASET")
    print("=" * 100)

    # Cosine similarity by label
    print("\n--- Cosine Similarity by Human Label ---")
    print(f"{'Label':<16s} {'Mean':>7s} {'Std':>7s} {'Median':>7s} {'Min':>7s} {'Max':>7s} {'N':>6s}")
    print("-" * 60)
    for cls in class_names:
        s = cosine_dist[cls]
        print(f"{cls:<16s} {s['mean']:>7.4f} {s['std']:>7.4f} {s['median']:>7.4f} {s['min']:>7.4f} {s['max']:>7.4f} {s['n']:>6d}")

    print(f"\nCosine gap (entailment - contradiction): {cosine_analysis['cosine_gap_ent_vs_con']:.4f}")
    print(f"Cosine AUC (ent vs con): {cosine_analysis['cosine_auc_ent_vs_con']:.4f}")
    print(f"Contradictions with cosine >= entailment mean: {cosine_analysis['contradiction_above_ent_mean_frac']:.1%}")

    # Per-model results
    for model_result in all_model_results:
        name = model_result["model_name"]
        metrics = model_result["metrics"]
        cm = model_result["confusion_matrix"]

        print(f"\n{'='*80}")
        print(f"Model: {name}")
        print(f"{'='*80}")
        print(f"Overall accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['n']})")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Inference time: {model_result['inference_time']:.1f}s ({model_result['ms_per_pair']:.1f}ms/pair)")

        print(f"\n  {'Class':<16s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Support':>8s}")
        print(f"  {'-'*46}")
        for cls in class_names:
            pc = metrics["per_class"][cls]
            print(f"  {cls:<16s} {pc['precision']:>7.4f} {pc['recall']:>7.4f} {pc['f1']:>7.4f} {pc['support']:>8d}")

        print(f"\n  Confusion Matrix (rows=gold, cols=predicted):")
        print(f"  {'':>16s} {'entailment':>12s} {'neutral':>12s} {'contradiction':>14s}")
        for g in class_names:
            row = [cm[g][p] for p in class_names]
            print(f"  {g:>16s} {row[0]:>12d} {row[1]:>12d} {row[2]:>14d}")

        # NLI-informed novelty analysis
        nia = model_result["nli_informed_novelty"]
        hcc = nia["high_cosine_contradictions"]
        lce = nia["low_cosine_entailments"]
        print(f"\n  NLI vs Cosine failure cases:")
        print(f"    High-cosine contradictions (cosine >= {hcc['cosine_threshold']:.3f}): {hcc['count']} pairs")
        print(f"      NLI correctly identified: {hcc['nli_correctly_identified']}/{hcc['count']} ({hcc['nli_accuracy_on_these']:.1%})")
        print(f"    Low-cosine entailments (cosine <= {lce['cosine_threshold']:.3f}): {lce['count']} pairs")
        print(f"      NLI correctly identified: {lce['nli_correctly_identified']}/{lce['count']} ({lce['nli_accuracy_on_these']:.1%})")

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY: Can NLI improve novelty detection?")
    print(f"{'='*80}")
    print(f"\nCosine similarity CANNOT distinguish contradictions from entailments well:")
    print(f"  Entailment mean cosine:    {cosine_analysis['entailment_mean_cosine']:.4f}")
    print(f"  Contradiction mean cosine: {cosine_analysis['contradiction_mean_cosine']:.4f}")
    print(f"  Gap:                       {cosine_analysis['cosine_gap_ent_vs_con']:.4f}")
    print(f"  AUC (ent vs con):          {cosine_analysis['cosine_auc_ent_vs_con']:.4f}")

    print(f"\nNLI cross-encoders CAN distinguish them:")
    for model_result in all_model_results:
        name = model_result["model_name"]
        m = model_result["metrics"]
        print(f"  {name}:")
        print(f"    3-class accuracy: {m['accuracy']:.1%}")
        print(f"    Contradiction F1: {m['per_class']['contradiction']['f1']:.4f}")
        print(f"    Entailment F1:    {m['per_class']['entailment']['f1']:.4f}")


def main():
    class_names = ["entailment", "neutral", "contradiction"]

    # Load SICK
    print("Loading SICK test split...")
    pairs = load_sick_test()
    gold_labels = [p["label"] for p in pairs]
    print(f"Loaded {len(pairs)} pairs")
    from collections import Counter
    print(f"Label distribution: {dict(Counter(gold_labels))}")

    # Cosine similarities
    print("\nComputing cosine similarities...")
    model = EmbeddingModel()
    cosines = compute_cosine_similarities(pairs, model)
    del model  # free memory

    cosine_dist = cosine_distribution_by_label(cosines, gold_labels, class_names)
    cosine_analysis = cosine_novelty_analysis(cosines, gold_labels)

    # NLI models
    all_model_results = []
    for model_name, config in NLI_MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {config['short_name']}")
        print(f"{'='*60}")

        nli_results, inference_time = run_nli_model(
            pairs, model_name, config["label_order"], batch_size=64
        )
        pred_labels = [r["label"] for r in nli_results]

        metrics = compute_metrics(gold_labels, pred_labels, class_names)
        cm = build_confusion_matrix(gold_labels, pred_labels, class_names)
        nia = nli_informed_novelty_analysis(nli_results, cosines, gold_labels)

        all_model_results.append({
            "model_name": config["short_name"],
            "model_full_name": model_name,
            "metrics": metrics,
            "confusion_matrix": cm,
            "inference_time": inference_time,
            "ms_per_pair": inference_time / len(pairs) * 1000,
            "nli_informed_novelty": nia,
        })

    # Print results
    print_results_table(all_model_results, cosine_dist, cosine_analysis)

    # Save JSON
    output = {
        "experiment": "exp6_nli_eval",
        "dataset": "SICK test split (yangwang825/sick)",
        "n_pairs": len(pairs),
        "label_distribution": dict(Counter(gold_labels)),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "cosine_distribution_by_label": cosine_dist,
        "cosine_novelty_analysis": cosine_analysis,
        "nli_model_results": all_model_results,
    }

    out_path = RESULTS_DIR / "exp6_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
