"""Calibration and validation utilities for LLM judges.

Provides functions to measure agreement between two sets of labels (e.g.,
LLM vs human gold labels, or two passes of the same LLM) using Cohen's
kappa and per-label precision/recall/F1.

Usage:
    from limbic.amygdala.calibrate import validate_llm_judge, intra_rater_reliability

    result = validate_llm_judge(gold_labels, llm_labels)
    print(result["kappa"], result["recommendation"])

    consistency = intra_rater_reliability(pass1, pass2)
    print(consistency["kappa"], consistency["quality"])
"""

from collections import Counter


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two raters on the same items.

    Handles any number of categories. Returns float in [-1, 1].
    Returns 0.0 when both observed and expected agreement are 1.0
    (all items in a single category — kappa is undefined, but 0.0
    is the conventional fallback since there is no variation to measure).

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Label lists must have the same length, got {len(labels_a)} and {len(labels_b)}"
        )
    if len(labels_a) == 0:
        raise ValueError("Label lists must not be empty")

    n = len(labels_a)
    all_labels = sorted(set(labels_a) | set(labels_b))

    # Observed agreement: fraction of items where raters agree
    p_observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement by chance: for each label, probability both raters
    # independently choose it, summed over all labels
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    p_expected = sum((count_a[label] / n) * (count_b[label] / n) for label in all_labels)

    if p_expected == 1.0:
        # Both raters put everything in the same category — kappa undefined
        return 0.0

    return (p_observed - p_expected) / (1.0 - p_expected)


def _per_label_metrics(
    gold_labels: list[str], pred_labels: list[str]
) -> dict[str, dict[str, float]]:
    """Compute per-label precision, recall, and F1 from two label lists.

    Returns dict mapping each label to {"precision", "recall", "f1"}.
    """
    all_labels = sorted(set(gold_labels) | set(pred_labels))
    confusion = Counter(zip(gold_labels, pred_labels))
    result = {}
    for label in all_labels:
        tp = confusion.get((label, label), 0)
        # Predicted as this label (sum over all gold labels)
        predicted_pos = sum(confusion.get((g, label), 0) for g in all_labels)
        # Actually this label (sum over all predicted labels)
        actual_pos = sum(confusion.get((label, p), 0) for p in all_labels)

        precision = tp / predicted_pos if predicted_pos > 0 else 0.0
        recall = tp / actual_pos if actual_pos > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[label] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def validate_llm_judge(
    gold_labels: list[str],
    llm_labels: list[str],
) -> dict:
    """Bootstrap Validation Protocol: compare LLM judge to human gold labels.

    Args:
        gold_labels: Human-annotated ground truth labels.
        llm_labels: Labels produced by the LLM judge.

    Returns:
        Dict with:
            - kappa: Cohen's kappa (chance-corrected agreement)
            - agreement: raw percent agreement (0.0 to 1.0)
            - n: number of items
            - recommendation: "trustworthy" (kappa>0.7), "moderate" (0.5-0.7),
              "unreliable" (<0.5)
            - confusion: dict mapping (gold, llm) string tuples to counts
            - per_label: dict mapping each label to {precision, recall, f1}

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(gold_labels) != len(llm_labels):
        raise ValueError(
            f"Label lists must have the same length, got {len(gold_labels)} and {len(llm_labels)}"
        )
    if len(gold_labels) == 0:
        raise ValueError("Label lists must not be empty")

    n = len(gold_labels)
    kappa = cohens_kappa(gold_labels, llm_labels)
    agreement = sum(g == l for g, l in zip(gold_labels, llm_labels)) / n

    if kappa > 0.7:
        recommendation = "trustworthy"
    elif kappa >= 0.5:
        recommendation = "moderate"
    else:
        recommendation = "unreliable"

    confusion = dict(Counter(zip(gold_labels, llm_labels)))
    # Convert tuple keys to string representation for JSON compatibility
    confusion_str = {str(k): v for k, v in confusion.items()}

    per_label = _per_label_metrics(gold_labels, llm_labels)

    return {
        "kappa": kappa,
        "agreement": agreement,
        "n": n,
        "recommendation": recommendation,
        "confusion": confusion_str,
        "per_label": per_label,
    }


def intra_rater_reliability(
    first_pass: list[str],
    second_pass: list[str],
) -> dict:
    """Measure self-consistency between two passes of rating the same items.

    Useful for checking whether an LLM judge produces stable outputs
    when run twice on the same inputs (e.g., with temperature > 0).

    Args:
        first_pass: Labels from the first rating pass.
        second_pass: Labels from the second rating pass.

    Returns:
        Dict with:
            - kappa: Cohen's kappa
            - agreement: raw percent agreement (0.0 to 1.0)
            - n: number of items
            - quality: "excellent" (>0.8), "acceptable" (0.6-0.8),
              "concerning" (<0.6)
            - disagreements: list of indices where passes disagree

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(first_pass) != len(second_pass):
        raise ValueError(
            f"Label lists must have the same length, got {len(first_pass)} and {len(second_pass)}"
        )
    if len(first_pass) == 0:
        raise ValueError("Label lists must not be empty")

    n = len(first_pass)
    kappa = cohens_kappa(first_pass, second_pass)
    agreement = sum(a == b for a, b in zip(first_pass, second_pass)) / n

    if kappa > 0.8:
        quality = "excellent"
    elif kappa >= 0.6:
        quality = "acceptable"
    else:
        quality = "concerning"

    disagreements = [i for i, (a, b) in enumerate(zip(first_pass, second_pass)) if a != b]

    return {
        "kappa": kappa,
        "agreement": agreement,
        "n": n,
        "quality": quality,
        "disagreements": disagreements,
    }
