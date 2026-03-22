"""Tests for amygdala.calibrate — LLM judge validation and inter-rater agreement."""

import pytest

from limbic.amygdala.calibrate import cohens_kappa, validate_llm_judge, intra_rater_reliability


# ── cohens_kappa ──────────────────────────────────────────────────────────


class TestCohensKappa:
    def test_perfect_agreement(self):
        labels = ["a", "b", "c", "a", "b"]
        assert cohens_kappa(labels, labels) == pytest.approx(1.0)

    def test_perfect_agreement_binary(self):
        a = ["yes", "no", "yes", "no", "yes"]
        assert cohens_kappa(a, a) == pytest.approx(1.0)

    def test_known_hand_computed_example(self):
        """Classic textbook example.

        Two raters, 50 items, binary (pos/neg):
            Rater B
                    pos  neg
        Rater A pos  20   5    = 25
                neg  10  15    = 25

        p_o = (20+15)/50 = 0.70
        p_e = (25/50)*(30/50) + (25/50)*(20/50) = 0.30 + 0.20 = 0.50
        kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
        """
        rater_a = ["pos"] * 25 + ["neg"] * 25
        rater_b = ["pos"] * 20 + ["neg"] * 5 + ["pos"] * 10 + ["neg"] * 15
        assert cohens_kappa(rater_a, rater_b) == pytest.approx(0.4, abs=1e-10)

    def test_complete_disagreement_binary_balanced(self):
        """Complete disagreement on balanced binary labels → kappa = -1."""
        a = ["yes", "no", "yes", "no"]
        b = ["no", "yes", "no", "yes"]
        assert cohens_kappa(a, b) == pytest.approx(-1.0)

    def test_random_agreement_near_zero(self):
        """Large random labeling should produce kappa near 0.

        We construct labels where observed agreement is close to chance.
        With 1000 items and two balanced raters choosing independently,
        p_observed ≈ 0.50, p_expected ≈ 0.50, so kappa ≈ 0.
        """
        import random
        rng = random.Random(42)
        n = 10000
        labels = ["pos", "neg"]
        a = [rng.choice(labels) for _ in range(n)]
        b = [rng.choice(labels) for _ in range(n)]
        k = cohens_kappa(a, b)
        assert abs(k) < 0.05  # should be close to 0

    def test_single_label_returns_zero(self):
        """If all items have the same label, kappa is undefined (p_e=1.0).
        Convention: return 0.0."""
        a = ["x", "x", "x"]
        b = ["x", "x", "x"]
        assert cohens_kappa(a, b) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cohens_kappa([], [])

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cohens_kappa(["a", "b"], ["a"])

    def test_three_categories(self):
        """Three-category example with known kappa.

                Rater B
                    A    B    C
        Rater A  A  10   2    0   = 12
                 B   1   8    1   = 10
                 C   0   3    5   = 8
                    11  13    6    N=30

        p_o = (10 + 8 + 5) / 30 = 23/30
        p_e = (12*11 + 10*13 + 8*6) / 30^2 = (132 + 130 + 48) / 900 = 310/900
        kappa = (23/30 - 310/900) / (1 - 310/900)
              = (690/900 - 310/900) / (590/900)
              = 380/590
        """
        rater_a = (["A"] * 10 + ["B"] * 2 + ["C"] * 0 +
                   ["A"] * 1 + ["B"] * 8 + ["C"] * 1 +
                   ["A"] * 0 + ["B"] * 3 + ["C"] * 5)
        rater_b = (["A"] * 10 + ["A"] * 2 + ["A"] * 0 +
                   ["B"] * 1 + ["B"] * 8 + ["B"] * 1 +
                   ["C"] * 0 + ["C"] * 3 + ["C"] * 5)
        # Wait, that's wrong. rater_b should follow the column layout.
        # Row-by-row: each item's labels are (rater_a_label, rater_b_label).
        # Row A: 10 items where both say A, 2 items where A says A and B says B, 0 items A→C
        # Row B: 1 item B→A, 8 items B→B, 1 item B→C
        # Row C: 0 items C→A, 3 items C→B, 5 items C→C
        gold = (["A"] * 10 + ["A"] * 2 + ["A"] * 0 +
                ["B"] * 1 + ["B"] * 8 + ["B"] * 1 +
                ["C"] * 0 + ["C"] * 3 + ["C"] * 5)
        pred = (["A"] * 10 + ["B"] * 2 + ["C"] * 0 +
                ["A"] * 1 + ["B"] * 8 + ["C"] * 1 +
                ["A"] * 0 + ["B"] * 3 + ["C"] * 5)
        expected_kappa = 380 / 590
        assert cohens_kappa(gold, pred) == pytest.approx(expected_kappa, abs=1e-10)

    def test_two_items(self):
        """Minimal valid input."""
        assert cohens_kappa(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_asymmetric_marginals(self):
        """Raters with very different base rates."""
        # Rater A: 8 pos, 2 neg. Rater B: 3 pos, 7 neg.
        # Agreement on: items 0,1,2 (both pos) and items 8,9 (both neg) = 5
        a = ["pos"] * 8 + ["neg"] * 2
        b = ["pos"] * 3 + ["neg"] * 5 + ["neg"] * 2
        k = cohens_kappa(a, b)
        # p_o = 5/10 = 0.50
        # p_e = (8/10)*(3/10) + (2/10)*(7/10) = 0.24 + 0.14 = 0.38
        # kappa = (0.50 - 0.38) / (1 - 0.38) = 0.12/0.62
        assert k == pytest.approx(0.12 / 0.62, abs=1e-10)


# ── validate_llm_judge ───────────────────────────────────────────────────


class TestValidateLlmJudge:
    def test_perfect_agreement_trustworthy(self):
        gold = ["relevant", "irrelevant", "relevant", "irrelevant", "relevant"]
        llm = ["relevant", "irrelevant", "relevant", "irrelevant", "relevant"]
        result = validate_llm_judge(gold, llm)
        assert result["kappa"] == pytest.approx(1.0)
        assert result["agreement"] == pytest.approx(1.0)
        assert result["n"] == 5
        assert result["recommendation"] == "trustworthy"

    def test_moderate_agreement(self):
        # Construct labels that give kappa around 0.5-0.7
        # 10 items, 7 agree, 3 disagree, balanced base rates
        gold = ["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
        llm = ["pos", "neg", "pos", "neg", "pos", "pos", "neg", "neg", "pos", "neg"]
        result = validate_llm_judge(gold, llm)
        assert result["recommendation"] == "moderate"
        assert 0.5 <= result["kappa"] <= 0.7

    def test_unreliable(self):
        # Mostly disagreement
        gold = ["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
        llm = ["neg", "pos", "neg", "pos", "neg", "pos", "pos", "neg"]
        result = validate_llm_judge(gold, llm)
        assert result["recommendation"] == "unreliable"
        assert result["kappa"] < 0.5

    def test_confusion_matrix(self):
        gold = ["a", "a", "b", "b"]
        llm = ["a", "b", "a", "b"]
        result = validate_llm_judge(gold, llm)
        # Confusion: (a,a)=1, (a,b)=1, (b,a)=1, (b,b)=1
        assert result["confusion"]["('a', 'a')"] == 1
        assert result["confusion"]["('a', 'b')"] == 1
        assert result["confusion"]["('b', 'a')"] == 1
        assert result["confusion"]["('b', 'b')"] == 1

    def test_per_label_metrics(self):
        gold = ["pos", "pos", "pos", "neg", "neg"]
        llm = ["pos", "pos", "neg", "neg", "neg"]
        result = validate_llm_judge(gold, llm)
        # pos: TP=2, FP=0, FN=1 → precision=1.0, recall=2/3, f1=0.8
        assert result["per_label"]["pos"]["precision"] == pytest.approx(1.0)
        assert result["per_label"]["pos"]["recall"] == pytest.approx(2.0 / 3.0)
        assert result["per_label"]["pos"]["f1"] == pytest.approx(0.8)
        # neg: TP=2, FP=1, FN=0 → precision=2/3, recall=1.0, f1=0.8
        assert result["per_label"]["neg"]["precision"] == pytest.approx(2.0 / 3.0)
        assert result["per_label"]["neg"]["recall"] == pytest.approx(1.0)
        assert result["per_label"]["neg"]["f1"] == pytest.approx(0.8)

    def test_per_label_zero_support(self):
        """Label present in predictions but not gold should have recall=0."""
        gold = ["a", "a", "a"]
        llm = ["a", "a", "b"]
        result = validate_llm_judge(gold, llm)
        assert result["per_label"]["b"]["recall"] == 0.0
        assert result["per_label"]["b"]["precision"] == pytest.approx(0.0)

    def test_multiclass(self):
        gold = ["high", "medium", "low", "high", "medium", "low"]
        llm = ["high", "medium", "medium", "high", "low", "low"]
        result = validate_llm_judge(gold, llm)
        assert result["n"] == 6
        assert 0.0 < result["agreement"] < 1.0
        assert "high" in result["per_label"]
        assert "medium" in result["per_label"]
        assert "low" in result["per_label"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_llm_judge([], [])

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            validate_llm_judge(["a"], ["a", "b"])

    def test_agreement_range(self):
        gold = ["a", "b", "c", "a"]
        llm = ["a", "c", "c", "b"]
        result = validate_llm_judge(gold, llm)
        assert 0.0 <= result["agreement"] <= 1.0


# ── intra_rater_reliability ──────────────────────────────────────────────


class TestIntraRaterReliability:
    def test_perfect_consistency(self):
        labels = ["support", "oppose", "neutral", "support", "oppose"]
        result = intra_rater_reliability(labels, labels)
        assert result["kappa"] == pytest.approx(1.0)
        assert result["agreement"] == pytest.approx(1.0)
        assert result["quality"] == "excellent"
        assert result["disagreements"] == []
        assert result["n"] == 5

    def test_acceptable_consistency(self):
        # 10 items, 8 agree → reasonable kappa for balanced labels
        pass1 = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]
        pass2 = ["a", "b", "a", "b", "a", "b", "a", "a", "b", "b"]
        result = intra_rater_reliability(pass1, pass2)
        assert result["quality"] == "acceptable"
        assert 0.6 <= result["kappa"] <= 0.8

    def test_concerning_consistency(self):
        pass1 = ["a", "b", "a", "b", "a", "b"]
        pass2 = ["b", "a", "b", "a", "a", "b"]
        result = intra_rater_reliability(pass1, pass2)
        assert result["quality"] == "concerning"
        assert result["kappa"] < 0.6

    def test_disagreement_indices(self):
        pass1 = ["a", "b", "c", "d", "e"]
        pass2 = ["a", "X", "c", "X", "e"]
        result = intra_rater_reliability(pass1, pass2)
        assert result["disagreements"] == [1, 3]

    def test_all_disagree(self):
        pass1 = ["a", "b", "a", "b"]
        pass2 = ["b", "a", "b", "a"]
        result = intra_rater_reliability(pass1, pass2)
        assert result["agreement"] == pytest.approx(0.0)
        assert result["disagreements"] == [0, 1, 2, 3]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            intra_rater_reliability([], [])

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            intra_rater_reliability(["a", "b"], ["a"])

    def test_single_item_agree(self):
        result = intra_rater_reliability(["x"], ["x"])
        assert result["agreement"] == pytest.approx(1.0)
        assert result["disagreements"] == []
        assert result["n"] == 1

    def test_single_item_disagree(self):
        result = intra_rater_reliability(["x"], ["y"])
        assert result["agreement"] == pytest.approx(0.0)
        assert result["disagreements"] == [0]


# ── Edge cases across all functions ──────────────────────────────────────


class TestEdgeCases:
    def test_single_label_kappa_zero(self):
        """When there's no variation, kappa is 0 by convention."""
        assert cohens_kappa(["same"] * 10, ["same"] * 10) == 0.0

    def test_validate_single_label(self):
        result = validate_llm_judge(["x"] * 5, ["x"] * 5)
        assert result["kappa"] == 0.0
        assert result["agreement"] == pytest.approx(1.0)

    def test_many_categories(self):
        """Works with many distinct labels."""
        labels = [str(i) for i in range(100)]
        k = cohens_kappa(labels, labels)
        assert k == pytest.approx(1.0)

    def test_kappa_symmetry(self):
        """kappa(a, b) == kappa(b, a)."""
        a = ["pos", "neg", "pos", "neg", "pos"]
        b = ["pos", "pos", "neg", "neg", "pos"]
        assert cohens_kappa(a, b) == pytest.approx(cohens_kappa(b, a))

    def test_validate_returns_all_keys(self):
        result = validate_llm_judge(["a", "b"], ["a", "b"])
        assert set(result.keys()) == {"kappa", "agreement", "n", "recommendation", "confusion", "per_label"}

    def test_intra_rater_returns_all_keys(self):
        result = intra_rater_reliability(["a", "b"], ["a", "b"])
        assert set(result.keys()) == {"kappa", "agreement", "n", "quality", "disagreements"}
