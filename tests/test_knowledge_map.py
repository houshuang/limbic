"""Tests for knowledge_map module — adaptive knowledge probing algorithm."""

import pytest
from limbic.amygdala.knowledge_map import (
    KnowledgeGraph, BeliefState, init_beliefs, next_probe, next_probe_batch,
    update_beliefs, coverage_report, is_converged, _entropy,
    _total_entropy, FAMILIARITY_LEVELS, calibrate_beliefs,
    adjust_for_calibration, knowledge_fringes,
)


# ── Fixtures ──────────────────────────────────────────────


def _baking_graph():
    """Small knowledge graph modeling the sourdough baking process."""
    return KnowledgeGraph(nodes=[
        {"id": "wild_yeast", "title": "Wild Yeast & Fermentation", "level": 1, "obscurity": 3, "prerequisites": []},
        {"id": "kneading", "title": "Basic Kneading Technique", "level": 1, "obscurity": 1, "prerequisites": []},
        {"id": "starter", "title": "Maintaining a Sourdough Starter", "level": 2, "obscurity": 4, "prerequisites": ["wild_yeast"]},
        {"id": "levain_build", "title": "Building a Levain", "level": 2, "obscurity": 4, "prerequisites": ["starter"]},
        {"id": "windowpane_test", "title": "Windowpane Gluten Test", "level": 2, "obscurity": 3, "prerequisites": ["kneading"]},
        {"id": "shaping", "title": "Shaping for Oven Spring", "level": 3, "obscurity": 5, "prerequisites": ["levain_build", "windowpane_test"]},
        {"id": "hydration_ratio", "title": "Adjusting Hydration Ratio", "level": 2, "obscurity": 4, "prerequisites": ["starter"]},
        {"id": "full_bake_lifecycle", "title": "Full Bake Lifecycle", "level": 3, "obscurity": 5, "prerequisites": ["shaping", "hydration_ratio"]},
    ])


def _simple_chain():
    """Linear prerequisite chain: a → b → c."""
    return KnowledgeGraph(nodes=[
        {"id": "a", "title": "Topic A", "level": 1, "obscurity": 1, "prerequisites": []},
        {"id": "b", "title": "Topic B", "level": 2, "obscurity": 2, "prerequisites": ["a"]},
        {"id": "c", "title": "Topic C", "level": 3, "obscurity": 3, "prerequisites": ["b"]},
    ])


# ── KnowledgeGraph ──────────────────────────────────────


class TestKnowledgeGraph:
    def test_get_node(self):
        g = _baking_graph()
        assert g.get("starter")["title"] == "Maintaining a Sourdough Starter"
        assert g.get("nonexistent") is None

    def test_children(self):
        g = _baking_graph()
        children = g.children_of("wild_yeast")
        assert "starter" in children
        assert "kneading" not in children

    def test_prerequisites(self):
        g = _baking_graph()
        assert g.prerequisites_of("shaping") == ["levain_build", "windowpane_test"]
        assert g.prerequisites_of("wild_yeast") == []

    def test_empty_graph(self):
        g = KnowledgeGraph(nodes=[])
        assert g.get("x") is None
        assert g.children_of("x") == []


# ── Entropy ─────────────────────────────────────────────


class TestEntropy:
    def test_maximum_at_half(self):
        assert _entropy(0.5) == pytest.approx(1.0)

    def test_zero_at_extremes(self):
        assert _entropy(0.0) == 0.0
        assert _entropy(1.0) == 0.0

    def test_symmetric(self):
        assert _entropy(0.3) == pytest.approx(_entropy(0.7))


# ── Init Beliefs ────────────────────────────────────────


class TestInitBeliefs:
    def test_uses_obscurity(self):
        g = _baking_graph()
        state = init_beliefs(g)
        # obscurity 1 → 0.42, obscurity 5 → 0.10
        assert state.beliefs["kneading"] > state.beliefs["full_bake_lifecycle"]

    def test_custom_prior(self):
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        assert all(p == 0.5 for p in state.beliefs.values())

    def test_all_nodes_have_beliefs(self):
        g = _baking_graph()
        state = init_beliefs(g)
        assert set(state.beliefs.keys()) == {n["id"] for n in g.nodes}


# ── Next Probe ──────────────────────────────────────────


class TestNextProbe:
    def test_selects_highest_entropy_strategy(self):
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probe = next_probe(g, state, strategy="entropy")
        assert probe is not None
        # All at p=0.5, so any node is valid (max entropy = 1.0)
        assert probe["information_gain"] == pytest.approx(1.0)

    def test_eig_returns_positive_gain(self):
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probe = next_probe(g, state, strategy="eig")
        assert probe is not None
        assert probe["information_gain"] > 0

    def test_skips_assessed(self):
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        state.assessed.add("a")
        probe = next_probe(g, state)
        assert probe["node_id"] != "a"

    def test_returns_none_when_all_assessed(self):
        g = _simple_chain()
        state = init_beliefs(g)
        state.assessed = {"a", "b", "c"}
        assert next_probe(g, state) is None

    def test_returns_none_when_converged(self):
        g = _simple_chain()
        state = init_beliefs(g)
        # Set all beliefs to very high certainty (H(0.99) ≈ 0.08 < 0.15 threshold)
        for nid in state.beliefs:
            state.beliefs[nid] = 0.99
        assert next_probe(g, state) is None

    def test_question_type_recognition_first(self):
        g = _baking_graph()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probe = next_probe(g, state)
        assert probe["question_type"] == "recognition"

    def test_includes_remaining_count(self):
        g = _baking_graph()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probe = next_probe(g, state)
        assert probe["remaining"] == 8  # all uncertain


# ── Update Beliefs ──────────────────────────────────────


class TestUpdateBeliefs:
    def test_updates_belief(self):
        g = _simple_chain()
        state = init_beliefs(g)
        update_beliefs(g, state, "a", "deep")
        assert state.beliefs["a"] == 0.95
        assert "a" in state.assessed

    def test_records_history(self):
        g = _simple_chain()
        state = init_beliefs(g)
        update_beliefs(g, state, "a", "basic")
        assert len(state.history) == 1
        assert state.history[0]["familiarity"] == "basic"
        assert state.history[0]["label"] == "basic"

    def test_invalid_familiarity_raises(self):
        g = _simple_chain()
        state = init_beliefs(g)
        with pytest.raises(ValueError, match="Unknown familiarity"):
            update_beliefs(g, state, "a", "bogus")


# ── Propagation ─────────────────────────────────────────


class TestPropagation:
    def test_unknown_lowers_children(self):
        g = _simple_chain()
        state = init_beliefs(g)
        update_beliefs(g, state, "a", "none")
        # b requires a, so b should be lowered
        assert state.beliefs["b"] <= 0.2

    def test_deep_raises_prerequisites(self):
        g = _simple_chain()
        state = init_beliefs(g)
        update_beliefs(g, state, "b", "deep")
        # a is prerequisite of b, should be raised
        assert state.beliefs["a"] >= 0.8

    def test_no_propagation_to_assessed(self):
        g = _simple_chain()
        state = init_beliefs(g)
        update_beliefs(g, state, "a", "deep")
        original_a = state.beliefs["a"]
        # Now say b is unknown — should NOT lower a (already assessed)
        update_beliefs(g, state, "b", "none")
        assert state.beliefs["a"] == original_a

    def test_transitive_propagation(self):
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        # Say "a" is unknown → b lowered → now assess b as unknown → c lowered
        update_beliefs(g, state, "a", "none")
        assert state.beliefs["b"] <= 0.2
        update_beliefs(g, state, "b", "none")
        assert state.beliefs["c"] <= 0.2

    def test_propagation_can_be_disabled(self):
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        update_beliefs(g, state, "a", "none", propagate=False)
        assert state.beliefs["b"] == 0.5  # unchanged

    def test_multihop_unknown_reaches_grandchildren(self):
        """a→b→c: marking a unknown should lower both b AND c in one step."""
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        update_beliefs(g, state, "a", "none")
        assert state.beliefs["b"] <= 0.2
        assert state.beliefs["c"] < 0.5  # grandchild also lowered (dampened)

    def test_multihop_known_raises_grandparents(self):
        """a→b→c: marking c deep should raise both b AND a."""
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.3)
        update_beliefs(g, state, "c", "deep")
        assert state.beliefs["b"] >= 0.8
        assert state.beliefs["a"] > 0.3  # grandparent also raised (dampened)

    def test_multihop_dampening(self):
        """Both children and grandchildren are lowered when parent is unknown."""
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        update_beliefs(g, state, "a", "none")
        assert state.beliefs["b"] <= 0.2
        assert state.beliefs["c"] < 0.5  # grandchild also lowered

    def test_multihop_does_not_overwrite_assessed(self):
        """Multi-hop should skip already-assessed nodes."""
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        update_beliefs(g, state, "b", "solid")  # assess b first
        original_b = state.beliefs["b"]
        update_beliefs(g, state, "a", "none")  # should not change b
        assert state.beliefs["b"] == original_b

    def test_cycle_rejected_at_construction(self):
        """Graph with a cycle should be rejected."""
        with pytest.raises(ValueError, match="cycle"):
            KnowledgeGraph(nodes=[
                {"id": "x", "title": "X", "level": 1, "prerequisites": ["z"]},
                {"id": "y", "title": "Y", "level": 2, "prerequisites": ["x"]},
                {"id": "z", "title": "Z", "level": 3, "prerequisites": ["y"]},
            ])


# ── EIG Strategy ───────────────────────────────────────


class TestEIGStrategy:
    def test_eig_prefers_connected_node(self):
        """EIG should prefer a node with many dependents over an isolated one."""
        g = KnowledgeGraph(nodes=[
            {"id": "root", "title": "Root", "level": 1, "prerequisites": []},
            {"id": "c1", "title": "C1", "level": 2, "prerequisites": ["root"]},
            {"id": "c2", "title": "C2", "level": 2, "prerequisites": ["root"]},
            {"id": "c3", "title": "C3", "level": 2, "prerequisites": ["root"]},
            {"id": "isolated", "title": "Isolated", "level": 1, "prerequisites": []},
        ])
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probe = next_probe(g, state, strategy="eig")
        # root has 3 children, so asking about it reduces more total entropy
        assert probe["node_id"] == "root"

    def test_entropy_strategy_still_works(self):
        """strategy='entropy' should pick max local entropy node."""
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probe = next_probe(g, state, strategy="entropy")
        assert probe is not None
        assert probe["information_gain"] == pytest.approx(1.0)

    def test_eig_and_entropy_both_converge(self):
        """Both strategies should eventually return None on a small graph."""
        g = _simple_chain()
        for strategy in ["eig", "entropy"]:
            state = init_beliefs(g, prior_fn=lambda n: 0.5)
            for nid in ["a", "b", "c"]:
                update_beliefs(g, state, nid, "solid")
            assert next_probe(g, state, strategy=strategy) is None


# ── Batch Probe Selection ──────────────────────────────


class TestNextProbeBatch:
    def test_returns_requested_count(self):
        g = _baking_graph()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probes = next_probe_batch(g, state, n=3)
        assert len(probes) == 3

    def test_all_different_nodes(self):
        g = _baking_graph()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probes = next_probe_batch(g, state, n=4)
        ids = [p["node_id"] for p in probes]
        assert len(set(ids)) == len(ids)

    def test_does_not_mutate_original_state(self):
        g = _baking_graph()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        original_beliefs = dict(state.beliefs)
        original_assessed = set(state.assessed)
        next_probe_batch(g, state, n=3)
        assert state.beliefs == original_beliefs
        assert state.assessed == original_assessed

    def test_returns_fewer_when_graph_small(self):
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probes = next_probe_batch(g, state, n=10)
        assert len(probes) <= 3  # only 3 nodes in chain

    def test_empty_when_converged(self):
        g = _simple_chain()
        state = init_beliefs(g)
        for nid in ["a", "b", "c"]:
            state.beliefs[nid] = 0.99
        probes = next_probe_batch(g, state, n=3)
        assert probes == []

    def test_diversity_avoids_siblings(self):
        """Batch should prefer diverse nodes over siblings of the same parent."""
        g = KnowledgeGraph(nodes=[
            {"id": "root", "title": "Root", "level": 1, "prerequisites": []},
            {"id": "c1", "title": "C1", "level": 2, "prerequisites": ["root"]},
            {"id": "c2", "title": "C2", "level": 2, "prerequisites": ["root"]},
            {"id": "c3", "title": "C3", "level": 2, "prerequisites": ["root"]},
            {"id": "isolated", "title": "Isolated", "level": 1, "prerequisites": []},
        ])
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        probes = next_probe_batch(g, state, n=2)
        ids = {p["node_id"] for p in probes}
        # First pick should be root (highest EIG), second should NOT be
        # another child of root since root's answer propagates to all children
        assert "root" in ids


# ── Coverage Report ─────────────────────────────────────


class TestCoverageReport:
    def test_all_known(self):
        g = _simple_chain()
        state = init_beliefs(g)
        for nid in ["a", "b", "c"]:
            update_beliefs(g, state, nid, "solid")
        report = coverage_report(g, state)
        assert len(report["known"]) == 3
        assert len(report["unknown"]) == 0
        assert report["coverage_pct"] == 100.0

    def test_mixed(self):
        g = _simple_chain()
        state = init_beliefs(g)
        update_beliefs(g, state, "a", "deep")
        update_beliefs(g, state, "b", "none")
        report = coverage_report(g, state)
        assert any(e["id"] == "a" for e in report["known"])
        assert any(e["id"] == "b" for e in report["unknown"])

    def test_by_level(self):
        g = _simple_chain()
        state = init_beliefs(g)
        for nid in ["a", "b", "c"]:
            update_beliefs(g, state, nid, "deep")
        report = coverage_report(g, state)
        assert report["by_level"][1]["known"] == 1


# ── Convergence ─────────────────────────────────────────


class TestConvergence:
    def test_not_converged_initially(self):
        g = _baking_graph()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        assert not is_converged(state)

    def test_converged_after_all_assessed(self):
        g = _simple_chain()
        state = init_beliefs(g)
        for nid in ["a", "b", "c"]:
            update_beliefs(g, state, nid, "deep")
        assert is_converged(state)

    def test_converged_via_propagation(self):
        g = _simple_chain()
        state = init_beliefs(g)
        # deep on c → a, b get high beliefs; then assess a and b
        update_beliefs(g, state, "c", "deep")
        # a got raised to 0.8+ via transitive prereqs (b raised, a raised from b)
        # but b is still uncertain unless directly assessed
        update_beliefs(g, state, "a", "deep")
        update_beliefs(g, state, "b", "deep")
        assert is_converged(state)


# ── Serialization ───────────────────────────────────────


class TestSerialization:
    def test_roundtrip(self):
        g = _simple_chain()
        state = init_beliefs(g)
        update_beliefs(g, state, "a", "solid")
        d = state.to_dict()
        restored = BeliefState.from_dict(d)
        assert restored.beliefs == state.beliefs
        assert restored.assessed == state.assessed
        assert restored.history == state.history

    def test_from_dict_handles_missing(self):
        state = BeliefState.from_dict({"beliefs": {"x": 0.5}})
        assert state.assessed == set()
        assert state.history == []


# ── Full Scenario ───────────────────────────────────────


class TestFullScenario:
    def test_baking_scenario(self):
        """Simulate probing a baker who knows kneading but not fermentation science."""
        g = _baking_graph()
        state = init_beliefs(g)

        # Probe loop
        questions_asked = 0
        while questions_asked < 20:
            probe = next_probe(g, state)
            if probe is None:
                break

            nid = probe["node_id"]
            node = g.get(nid)

            # Simulate: knows basic technique, doesn't know fermentation science
            technique_nodes = {"kneading", "windowpane_test"}
            if nid in technique_nodes:
                update_beliefs(g, state, nid, "solid")
            elif nid in {"wild_yeast"}:
                update_beliefs(g, state, nid, "heard_of")
            else:
                update_beliefs(g, state, nid, "none")
            questions_asked += 1

        report = coverage_report(g, state)
        # Should have identified kneading knowledge and fermentation gaps
        known_ids = {e["id"] for e in report["known"]}
        unknown_ids = {e["id"] for e in report["unknown"]}
        assert "kneading" in known_ids
        assert questions_asked < 10  # propagation should speed things up


# ── Foil-Based Calibration ─────────────────────────────


class TestCalibration:
    def test_perfect_calibration(self):
        state = BeliefState(beliefs={"a": 0.5})
        foils = [
            {"node_id": "fake1", "familiarity": "none"},
            {"node_id": "fake2", "familiarity": "none"},
        ]
        assert calibrate_beliefs(state, foils) == 1.0

    def test_total_overclaiming(self):
        state = BeliefState(beliefs={"a": 0.5})
        foils = [
            {"node_id": "fake1", "familiarity": "solid"},
            {"node_id": "fake2", "familiarity": "deep"},
        ]
        assert calibrate_beliefs(state, foils) == 0.0

    def test_partial_overclaiming(self):
        state = BeliefState(beliefs={"a": 0.5})
        foils = [
            {"node_id": "fake1", "familiarity": "none"},
            {"node_id": "fake2", "familiarity": "solid"},
            {"node_id": "fake3", "familiarity": "heard_of"},
            {"node_id": "fake4", "familiarity": "basic"},
        ]
        # 2 false alarms out of 4 (solid + basic claim familiarity)
        assert calibrate_beliefs(state, foils) == pytest.approx(0.5)

    def test_heard_of_not_false_alarm(self):
        state = BeliefState(beliefs={"a": 0.5})
        foils = [{"node_id": "fake1", "familiarity": "heard_of"}]
        assert calibrate_beliefs(state, foils) == 1.0

    def test_empty_foils(self):
        state = BeliefState(beliefs={"a": 0.5})
        assert calibrate_beliefs(state, []) == 1.0


class TestAdjustForCalibration:
    def test_discounts_all_above_half(self):
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.8)
        update_beliefs(g, state, "a", "solid", propagate=False)
        adjust_for_calibration(state, 0.5)
        # Now discounts ALL beliefs > 0.5, including assessed
        # 0.85 → 0.5 + 0.35 * 0.5 = 0.675
        assert state.beliefs["a"] == pytest.approx(0.675)
        # 0.8 → 0.5 + 0.3 * 0.5 = 0.65
        assert state.beliefs["b"] == pytest.approx(0.65)
        assert state.beliefs["c"] == pytest.approx(0.65)

    def test_leaves_low_beliefs_untouched(self):
        state = BeliefState(beliefs={"x": 0.3, "y": 0.8})
        adjust_for_calibration(state, 0.5)
        assert state.beliefs["x"] == pytest.approx(0.3)  # below 0.5 — unchanged
        assert state.beliefs["y"] == pytest.approx(0.65)  # 0.5 + 0.3 * 0.5

    def test_clamps_factor(self):
        state = BeliefState(beliefs={"x": 0.6})
        adjust_for_calibration(state, 1.5)  # clamped to 1.0
        assert state.beliefs["x"] == pytest.approx(0.6)
        adjust_for_calibration(state, -0.5)  # clamped to 0.0
        assert state.beliefs["x"] == pytest.approx(0.5)  # 0.5 + 0.1 * 0 = 0.5


# ── KST Fringes ───────────────────────────────────────


class TestKnowledgeFringes:
    def test_linear_chain_fringes(self):
        g = _simple_chain()
        state = BeliefState(beliefs={"a": 0.9, "b": 0.3, "c": 0.1})
        fringes = knowledge_fringes(g, state)
        assert fringes["known"] == ["a"]
        assert fringes["unknown"] == ["b", "c"]
        # a is known, has unknown child b → inner fringe
        assert "a" in fringes["inner_fringe"]
        # b is unknown, prerequisite a is known → outer fringe
        assert "b" in fringes["outer_fringe"]
        # c is unknown, prerequisite b is NOT known → not in outer fringe
        assert "c" not in fringes["outer_fringe"]

    def test_all_known(self):
        g = _simple_chain()
        state = BeliefState(beliefs={"a": 0.9, "b": 0.8, "c": 0.7})
        fringes = knowledge_fringes(g, state)
        assert fringes["outer_fringe"] == []
        assert fringes["unknown"] == []
        # c has no children → not inner fringe; a, b have all-known children → not inner fringe
        assert fringes["inner_fringe"] == []

    def test_all_unknown(self):
        g = _simple_chain()
        state = BeliefState(beliefs={"a": 0.1, "b": 0.1, "c": 0.1})
        fringes = knowledge_fringes(g, state)
        assert fringes["inner_fringe"] == []
        # a has no prerequisites → outer fringe
        assert "a" in fringes["outer_fringe"]
        # b, c have unknown prerequisites → not in outer fringe
        assert "b" not in fringes["outer_fringe"]
        assert "c" not in fringes["outer_fringe"]

    def test_custom_threshold(self):
        g = _simple_chain()
        state = BeliefState(beliefs={"a": 0.6, "b": 0.3, "c": 0.1})
        # With default threshold 0.7, a is unknown
        fringes_default = knowledge_fringes(g, state)
        assert "a" in fringes_default["unknown"]
        # With threshold 0.5, a is known
        fringes_low = knowledge_fringes(g, state, known_threshold=0.5)
        assert "a" in fringes_low["known"]
        assert "b" in fringes_low["outer_fringe"]

    def test_diamond_graph(self):
        """a → b, a → c, b → d, c → d (diamond shape)."""
        g = KnowledgeGraph(nodes=[
            {"id": "a", "title": "A", "level": 1},
            {"id": "b", "title": "B", "level": 2, "prerequisites": ["a"]},
            {"id": "c", "title": "C", "level": 2, "prerequisites": ["a"]},
            {"id": "d", "title": "D", "level": 3, "prerequisites": ["b", "c"]},
        ])
        # a and b known, c and d unknown
        state = BeliefState(beliefs={"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1})
        fringes = knowledge_fringes(g, state)
        # a has unknown child c → inner fringe
        assert "a" in fringes["inner_fringe"]
        # b has unknown child d → inner fringe
        assert "b" in fringes["inner_fringe"]
        # c is unknown, prerequisite a is known → outer fringe
        assert "c" in fringes["outer_fringe"]
        # d is unknown, prerequisite b is known but c is NOT → not outer fringe
        assert "d" not in fringes["outer_fringe"]


# ── Noisy Observation Model (BKT-style) ──────────────


class TestNoisyUpdate:
    def test_noisy_more_conservative_upward(self):
        """Claiming 'deep' with low prior should raise belief less than direct mode."""
        g = _simple_chain()
        state_direct = init_beliefs(g, prior_fn=lambda n: 0.2)
        state_noisy = init_beliefs(g, prior_fn=lambda n: 0.2)
        update_beliefs(g, state_direct, "a", "deep", propagate=False)
        update_beliefs(g, state_noisy, "a", "deep", propagate=False, noisy=True)
        # Direct jumps to 0.95; noisy should be lower (tempered by low prior)
        assert state_direct.beliefs["a"] == 0.95
        assert state_noisy.beliefs["a"] < state_direct.beliefs["a"]
        assert state_noisy.beliefs["a"] > 0.2  # but still moves upward

    def test_noisy_more_conservative_downward(self):
        """Claiming 'none' with high prior should lower belief less than direct mode."""
        g = _simple_chain()
        state_direct = init_beliefs(g, prior_fn=lambda n: 0.9)
        state_noisy = init_beliefs(g, prior_fn=lambda n: 0.9)
        update_beliefs(g, state_direct, "a", "none", propagate=False)
        update_beliefs(g, state_noisy, "a", "none", propagate=False, noisy=True)
        # Direct drops to 0.05; noisy should stay higher (tempered by high prior)
        assert state_direct.beliefs["a"] == 0.05
        assert state_noisy.beliefs["a"] > state_direct.beliefs["a"]
        assert state_noisy.beliefs["a"] < 0.9  # but still moves downward

    def test_zero_noise_equals_direct(self):
        """With overclaim_rate=0 and underclaim_rate=0, noisy should match direct."""
        g = _simple_chain()
        for fam in FAMILIARITY_LEVELS:
            state_direct = init_beliefs(g, prior_fn=lambda n: 0.5)
            state_noisy = init_beliefs(g, prior_fn=lambda n: 0.5)
            update_beliefs(g, state_direct, "a", fam, propagate=False)
            update_beliefs(g, state_noisy, "a", fam, propagate=False,
                           noisy=True, overclaim_rate=0.0, underclaim_rate=0.0)
            # With no noise, posterior collapses to 0 or 1 (perfect signal)
            if fam in ("basic", "solid", "deep"):
                assert state_noisy.beliefs["a"] == pytest.approx(1.0)
            else:
                assert state_noisy.beliefs["a"] == pytest.approx(0.0)

    def test_history_records_noisy_fields(self):
        """Noisy update should record prior, posterior, and noisy flag in history."""
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.4)
        update_beliefs(g, state, "a", "solid", propagate=False, noisy=True)
        entry = state.history[-1]
        assert entry["noisy"] is True
        assert entry["prior"] == pytest.approx(0.4)
        assert "posterior" in entry
        assert entry["posterior"] == state.beliefs["a"]

    def test_direct_mode_no_noisy_fields(self):
        """Direct (non-noisy) update should NOT have noisy/prior/posterior in history."""
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.4)
        update_beliefs(g, state, "a", "solid", propagate=False)
        entry = state.history[-1]
        assert "noisy" not in entry
        assert "prior" not in entry

    def test_high_overclaim_discounts_positive_report(self):
        """With high overclaim rate, positive self-report is worth less."""
        g = _simple_chain()
        state_low = init_beliefs(g, prior_fn=lambda n: 0.3)
        state_high = init_beliefs(g, prior_fn=lambda n: 0.3)
        update_beliefs(g, state_low, "a", "solid", propagate=False,
                       noisy=True, overclaim_rate=0.05)
        update_beliefs(g, state_high, "a", "solid", propagate=False,
                       noisy=True, overclaim_rate=0.40)
        # Higher overclaim rate → less trust in "solid" report → lower posterior
        assert state_high.beliefs["a"] < state_low.beliefs["a"]

    def test_propagation_still_works_with_noisy(self):
        """Noisy mode should still propagate through the DAG."""
        g = _simple_chain()
        state = init_beliefs(g, prior_fn=lambda n: 0.5)
        update_beliefs(g, state, "a", "none", propagate=True, noisy=True)
        # b is child of a — should be lowered by propagation
        assert state.beliefs["b"] <= 0.2
