"""Adaptive knowledge mapping via information-theoretic probing.

Given a knowledge graph (nodes with prerequisites), efficiently maps what
someone knows using entropy-maximizing probe selection and belief propagation
through the prerequisite DAG.

Two propagation backends:
  - "heuristic" (default): rule-based clamp+dampen, zero dependencies, ~0.01ms
  - "bayesian": exact inference via pgmpy DiscreteBayesianNetwork, +7-10%
    accuracy, ~1.7ms. Requires `pip install limbic[bayesian]`.

Usage:
    from limbic.amygdala.knowledge_map import KnowledgeGraph, init_beliefs, next_probe, update_beliefs

    graph = KnowledgeGraph(nodes=[
        {"id": "crdt", "title": "CRDTs", "level": 1},
        {"id": "mirror", "title": "Mirror", "level": 2, "prerequisites": ["crdt"]},
    ])
    state = init_beliefs(graph)                          # heuristic (default)
    state = init_beliefs(graph, propagator="bayesian")   # exact Bayesian inference
    probe = next_probe(graph, state)       # → {node_id: "crdt", ...}
    update_beliefs(graph, state, "crdt", "solid")  # user knows CRDTs well
    probe = next_probe(graph, state)       # → {node_id: "mirror", ...} (prerequisites met)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# Familiarity → (belief probability, knowledge label)
FAMILIARITY_LEVELS = {
    "none": (0.05, "unknown"),
    "heard_of": (0.40, "mentioned"),
    "basic": (0.70, "basic"),
    "solid": (0.85, "solid"),
    "deep": (0.95, "deep"),
}

# Dampening factor per hop for multi-hop propagation
_DAMPEN = 0.8


def _find_cycle(by_id: dict[str, dict]) -> str | None:
    """Return a node ID involved in a cycle, or None if the graph is a DAG.

    Uses DFS with white(0)/gray(1)/black(2) coloring.
    """
    color: dict[str, int] = {nid: 0 for nid in by_id}

    def _dfs(nid: str) -> str | None:
        color[nid] = 1  # gray — in progress
        for prereq in by_id[nid].get("prerequisites", []):
            if prereq not in color:
                continue
            if color[prereq] == 1:
                return prereq  # back edge → cycle
            if color[prereq] == 0:
                found = _dfs(prereq)
                if found is not None:
                    return found
        color[nid] = 2  # black — done
        return None

    for nid in by_id:
        if color[nid] == 0:
            found = _dfs(nid)
            if found is not None:
                return found
    return None


@dataclass
class KnowledgeGraph:
    """A DAG of concepts with optional prerequisites and metadata."""
    nodes: list[dict]  # {id, title, description?, level?, obscurity?, prerequisites?}

    def __post_init__(self):
        self._by_id: dict[str, dict] = {}
        for n in self.nodes:
            nid = n["id"]
            if nid in self._by_id:
                raise ValueError(f"Duplicate node ID: {nid!r}")
            self._by_id[nid] = n
        self._children: dict[str, list[str]] = {}
        for n in self.nodes:
            for prereq in n.get("prerequisites", []):
                self._children.setdefault(prereq, []).append(n["id"])
        cycle_node = _find_cycle(self._by_id)
        if cycle_node is not None:
            raise ValueError(f"Prerequisite cycle detected involving node {cycle_node!r}")

    def get(self, node_id: str) -> dict | None:
        return self._by_id.get(node_id)

    def children_of(self, node_id: str) -> list[str]:
        return self._children.get(node_id, [])

    def prerequisites_of(self, node_id: str) -> list[str]:
        return self._by_id.get(node_id, {}).get("prerequisites", [])


@dataclass
class BeliefState:
    """Mutable probability distribution over knowledge graph nodes."""
    beliefs: dict[str, float] = field(default_factory=dict)
    assessed: set[str] = field(default_factory=set)
    history: list[dict] = field(default_factory=list)
    _propagator: str = field(default="heuristic", repr=False)
    _bn_model: Any = field(default=None, repr=False)  # cached pgmpy model

    def to_dict(self) -> dict:
        return {"beliefs": self.beliefs, "assessed": list(self.assessed),
                "history": self.history, "propagator": self._propagator}

    @classmethod
    def from_dict(cls, d: dict) -> BeliefState:
        return cls(beliefs=d["beliefs"], assessed=set(d.get("assessed", [])),
                   history=d.get("history", []),
                   _propagator=d.get("propagator", "heuristic"))


def _entropy(p: float) -> float:
    """Shannon entropy of a binary distribution."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _total_entropy(state: BeliefState) -> float:
    """Sum of Shannon entropy across all nodes."""
    return sum(_entropy(p) for p in state.beliefs.values())


def init_beliefs(
    graph: KnowledgeGraph, prior_fn=None, propagator: str = "heuristic",
) -> BeliefState:
    """Initialize beliefs for all nodes.

    Args:
        graph: The knowledge graph.
        prior_fn: Custom prior function(node_dict) -> float. Default uses obscurity.
        propagator: "heuristic" (rule-based, zero deps) or "bayesian" (pgmpy exact
            inference, +7-10% accuracy, requires pgmpy installed).
    """
    beliefs = {}
    for node in graph.nodes:
        if prior_fn:
            beliefs[node["id"]] = prior_fn(node)
        else:
            obscurity = node.get("obscurity", 3)
            beliefs[node["id"]] = max(0.05, 0.5 - obscurity * 0.08)

    state = BeliefState(beliefs=beliefs, _propagator=propagator)

    if propagator == "bayesian":
        state._bn_model = _build_bayesian_model(graph)

    return state


def next_probe(
    graph: KnowledgeGraph, state: BeliefState, strategy: str = "eig",
) -> dict | None:
    """Select the next node to probe.

    Strategies:
        "eig" — Expected Information Gain: pick node that reduces total graph
                 entropy most in expectation (simulates each possible answer).
        "entropy" — Greedy max-entropy: pick the single most uncertain node.

    Returns dict with {node_id, node, information_gain, question_type} or None if converged.
    """
    # Only consider nodes we're genuinely uncertain about (belief in 0.05–0.85).
    # Floor of 0.05 matches the minimum prior from init_beliefs, ensuring all
    # initialized nodes (including high-obscurity ones) are eligible for probing.
    candidates = [
        n for n in graph.nodes
        if n["id"] not in state.assessed
        and 0.05 <= state.beliefs.get(n["id"], 0.3) <= 0.85
    ]
    if not candidates:
        return None

    if strategy == "eig":
        target, ig = _select_eig(graph, state, candidates)
    else:
        scored = [(n, _entropy(state.beliefs.get(n["id"], 0.3))) for n in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        target, ig = scored[0]

    p_knows = state.beliefs.get(target["id"], 0.3)
    q_count = len(state.history)
    if q_count < 3:
        question_type = "recognition"
    elif q_count % 4 == 0 and len(candidates) >= 2:
        question_type = "comparison"
    elif p_knows > 0.6:
        question_type = "scoping"
    else:
        question_type = "recognition"

    return {
        "node_id": target["id"],
        "node": target,
        "information_gain": ig,
        "question_type": question_type,
        "p_knows": p_knows,
        "question_number": q_count + 1,
        "remaining": sum(1 for n in candidates if 0.2 < state.beliefs.get(n["id"], 0.3) < 0.8),
    }


def _select_eig(
    graph: KnowledgeGraph, state: BeliefState, candidates: list[dict],
) -> tuple[dict, float]:
    """Pick candidate with highest Expected Information Gain.

    For each candidate, simulate every possible familiarity answer, compute
    the resulting total entropy, weight by answer likelihood, and pick the
    node with lowest expected remaining entropy.
    """
    current_entropy = _total_entropy(state)
    best_node = candidates[0]
    best_eig = -1.0

    for node in candidates:
        nid = node["id"]
        p = state.beliefs.get(nid, 0.3)
        expected_post_entropy = 0.0
        total_weight = 0.0

        for fam, (fam_belief, _) in FAMILIARITY_LEVELS.items():
            weight = max(0.0, 1.0 - abs(p - fam_belief))
            if weight < 1e-6:
                continue
            total_weight += weight
            sim_state = BeliefState(
                beliefs=dict(state.beliefs), assessed=set(state.assessed),
            )
            update_beliefs(graph, sim_state, nid, fam)
            expected_post_entropy += weight * _total_entropy(sim_state)

        if total_weight > 0:
            expected_post_entropy /= total_weight
        eig = current_entropy - expected_post_entropy
        if eig > best_eig:
            best_eig = eig
            best_node = node

    return best_node, best_eig


def update_beliefs(
    graph: KnowledgeGraph,
    state: BeliefState,
    node_id: str,
    familiarity: str,
    propagate: bool = True,
    noisy: bool = False,
    overclaim_rate: float = 0.15,
    underclaim_rate: float = 0.05,
) -> None:
    """Update beliefs based on a user's self-reported familiarity. Mutates state.

    When noisy=True, treats self-report as a noisy observation and applies
    Bayesian updating (BKT-style) instead of directly setting the belief.
    overclaim_rate = P(report_known | doesn't_know),
    underclaim_rate = P(report_unknown | actually_knows).
    """
    if familiarity not in FAMILIARITY_LEVELS:
        raise ValueError(f"Unknown familiarity '{familiarity}'. Use: {list(FAMILIARITY_LEVELS)}")

    raw_belief, label = FAMILIARITY_LEVELS[familiarity]
    history_entry = {"node_id": node_id, "familiarity": familiarity, "label": label}

    if noisy:
        p_prior = state.beliefs.get(node_id, 0.3)
        reports_known = familiarity in ("basic", "solid", "deep")
        if reports_known:
            p_report_given_knows = 1.0 - underclaim_rate
            p_report_given_not = overclaim_rate
        else:
            p_report_given_knows = underclaim_rate
            p_report_given_not = 1.0 - overclaim_rate
        p_report = p_report_given_knows * p_prior + p_report_given_not * (1.0 - p_prior)
        posterior = p_report_given_knows * p_prior / p_report if p_report > 0 else p_prior
        state.beliefs[node_id] = posterior
        history_entry.update({"noisy": True, "prior": p_prior, "posterior": posterior,
                              "belief": posterior})
    else:
        state.beliefs[node_id] = raw_belief
        history_entry["belief"] = raw_belief

    state.assessed.add(node_id)
    state.history.append(history_entry)

    if propagate:
        _propagate(graph, state, node_id, familiarity)


def _propagate(graph: KnowledgeGraph, state: BeliefState, node_id: str, familiarity: str) -> None:
    """Propagate belief update through the prerequisite DAG.

    Dispatches to heuristic (rule-based) or Bayesian (pgmpy exact inference).
    """
    if state._propagator == "bayesian" and state._bn_model is not None:
        _propagate_bayesian(graph, state)
    else:
        _propagate_heuristic(graph, state, node_id, familiarity)


def _propagate_heuristic(
    graph: KnowledgeGraph, state: BeliefState, node_id: str, familiarity: str,
) -> None:
    """Rule-based bidirectional propagation through prerequisite DAG.

    Unknown signal -> lower descendants (if I don't know a prereq, I probably
    don't know what depends on it).
    Known signal -> raise ancestors (if I know something advanced, I probably
    know its prerequisites) AND raise children whose prerequisites are now
    met (if I know all prereqs for X, I'm more likely to know X).
    Assessed nodes are never overwritten.
    """
    if familiarity in ("none", "heard_of"):
        _propagate_down(graph, state, node_id, ceiling=0.1, depth=0, visited=set())
    if familiarity in ("solid", "deep"):
        _propagate_up(graph, state, node_id, floor=0.8, depth=0, visited=set())
        _propagate_prereqs_met(graph, state, node_id, depth=0, visited=set())


def _propagate_down(
    graph: KnowledgeGraph, state: BeliefState,
    node_id: str, ceiling: float, depth: int, visited: set[str],
) -> None:
    """Recursively lower descendants. Ceiling dampens by _DAMPEN^depth."""
    for child_id in graph.children_of(node_id):
        if child_id in visited or child_id in state.assessed:
            continue
        visited.add(child_id)
        dampened = ceiling * (_DAMPEN ** depth)
        current = state.beliefs.get(child_id, 0.3)
        state.beliefs[child_id] = min(current, dampened)
        _propagate_down(graph, state, child_id, ceiling, depth + 1, visited)


def _propagate_up(
    graph: KnowledgeGraph, state: BeliefState,
    node_id: str, floor: float, depth: int, visited: set[str],
) -> None:
    """Recursively raise ancestors. Floor approaches 0.5 with _DAMPEN^depth."""
    for prereq_id in graph.prerequisites_of(node_id):
        if prereq_id in visited or prereq_id in state.assessed:
            continue
        visited.add(prereq_id)
        dampened = 1.0 - (1.0 - floor) * (_DAMPEN ** depth)
        current = state.beliefs.get(prereq_id, 0.3)
        state.beliefs[prereq_id] = max(current, dampened)
        _propagate_up(graph, state, prereq_id, floor, depth + 1, visited)


def _propagate_prereqs_met(
    graph: KnowledgeGraph, state: BeliefState,
    node_id: str, depth: int, visited: set[str],
) -> None:
    """Raise children whose prerequisites are believed known.

    Models P(child_known | prereqs_known) ≈ 0.85, dampened per hop.
    Uses minimum prerequisite belief as the signal strength, so partially
    confident prerequisites still propagate (just less strongly).
    """
    for child_id in graph.children_of(node_id):
        if child_id in visited or child_id in state.assessed:
            continue
        visited.add(child_id)
        prereqs = graph.prerequisites_of(child_id)
        prereq_beliefs = [state.beliefs.get(p, 0.3) for p in prereqs]
        min_prereq = min(prereq_beliefs) if prereq_beliefs else 0.3
        if min_prereq >= 0.5:
            # Scale floor by weakest prereq confidence: strong prereqs → ~0.85, weak → ~0.5
            base = 0.5 + 0.35 * ((min_prereq - 0.5) / 0.5)  # maps 0.5→0.5, 1.0→0.85
            floor = base * (_DAMPEN ** (depth * 0.5))  # gentler dampen: sqrt of original
            current = state.beliefs.get(child_id, 0.3)
            state.beliefs[child_id] = max(current, floor)
            _propagate_prereqs_met(graph, state, child_id, depth + 1, visited)


# ── Bayesian propagation (pgmpy) ──────────────────────────


def _build_bayesian_model(graph: KnowledgeGraph):
    """Build a pgmpy DiscreteBayesianNetwork from the knowledge graph.

    Each node is binary (0=unknown, 1=known). CPDs encode:
    - Roots: P(known) based on obscurity
    - With prerequisites: P(known | all_prereqs_known) = 0.85,
      P(known | any_prereq_unknown) = 0.15 (noisy-AND gate)
    """
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
    except ImportError:
        log.warning(
            "pgmpy not installed — falling back to heuristic propagation. "
            "Install with: pip install limbic[bayesian]"
        )
        return None

    import numpy as np

    edges = []
    for n in graph.nodes:
        for prereq in n.get("prerequisites", []):
            edges.append((prereq, n["id"]))

    if not edges:
        return None  # flat graph — no structure to exploit

    model = DiscreteBayesianNetwork(edges)

    for n in graph.nodes:
        nid = n["id"]
        prereqs = n.get("prerequisites", [])

        if not prereqs:
            obscurity = n.get("obscurity", 3)
            p_known = max(0.1, 0.7 - obscurity * 0.1)
            cpd = TabularCPD(nid, 2, [[1 - p_known], [p_known]])
        elif len(prereqs) == 1:
            cpd = TabularCPD(
                nid, 2,
                [[0.85, 0.15],   # P(unknown | parent=unknown, parent=known)
                 [0.15, 0.85]],  # P(known   | parent=unknown, parent=known)
                evidence=[prereqs[0]], evidence_card=[2],
            )
        else:
            n_parents = len(prereqs)
            n_combos = 2 ** n_parents
            values = np.zeros((2, n_combos))
            for combo in range(n_combos):
                parent_states = [(combo >> i) & 1 for i in range(n_parents)]
                all_known = all(s == 1 for s in parent_states)
                p_known = 0.85 if all_known else 0.15
                values[0, combo] = 1 - p_known
                values[1, combo] = p_known
            cpd = TabularCPD(
                nid, 2, values,
                evidence=prereqs, evidence_card=[2] * n_parents,
            )
        model.add_cpds(cpd)

    assert model.check_model(), "pgmpy model validation failed"
    return model


def _propagate_bayesian(graph: KnowledgeGraph, state: BeliefState) -> None:
    """Use pgmpy variable elimination to compute exact posteriors for all unassessed nodes."""
    from pgmpy.inference import VariableElimination

    evidence = {}
    for nid in state.assessed:
        is_known = state.beliefs.get(nid, 0.3) >= 0.5
        evidence[nid] = 1 if is_known else 0

    if not evidence:
        return

    infer = VariableElimination(state._bn_model)
    for node in graph.nodes:
        nid = node["id"]
        if nid in state.assessed:
            continue
        try:
            result = infer.query([nid], evidence=evidence)
            state.beliefs[nid] = float(result.values[1])
        except Exception:
            pass  # keep prior if inference fails


def coverage_report(graph: KnowledgeGraph, state: BeliefState) -> dict:
    """Summarize what's known, unknown, and uncertain."""
    known, unknown, uncertain = [], [], []
    for node in graph.nodes:
        nid = node["id"]
        p = state.beliefs.get(nid, 0.3)
        entry = {"id": nid, "title": node["title"], "belief": p}
        if p >= 0.7:
            known.append(entry)
        elif p <= 0.2:
            unknown.append(entry)
        else:
            uncertain.append(entry)

    by_level: dict[int, dict] = {}
    for node in graph.nodes:
        level = node.get("level", 1)
        bucket = by_level.setdefault(level, {"known": 0, "unknown": 0, "uncertain": 0, "total": 0})
        p = state.beliefs.get(node["id"], 0.3)
        bucket["total"] += 1
        if p >= 0.7: bucket["known"] += 1
        elif p <= 0.2: bucket["unknown"] += 1
        else: bucket["uncertain"] += 1

    return {
        "known": sorted(known, key=lambda x: -x["belief"]),
        "unknown": sorted(unknown, key=lambda x: x["belief"]),
        "uncertain": sorted(uncertain, key=lambda x: -_entropy(x["belief"])),
        "by_level": by_level,
        "total_nodes": len(graph.nodes),
        "assessed": len(state.assessed),
        "coverage_pct": round(100 * len(state.assessed) / max(1, len(graph.nodes)), 1),
    }


def is_converged(state: BeliefState, threshold: int = 3) -> bool:
    """True when few nodes remain in the uncertain zone (0.2 < p < 0.8)."""
    uncertain = sum(1 for p in state.beliefs.values() if 0.2 < p < 0.8)
    return uncertain <= threshold


# ── Foil-Based Calibration (Overclaiming Detection) ──────────


def calibrate_beliefs(state: BeliefState, foil_responses: list[dict]) -> float:
    """Compute calibration factor from foil concept responses.

    Uses signal detection theory (Paulhus OCQ): if a user claims familiarity
    with fabricated concepts, their self-reports are inflated.

    Args:
        state: Current belief state (unused, reserved for future per-session tracking).
        foil_responses: list of {"node_id": str, "familiarity": str} for fabricated concepts.

    Returns:
        Calibration factor 0.0-1.0 (1.0 = perfectly calibrated, lower = overclaiming).
    """
    if not foil_responses:
        return 1.0
    false_alarms = sum(
        1 for r in foil_responses if r["familiarity"] not in ("none", "heard_of")
    )
    false_alarm_rate = false_alarms / len(foil_responses)
    return 1.0 - false_alarm_rate


def adjust_for_calibration(state: BeliefState, calibration_factor: float) -> None:
    """Apply calibration factor to unassessed beliefs. Mutates state.

    Directly-assessed nodes keep their values; only propagated/prior beliefs
    are discounted. A calibration_factor of 0.7 means the user overclaims ~30%.
    """
    calibration_factor = max(0.0, min(1.0, calibration_factor))
    for node_id in state.beliefs:
        if node_id not in state.assessed:
            state.beliefs[node_id] *= calibration_factor


# ── KST Inner/Outer Fringe ──────────────────────────────────


def knowledge_fringes(
    graph: KnowledgeGraph, state: BeliefState, known_threshold: float = 0.7
) -> dict:
    """Compute KST inner and outer fringes.

    Inner fringe ("peaks"): known nodes with at least one unknown child.
    Outer fringe ("ready to learn"): unknown nodes where all prerequisites are known.

    Returns:
        {"inner_fringe": [...], "outer_fringe": [...], "known": [...], "unknown": [...]}
    """
    known = [n["id"] for n in graph.nodes if state.beliefs.get(n["id"], 0) >= known_threshold]
    unknown = [n["id"] for n in graph.nodes if state.beliefs.get(n["id"], 0) < known_threshold]
    known_set = set(known)

    inner_fringe = [
        nid for nid in known
        if any(c not in known_set for c in graph.children_of(nid))
    ]
    outer_fringe = [
        nid for nid in unknown
        if all(p in known_set for p in graph.prerequisites_of(nid))
    ]
    return {
        "inner_fringe": inner_fringe,
        "outer_fringe": outer_fringe,
        "known": known,
        "unknown": unknown,
    }
