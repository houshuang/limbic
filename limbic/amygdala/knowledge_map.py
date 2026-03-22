"""Adaptive knowledge mapping via information-theoretic probing.

Given a knowledge graph (nodes with prerequisites), efficiently maps what
someone knows using Shannon entropy maximization and Bayesian belief propagation.
No storage, no LLM, no UI — pure algorithm. Caller provides the graph,
gets back which node to probe next, feeds answers back.

Usage:
    from amygdala.knowledge_map import KnowledgeGraph, init_beliefs, next_probe, update_beliefs

    graph = KnowledgeGraph(nodes=[
        {"id": "crdt", "title": "CRDTs", "level": 1},
        {"id": "mirror", "title": "Mirror", "level": 2, "prerequisites": ["crdt"]},
    ])
    state = init_beliefs(graph)
    probe = next_probe(graph, state)       # → {node_id: "crdt", ...}
    update_beliefs(graph, state, "crdt", "solid")  # user knows CRDTs well
    probe = next_probe(graph, state)       # → {node_id: "mirror", ...} (prerequisites met)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


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


@dataclass
class KnowledgeGraph:
    """A DAG of concepts with optional prerequisites and metadata."""
    nodes: list[dict]  # {id, title, description?, level?, obscurity?, prerequisites?}

    def __post_init__(self):
        self._by_id = {n["id"]: n for n in self.nodes}
        self._children: dict[str, list[str]] = {}
        for n in self.nodes:
            for prereq in n.get("prerequisites", []):
                self._children.setdefault(prereq, []).append(n["id"])

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

    def to_dict(self) -> dict:
        return {"beliefs": self.beliefs, "assessed": list(self.assessed), "history": self.history}

    @classmethod
    def from_dict(cls, d: dict) -> BeliefState:
        return cls(beliefs=d["beliefs"], assessed=set(d.get("assessed", [])),
                   history=d.get("history", []))


def _entropy(p: float) -> float:
    """Shannon entropy of a binary distribution."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _total_entropy(state: BeliefState) -> float:
    """Sum of Shannon entropy across all nodes."""
    return sum(_entropy(p) for p in state.beliefs.values())


def init_beliefs(graph: KnowledgeGraph, prior_fn=None) -> BeliefState:
    """Initialize beliefs for all nodes. Default prior uses obscurity if available."""
    beliefs = {}
    for node in graph.nodes:
        if prior_fn:
            beliefs[node["id"]] = prior_fn(node)
        else:
            obscurity = node.get("obscurity", 3)
            beliefs[node["id"]] = max(0.05, 0.5 - obscurity * 0.08)
    return BeliefState(beliefs=beliefs)


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
    # Only consider nodes we're genuinely uncertain about (belief in 0.15–0.85).
    # Nodes outside this range are already confident enough — asking wastes a question.
    candidates = [
        n for n in graph.nodes
        if n["id"] not in state.assessed
        and 0.15 <= state.beliefs.get(n["id"], 0.3) <= 0.85
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
    """Multi-hop belief propagation through prerequisite DAG.

    Unknown signal -> recursively lower all descendants with dampening per hop.
    Known signal -> recursively raise all ancestors with dampening per hop.
    Assessed nodes are never overwritten. Visited set prevents cycles.
    """
    if familiarity in ("none", "heard_of"):
        _propagate_down(graph, state, node_id, ceiling=0.1, depth=0, visited=set())
    if familiarity in ("solid", "deep"):
        _propagate_up(graph, state, node_id, floor=0.8, depth=0, visited=set())


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
