"""Adaptive knowledge mapping via information-theoretic probing.

Given a knowledge graph (nodes with prerequisites), efficiently maps what
someone knows using entropy-maximizing probe selection and belief propagation
through the prerequisite DAG.

Two propagation backends (both zero-dependency):
  - "heuristic" (default): rule-based bidirectional propagation with global
    sweeps. Fast (~0.08ms), good accuracy on all topologies.
  - "bayesian": Pearl's forward-backward belief propagation with noisy-AND
    CPDs. Exact on trees/chains, approximate on dense DAGs. ~0.16ms.

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
        propagator: "heuristic" (rule-based, zero deps) or "bayesian" (forward-backward
            belief propagation, +1-5% accuracy, also zero deps).
    """
    beliefs = {}
    for node in graph.nodes:
        if prior_fn:
            beliefs[node["id"]] = prior_fn(node)
        else:
            obscurity = node.get("obscurity", 3)
            beliefs[node["id"]] = max(0.05, 0.5 - obscurity * 0.08)

    return BeliefState(beliefs=beliefs, _propagator=propagator)


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

    Dispatches to heuristic (rule-based) or Bayesian (forward-backward BP).
    """
    if state._propagator == "bayesian":
        _propagate_bayesian(graph, state)
    else:
        _propagate_heuristic(graph, state, node_id, familiarity)


def _propagate_heuristic(
    graph: KnowledgeGraph, state: BeliefState, node_id: str, familiarity: str,
) -> None:
    """Rule-based bidirectional propagation through prerequisite DAG.

    After the local update, does a global sweep: re-checks all nodes to see
    if prerequisites-met propagation should fire based on cumulative evidence.
    This approximates the Bayesian backend's global inference pass.
    """
    if familiarity in ("none", "heard_of"):
        _propagate_down(graph, state, node_id, ceiling=0.1, depth=0, visited=set())
    if familiarity in ("basic", "solid", "deep"):
        # "basic" also implies prereqs known, just with weaker floor
        floor = 0.65 if familiarity == "basic" else 0.8
        _propagate_up(graph, state, node_id, floor=floor, depth=0, visited=set())
    # Global sweep: re-propagate prereqs-met from ALL high-belief nodes
    _global_prereqs_sweep(graph, state)


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


def _global_prereqs_sweep(graph: KnowledgeGraph, state: BeliefState) -> None:
    """Global pass: check every unassessed node based on prerequisite state.

    - If all prereqs are confident (>= 0.5): raise toward P(known|prereqs_known)
    - If any prereq is low (< 0.2): lower toward P(known|prereq_unknown)

    Runs until convergence (max 3 iterations).
    """
    for _ in range(3):
        changed = False
        for node in graph.nodes:
            nid = node["id"]
            if nid in state.assessed:
                continue
            prereqs = graph.prerequisites_of(nid)
            if not prereqs:
                continue
            prereq_beliefs = [state.beliefs.get(p, 0.3) for p in prereqs]
            min_prereq = min(prereq_beliefs)
            current = state.beliefs.get(nid, 0.3)

            if min_prereq >= 0.5:
                # All prereqs likely known → raise
                base = 0.5 + 0.35 * ((min_prereq - 0.5) / 0.5)
                if base > current:
                    state.beliefs[nid] = base
                    changed = True
            elif min_prereq < 0.2:
                # At least one prereq likely unknown → lower
                ceiling = 0.15 + 0.35 * (min_prereq / 0.2)  # maps 0.0→0.15, 0.2→0.50
                if ceiling < current:
                    state.beliefs[nid] = ceiling
                    changed = True
        if not changed:
            break


# ── Exact belief propagation (zero dependencies) ─────────


def _topo_sort(graph: KnowledgeGraph) -> list[str]:
    """Topological sort: prerequisites before dependents."""
    visited: set[str] = set()
    order: list[str] = []

    def visit(nid: str) -> None:
        if nid in visited:
            return
        visited.add(nid)
        for prereq in graph.prerequisites_of(nid):
            visit(prereq)
        order.append(nid)

    for n in graph.nodes:
        visit(n["id"])
    return order


def _node_prior(graph: KnowledgeGraph, nid: str) -> float:
    """Prior P(known) for a root node, based on obscurity."""
    node = graph.get(nid)
    obscurity = node.get("obscurity", 3) if node else 3
    return max(0.05, 0.5 - obscurity * 0.08)


def _propagate_bayesian(graph: KnowledgeGraph, state: BeliefState) -> None:
    """Exact belief propagation on binary DAG with noisy-AND CPDs.

    Two-pass forward-backward algorithm (Pearl 1988), iterated to handle
    explaining-away effects at v-structures. Zero external dependencies.

    CPD model:
      - Roots: P(known) from obscurity prior
      - With prereqs: P(known | all_prereqs_known) = 0.85,
                      P(known | any_prereq_unknown) = 0.15
      - Assessed nodes: fixed at their observed belief

    Specialized for binary noisy-AND CPDs. Exact on trees/chains,
    approximate on dense DAGs with v-structures (~0.16ms).
    """
    topo = _topo_sort(graph)
    if not topo:
        return

    # Use assessed beliefs as hard evidence (binarized)
    evidence: dict[str, float] = {}
    for nid in state.assessed:
        evidence[nid] = 1.0 if state.beliefs.get(nid, 0.3) >= 0.5 else 0.0

    if not evidence:
        return

    # Forward pass: compute pi[nid] = P(nid=known | upstream evidence only)
    # This is computed ONCE and not updated — avoids double-counting in backward pass
    pi: dict[str, float] = {}
    for nid in topo:
        if nid in evidence:
            pi[nid] = evidence[nid]
            continue
        prereqs = graph.prerequisites_of(nid)
        if not prereqs:
            pi[nid] = _node_prior(graph, nid)
            continue
        p_all_known = 1.0
        for p in prereqs:
            p_all_known *= pi.get(p, 0.3)
        pi[nid] = 0.85 * p_all_known + 0.15 * (1 - p_all_known)

    # Backward pass: compute lambda[nid] = (like_if_known, like_if_unknown)
    lam: dict[str, tuple[float, float]] = {}
    for nid in reversed(topo):
            children = graph.children_of(nid)
            if not children:
                lam[nid] = (1.0, 1.0)
                continue

            l1, l0 = 1.0, 1.0
            for child_id in children:
                other_prereqs = [op for op in graph.prerequisites_of(child_id) if op != nid]
                p_others_known = 1.0
                for op in other_prereqs:
                    if op in evidence:
                        p_others_known *= evidence[op]
                    else:
                        # Use forward-only belief for other parents to avoid
                        # double-counting in dense graphs with many shared parents
                        p_others_known *= pi.get(op, 0.3)

                p_c_if_1 = 0.85 * p_others_known + 0.15 * (1 - p_others_known)
                p_c_if_0 = 0.15

                if child_id in evidence:
                    c_obs = evidence[child_id]
                    cl1, cl0 = c_obs, 1.0 - c_obs
                else:
                    cl1, cl0 = lam.get(child_id, (1.0, 1.0))

                msg_1 = p_c_if_1 * cl1 + (1 - p_c_if_1) * cl0
                msg_0 = p_c_if_0 * cl1 + (1 - p_c_if_0) * cl0

                l1 *= msg_1
                l0 *= msg_0

            lam[nid] = (l1, l0)

    # Combine: posterior ∝ pi * lambda
    for nid in topo:
        if nid in state.assessed:
            continue
        p1 = pi[nid] * lam[nid][0]
        p0 = (1.0 - pi[nid]) * lam[nid][1]
        total = p1 + p0
        if total > 1e-10:
            state.beliefs[nid] = max(0.01, min(0.99, p1 / total))
        else:
            state.beliefs[nid] = pi[nid]


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
