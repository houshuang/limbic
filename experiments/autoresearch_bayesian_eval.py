#!/usr/bin/env python3
"""Evaluation harness for autoresearch grid sweep on Bayesian CPD parameters.

Measures Bayesian propagation accuracy across topologies and K values.
The parameters being optimized are module-level constants in knowledge_map.py:
  _CPD_HIGH: P(known | all prereqs known)
  _CPD_LOW:  P(known | any prereq unknown)
  _EVIDENCE_THRESHOLD: belief threshold for binarizing assessed evidence

Usage:
    python3 experiments/autoresearch_bayesian_eval.py
    # Output: SCORE=0.6951
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from limbic.amygdala.knowledge_map import (
    KnowledgeGraph, init_beliefs, update_beliefs,
)


def make_wide_tree(branches=5, depth=4):
    nodes = [{"id": "root", "title": "Root", "level": 1, "obscurity": 2, "prerequisites": []}]
    for b in range(branches):
        parent = "root"
        for d in range(depth):
            nid = f"b{b}_d{d}"
            nodes.append({"id": nid, "title": f"B{b}D{d}", "level": d+2, "obscurity": 3, "prerequisites": [parent]})
            parent = nid
    return KnowledgeGraph(nodes=nodes)

def make_dense_dag(n=20, n_edges=40, seed=42):
    rng = random.Random(seed)
    nodes = [{"id": f"d{i:02d}", "title": f"D{i}", "level": (i//5)+1, "obscurity": 3, "prerequisites": []} for i in range(n)]
    possible = [(i,j) for i in range(n) for j in range(i+1,n)]
    rng.shuffle(possible)
    for src, dst in possible[:n_edges]:
        nodes[dst]["prerequisites"].append(f"d{src:02d}")
    return KnowledgeGraph(nodes=nodes)

def make_chain(n=20):
    nodes = []
    for i in range(n):
        prereqs = [f"c{i-1:02d}"] if i > 0 else []
        nodes.append({"id": f"c{i:02d}", "title": f"C{i}", "level": i+1, "obscurity": 3, "prerequisites": prereqs})
    return KnowledgeGraph(nodes=nodes)

def make_diamond(n_layers=5, width=3):
    nodes = [{"id": "root", "title": "Root", "level": 1, "obscurity": 2, "prerequisites": []}]
    prev_layer = ["root"]
    for layer in range(n_layers):
        curr_layer = []
        for w in range(width):
            nid = f"l{layer}_w{w}"
            nodes.append({"id": nid, "title": f"L{layer}W{w}", "level": layer+2, "obscurity": 3, "prerequisites": list(prev_layer)})
            curr_layer.append(nid)
        prev_layer = curr_layer
    return KnowledgeGraph(nodes=nodes)

def generate_truth(graph, seed=42):
    rng = random.Random(seed)
    visited, order = set(), []
    def visit(nid):
        if nid in visited: return
        visited.add(nid)
        for p in graph.prerequisites_of(nid): visit(p)
        order.append(nid)
    for n in graph.nodes: visit(n["id"])
    truth = {}
    for nid in order:
        prereqs = graph.prerequisites_of(nid)
        all_known = all(truth.get(p) in ("basic","solid","deep") for p in prereqs) if prereqs else True
        p = 0.75 if all_known else 0.15
        truth[nid] = rng.choice(["basic","solid","deep"]) if rng.random() < p else rng.choice(["none","heard_of"])
    return truth

def accuracy(graph, truth, assessed):
    state = init_beliefs(graph, propagator="bayesian")
    for nid in assessed:
        update_beliefs(graph, state, nid, truth[nid])
    correct = total = 0
    for nid, fam in truth.items():
        if nid in assessed: continue
        total += 1
        if (state.beliefs.get(nid, 0.3) >= 0.5) == (fam in ("basic","solid","deep")): correct += 1
    return correct / max(total, 1)


TOPOLOGIES = {
    "wide_tree": make_wide_tree,
    "dense_dag": lambda: make_dense_dag(20, 40),
    "chain_20": make_chain,
    "diamond_5x3": lambda: make_diamond(5, 3),
}
K_VALUES = [3, 5, 8]
N_TRIALS = 30

rng = random.Random(42)

total = n_evals = 0
for topo_name, make_fn in TOPOLOGIES.items():
    graph = make_fn()
    for K in K_VALUES:
        for trial in range(N_TRIALS):
            truth = generate_truth(graph, seed=trial*7+13)
            assessed = rng.sample(list(truth.keys()), min(K, len(truth)))
            total += accuracy(graph, truth, assessed)
            n_evals += 1

score = total / n_evals
print(f"SCORE={score:.4f}")
