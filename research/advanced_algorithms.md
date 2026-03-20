# Advanced Algorithms for Adaptive Knowledge Assessment

Research survey for improving `amygdala.knowledge_map`. Date: 2026-03-19.

## Current Implementation Summary

The existing `knowledge_map.py` (~200 lines) uses:
- **Question selection**: Greedy Shannon entropy maximization (pick node with P closest to 0.5)
- **Belief update**: Direct mapping from 5-level familiarity scale to probability, then simple rule-based propagation through prerequisite DAG (doesn't know X → clamp children to 0.2; knows X well → raise prerequisites to 0.8)
- **Convergence**: Stop when few nodes remain in uncertain zone (0.2 < p < 0.8)
- **Self-report only**: No verification of claimed knowledge

This works but has clear limitations: propagation is ad-hoc (not principled Bayesian), question selection is myopic (one-step greedy), no calibration for overconfident self-reports, and the belief model is a single scalar per node rather than a proper joint distribution.

---

## 1. Knowledge Space Theory (KST)

### What It Is

Knowledge Space Theory, developed by Doignon & Falmagne (1985), models knowledge as a combinatorial structure. A *knowledge space* is a pair (Q, K) where Q is a set of items (problems/concepts) and K is a family of subsets of Q (the *knowledge states*), closed under union. Each knowledge state represents a plausible combination of items a person could know.

**Key concepts:**
- **Surmise relation**: A quasi-order on Q where q ≤ r means "knowing r implies knowing q" (r surmises q). This is the prerequisite structure — exactly what our DAG encodes.
- **Knowledge state**: A subset K ⊆ Q that is "downward closed" under the surmise relation — if you know an item, you know all its prerequisites.
- **Inner fringe** of state K: Items in K whose removal yields another valid state. These represent the "peaks" of current competence — what you most recently learned.
- **Outer fringe** of state K: Items NOT in K whose addition yields another valid state. These are what you're "ready to learn next."

### How ALEKS Implements It

ALEKS maintains a probability distribution over all feasible knowledge states. The assessment algorithm:

1. Start with uniform (or prior-informed) distribution over states.
2. Select a question that "splits" the distribution — choose item q such that the sum of probabilities of states containing q is close to 0.5.
3. After response, update via Bayes' rule using a BLIM (Basic Local Independence Model) with per-item lucky-guess (eta) and careless-error (beta) parameters.
4. Repeat until one state has probability above threshold.

For a domain like Algebra 1 (~350 concepts), there are millions of feasible states, but the Markov assessment procedure converges in ~25-30 questions.

The BLIM update for item q, response r (correct/incorrect):
```
P(K | response) ∝ P(response | K) · P(K)
where P(correct | K) = (1 - beta_q) if q ∈ K, else eta_q
      P(incorrect | K) = beta_q if q ∈ K, else (1 - eta_q)
```

### How It Differs From Our Approach

| Aspect | Our implementation | KST/ALEKS |
|--------|-------------------|-----------|
| State representation | Independent P per node | Joint distribution over all feasible knowledge states |
| Question selection | Max entropy on single node | Split the state distribution (effectively max mutual information) |
| Update | Ad-hoc rule-based propagation | Principled Bayesian update on state distribution |
| Error model | None (trusts self-report) | Lucky-guess and careless-error parameters per item |

### Worth the Complexity?

**For our use case: partially.** Full KST with millions of states is overkill for ~50-70 node curricula with self-report (not test items). But two ideas are directly useful:

1. **The fringe concept** — "what are you ready to learn?" and "what are the peaks of your knowledge?" are exactly the kind of actionable output the application layer's curriculum system needs. We already have the DAG structure to compute fringes cheaply.
2. **The half-split question selection** — selecting questions that split the probability mass in half is equivalent to maximizing mutual information, which is provably more efficient than our greedy max-entropy approach when beliefs are correlated (which they are, via prerequisites).

### Open-Source Implementations

- **R**: `kst` package on CRAN (deterministic assessment only), `kstMatrix`, `DAKS`, `pks`
- **Python**: `ishwar6/KST-Learning-Path` on GitHub (basic KST algorithms), `milansegedinac/kst` (Python port)
- Neither has the probabilistic ALEKS-style assessment. Would need to implement the BLIM update ourselves.

---

## 2. Item Response Theory (IRT) / Computerized Adaptive Testing (CAT)

### Core Concepts

IRT models the probability of a correct response as a function of person ability (theta) and item parameters. The 3PL model:

```
P(correct | theta, a, b, c) = c + (1 - c) / (1 + exp(-a(theta - b)))
```

Where: a = discrimination, b = difficulty, c = guessing parameter.

**Fisher Information** for an item at ability level theta:
```
I(theta) = a^2 · (P - c)^2 · Q / ((1 - c)^2 · P)
where P = P(correct|theta), Q = 1 - P
```

The standard CAT algorithm: select the item maximizing Fisher Information at current ability estimate, administer it, update ability via MLE or EAP (Expected a Posteriori), repeat.

### Modern Advances

**Cognitive Diagnostic Models (CDMs)** extend IRT to multiple latent skills:
- **Q-matrix**: Binary J x K matrix defining which skills each item requires
- **DINA model**: Conjunctive — must master ALL required skills (non-compensatory)
- **DINO model**: Disjunctive — mastering ANY required skill suffices (compensatory)
- **G-DINA**: Generalized model subsuming both

CDMs with **attribute hierarchies** directly incorporate prerequisite structures — "mastery of any child attribute should be no less than the mastery of its parent attribute." This is exactly our DAG constraint.

**CD-CAT** (Cognitive Diagnosis CAT) selects items to maximize Shannon Entropy or KL-divergence between posterior attribute profiles, subject to content constraints.

### How It Differs From Our Approach

IRT/CDMs assume actual test items with right/wrong responses. Our system uses self-report familiarity. The mathematical machinery is powerful but designed for a different signal. That said:

- **CDM attribute profiles** are essentially what we're estimating — a binary or continuous vector of skill mastery across curriculum nodes
- **Q-matrix with hierarchy constraints** is isomorphic to our prerequisite DAG
- The **EAP estimation** for updating ability after each response is more principled than our direct-assignment approach

### Worth the Complexity?

**Selective adoption.** Full IRT is designed for item banks with calibrated difficulty parameters — we don't have that. But CDM concepts map well:

- Our curriculum nodes ARE the attributes in a CDM
- Our prerequisite DAG IS the attribute hierarchy
- If we ever add verification questions (not just self-report), CDM-style updates would be the right framework

**Concrete improvement**: Replace our ad-hoc propagation with EAP-style posterior updates using the DAG structure as a prior constraint. This is ~30 lines of code for a significant accuracy improvement.

### Open-Source Implementations

- **Python**: `catsim` (CAT simulator with IRT, pip-installable), `EduCAT` (CDM-based CAT from USTC, github.com/bigdata-ustc/EduCAT), `EduCDM` (cognitive diagnosis models)
- **R**: `mirtCAT` (multidimensional IRT CAT), `catR` (CAT routines)

---

## 3. Bayesian Knowledge Tracing (BKT) / Deep Knowledge Tracing (DKT)

### BKT Model

BKT (Corbett & Anderson, 1995) is a Hidden Markov Model with four parameters per skill:

- **P(L0)**: Prior probability of knowing the skill initially (~0.10)
- **P(T)**: Probability of learning the skill after each opportunity (~0.30)
- **P(G)**: Probability of guessing correctly when skill is unknown (~0.10)
- **P(S)**: Probability of slipping (incorrect despite knowing) (~0.03)

Update equations after observing a correct response:
```
P(Lt | correct) = P(Lt)(1 - P(S)) / [P(Lt)(1 - P(S)) + (1 - P(Lt))P(G)]
P(Lt+1) = P(Lt | obs) + (1 - P(Lt | obs)) · P(T)
```

After incorrect:
```
P(Lt | incorrect) = P(Lt)P(S) / [P(Lt)P(S) + (1 - P(Lt))(1 - P(G))]
P(Lt+1) = P(Lt | obs) + (1 - P(Lt | obs)) · P(T)
```

### Deep Knowledge Tracing (DKT)

DKT (Piech et al., 2015) replaces the HMM with an LSTM that takes sequences of (question, response) pairs and predicts future performance. More expressive but less interpretable. Recent variants:

- **AKT** (Attention-based KT): Uses attention mechanisms to weight historical interactions
- **DKT2**: Uses xLSTM architecture with IRT-inspired output layer for interpretability
- **Mamba4KT**: State-space model alternative to transformers

### How They Differ From Our Approach

BKT/DKT track knowledge **over time** through observed performance on tasks. Our system does a **one-shot assessment** via self-report. Key differences:

| Aspect | Our implementation | BKT | DKT |
|--------|-------------------|-----|-----|
| Signal | Self-report familiarity | Binary correct/incorrect | Binary correct/incorrect |
| Temporal | Single assessment session | Tracks learning over time | Tracks learning over time |
| Model | Independent beliefs + propagation | HMM per skill | LSTM over interaction sequences |
| Error model | None | Guess + slip parameters | Learned implicitly |

### Worth the Complexity?

**Not for the assessment module itself, but potentially for the larger the application layer system.** BKT's core insight — modeling knowledge as a latent state with noisy observations — could improve our system if we add actual verification (see Section 6). The guess/slip parameters are directly analogous to overclaiming/underclaiming rates.

**Specific insight worth borrowing**: BKT's update equations are a clean Bayesian update that accounts for noise in observations. If we treat self-report as a noisy observation (not ground truth), we could use BKT-style updates with "guess" = probability of overclaiming and "slip" = probability of underclaiming. This would be a ~20-line improvement to `update_beliefs`.

### Open-Source Implementations

- **Python**: `pyBKT` (pip install pyBKT, well-maintained, includes EM fitting and extensions)
- **PyTorch**: Multiple DKT implementations on GitHub (`shinyflight/Deep-Knowledge-Tracing`)
- **Comprehensive**: `bigdata-ustc/EduKTM` (knowledge tracing models in PyTorch)

---

## 4. Active Learning / Optimal Experiment Design

### Beyond Greedy Entropy

Our current approach selects the node with maximum Shannon entropy — the node where we're most uncertain. This is the simplest active learning strategy. Better alternatives exist:

**Mutual Information / Expected Information Gain (EIG):**
```
EIG(q) = H(beliefs) - E_response[H(beliefs | response_q)]
```

Instead of looking at uncertainty about ONE node, EIG measures how much asking about node q would reduce uncertainty about ALL nodes. This naturally accounts for prerequisite structure — asking about a node that has many dependents will have higher EIG because the response propagates information.

**Difference from our approach**: Our greedy entropy picks the most uncertain single node. EIG picks the node whose answer would tell us most about the entire graph. For a well-connected prerequisite DAG, these can differ significantly.

**Expected Model Change:**
Select the question that would cause the largest change to the current belief state, regardless of which answer is given:
```
EMC(q) = E_response[||beliefs_new - beliefs_old||]
```

This naturally selects "bottleneck" nodes in the prerequisite graph whose answers cascade through many dependents.

**Multi-Step Lookahead:**
Instead of greedily picking the single best next question, plan 2-3 questions ahead. This is formalized as a POMDP (Partially Observable Markov Decision Process):
- **State**: True knowledge state (hidden)
- **Observation**: Self-reported familiarity
- **Action**: Which node to probe next
- **Reward**: Information gained (or negative entropy remaining)

True multi-step optimization is computationally intractable for large state spaces, but approximations exist:
- **Two-step lookahead**: Enumerate all possible responses to question 1, then for each, find the best question 2. O(n^2 * |responses|) — feasible for 50-70 nodes with 5 response levels.
- **Rollout policies**: Use the greedy policy as a base, improve it by one step of lookahead.
- **Monte Carlo Tree Search**: Sample possible trajectories.

**Batch Selection:**
Select k questions at once (useful for reducing rounds). Approaches:
- **BADGE** (Diverse Gradient Embedding): Combine uncertainty with diversity via k-means++ in gradient space
- **Suggestive Annotation**: Filter top-K uncertain, then maximize representativeness
- Simple: Select top-k by EIG, then de-duplicate by removing questions whose information overlaps

### How This Differs From Our Approach

Our greedy max-entropy approach ignores: (a) how information propagates through the graph, (b) what we'd ask next, and (c) the possibility of asking multiple questions per round.

### Worth the Complexity?

**EIG: definitely yes** — this is the single highest-impact improvement. For a 70-node curriculum, computing EIG requires simulating the Bayesian update for each candidate node under each possible response. With 5 familiarity levels and 70 nodes, that's 350 forward passes — trivially fast. Implementation: ~30-40 lines.

**Two-step lookahead: probably not worth it.** EIG already captures most of the benefit of planning ahead, especially with good propagation. The improvement over EIG diminishes rapidly with graph connectivity.

**Batch selection: worth adding** if we want to support "assess these 5 nodes at once" in the UI. Simple top-k-with-diversity would be ~15 lines.

---

## 5. LLM-Based Adaptive Assessment

### Recent Work (2024-2026)

**GENCAT (Generative Computerized Adaptive Testing, Feb 2026):**
A framework using LLMs for both response modeling and question selection. Key ideas:
- **GIRT model** (Generative IRT): Represents student knowledge as a latent vector z_i, projects to per-skill mastery via MLP+sigmoid. Uses "soft prompting" — interpolates between TRUE/FALSE token embeddings based on mastery level.
- **Three selection algorithms**: Uncertainty-based (P closest to 0.5), diversity-based (semantic diversity of sampled responses via CodeBERT embeddings), information-based (Fisher information).
- Trained via SFT then DPO for knowledge-response alignment.
- Applied to programming assessment with open-ended code responses.
- Limitation: Requires domain-specific training data. Evaluated only on programming.

**Socratic LLM Tutoring:**
LLMs implementing Socratic dialogue: ask probing questions, analyze responses for misconceptions, adapt difficulty. Recent evaluation frameworks (2025) operationalize three phases:
- **Perception**: Detecting student knowledge state from responses
- **Orchestration**: Choosing appropriate pedagogical strategy
- **Elicitation**: Formulating effective probing questions

**Knowledge Tracing in Dialogue (LAK 2025):**
Viewing tutor-student dialogues as formative assessment sequences. Each tutor turn is classified as a "move" (clarification, error identification, feedback, Socratic probing). Student responses are traced for knowledge signals.

**TreeInstruct:**
Estimates student knowledge to dynamically construct a question tree, adapting depth and breadth based on demonstrated understanding.

### How This Differs From Our Approach

We use a fixed familiarity scale (none/heard_of/basic/solid/deep). LLM-based approaches can:
- Generate domain-specific probing questions ("Explain how X relates to Y")
- Analyze free-form responses for depth of understanding
- Detect misconceptions from how someone describes a concept
- Adapt question phrasing to the assessed level

### Worth the Complexity?

**For knowledge_map.py: no.** The module is designed as a pure algorithm library — no LLM dependency, no latency, deterministic. LLM-based assessment belongs in the application layer (the application layer's curriculum system), not the shared library.

**However**: The module could expose hooks that an LLM-based system would use:
- `next_probe()` already returns `question_type` — an LLM could use this to generate appropriate questions
- A new `update_beliefs_from_score(node_id, score: float)` could accept LLM-graded assessment scores instead of fixed familiarity levels
- The graph structure could inform LLM prompt generation ("Ask about X in the context of prerequisite Y")

**Concrete recommendation**: Add a `next_probe_batch(n)` function and a continuous-score update path. Keep the core algorithm LLM-free.

---

## 6. Calibration and Overconfidence Detection

### The Problem

Self-report is unreliable. The Dunning-Kruger effect shows that the least competent individuals are most likely to overestimate their knowledge. In domain assessment:
- Beginners who've heard a few terms may claim "solid" understanding
- Experts may underestimate (claim "basic" for what is actually deep knowledge)
- Some domains trigger more overclaiming than others

### Techniques

**Overclaiming Questionnaire (OCQ) — Paulhus Lab, UBC:**
Mix real concepts with fabricated "foil" items. If someone claims familiarity with non-existent concepts, they're overclaiming. Scored using Signal Detection Theory:
- **Hit rate**: Proportion of real items claimed as familiar
- **False alarm rate**: Proportion of foils claimed as familiar
- **d'** (sensitivity): Ability to discriminate real from fake
- **c** (criterion/bias): Overall tendency to claim familiarity

Applied to knowledge assessment: sprinkle 3-5 plausible-sounding but non-existent concepts into a curriculum assessment. If someone claims to know "the Aristophanes conjecture on Sicilian urban planning" (fabricated), their other self-reports should be discounted.

**Consistency Checks:**
- Ask about the same concept in different ways at different points
- Ask about a concept and its prerequisite — claiming to know X but not its prerequisite flags inconsistency
- Ask about a concept at two different granularity levels

**Calibration Scoring:**
After self-report, optionally present a verification question. Compute discrepancy between claimed knowledge and demonstrated knowledge. Use this to adjust a per-user calibration factor:
```
adjusted_belief = raw_belief * calibration_factor
where calibration_factor = demonstrated_accuracy / claimed_accuracy
```

**Follow-up Verification:**
For high-confidence claims, ask a specific question: "You said you know X well. Can you explain how X relates to Y?" Grade the response (manually or via LLM) and adjust beliefs.

### How This Differs From Our Approach

We currently trust self-report completely. There is no overclaiming detection, no consistency checking, no calibration.

### Worth the Complexity?

**Foil items: definitely yes** for the application layer's curriculum system. Generating 3-5 plausible foils per domain is trivial with an LLM. If someone claims familiarity with a foil, multiply all their beliefs by 0.7 (or whatever the measured discount factor is). Implementation in knowledge_map.py: ~25 lines for a `calibration_factor` field in BeliefState and an `add_foil_response()` method.

**Prerequisite consistency checking: already implicit** in our propagation — if someone claims to know X but not its prerequisite, the propagation catches this. But we should make it explicit: flag the inconsistency, ask the user to clarify, and potentially discount the inconsistent claim.

**Per-user calibration factor: simple and effective.** Track the ratio of verified claims to total claims over time. ~10 lines.

### Key Reference

Paulhus & Harms (2004) "Measuring cognitive ability with the overclaiming technique" — the foundational paper on using fabricated items in knowledge assessment.

---

## 7. Graph-Based Belief Propagation

### Pearl's Belief Propagation

Belief Propagation (BP), proposed by Judea Pearl (1982), passes messages between nodes in a graphical model to compute marginal probabilities. For tree-structured graphs (which our prerequisite DAGs often are), BP is exact. For graphs with cycles ("loopy BP"), it's approximate but often effective.

**Message passing on a factor graph:**
- Variable-to-factor messages: Product of all incoming factor messages except the target
- Factor-to-variable messages: Sum-product over factor, weighted by incoming variable messages

For our problem: each node has a binary state (knows/doesn't know). The prerequisite structure defines factors: P(child=1 | parent=0) should be low. Observations (self-reports) are additional factors linking hidden state to observed familiarity.

### Our Current Propagation vs. BP

Our propagation is a simplified, one-pass version:
```python
# If doesn't know X → children get clamped to 0.2
# If knows X well → prerequisites get raised to 0.8
```

This has problems:
- **One direction only**: Each update propagates once, no iterative convergence
- **Hard clamping**: min/max operations lose probability information
- **No uncertainty reasoning**: Doesn't account for how certain we are about the triggering observation
- **No transitive propagation in one step**: If A→B→C and we learn about A, B gets updated but C only gets updated when B is assessed

### Proper BP Would:

1. Define conditional probability tables: P(child_knows | parent_knows) and P(child_knows | parent_not_knows)
2. Define observation likelihood: P(self_report=X | truly_knows) and P(self_report=X | truly_not_knows)
3. Run iterative message passing until convergence
4. Get calibrated marginal probabilities for all nodes simultaneously

### Python Implementations

- **pgmpy**: Full-featured, supports exact and approximate inference on Bayesian networks. Well-documented but can be slow for larger graphs.
- **pomegranate**: Faster, supports BNs and HMMs. GPU acceleration available.
- **PGMax** (Google DeepMind): JAX-based factor graph inference with loopy BP. Very fast (GPU-accelerated) but heavyweight dependency.
- **PFG**: Lightweight factor graph library for loopy BP. Minimal dependencies.

### Worth the Complexity?

**This is the most impactful algorithmic improvement, but can be done without a library.**

For a DAG with ~70 nodes and binary states, we don't need a general-purpose inference library. We can implement BP directly in ~60 lines:

```
# Sketch of simplified BP for prerequisite DAG
# (not actual implementation, just illustrating the approach)
#
# For each edge (parent → child):
#   P(child=1 | parent=1) = 0.9  (usually know children if you know parent)
#   P(child=1 | parent=0) = 0.1  (unlikely to know child without parent)
#
# For observation at node n:
#   P(report="solid" | knows=1) = 0.85
#   P(report="solid" | knows=0) = 0.10  (overclaiming)
#
# Message passing: iterate until convergence
#   parent_to_child: P(child) updated based on belief about parent
#   child_to_parent: P(parent) updated based on belief about child
```

For tree-structured DAGs (no shared prerequisites), this converges in one forward-backward pass. For DAGs with shared prerequisites (common), loopy BP typically converges in 3-5 iterations for graphs this small.

**Recommendation**: Implement a simple 2-pass BP (forward then backward through topological order) as a replacement for `_propagate()`. This handles the common case (tree-like DAGs) exactly and provides good approximations for DAGs with shared nodes. ~50-60 lines, no external dependencies.

---

## Synthesis: Recommended Improvements

Ranked by impact/complexity ratio for a ~200-line library:

### Tier 1: High Impact, Low Complexity (do these)

1. **Replace greedy max-entropy with Expected Information Gain (EIG)**
   - ~30 lines. For each candidate node, simulate updates under each possible response, measure total entropy reduction. Accounts for graph structure naturally.
   - Expected improvement: 20-40% fewer questions to convergence on well-connected graphs.

2. **Replace ad-hoc propagation with simplified BP**
   - ~50-60 lines. Define CPTs for prerequisite edges, run forward-backward pass. Handles transitive propagation, shared prerequisites, and uncertain observations correctly.
   - Expected improvement: Properly calibrated beliefs instead of hard-clamped values. Catches more inferences (e.g., if you know C which requires B which requires A, belief in A should increase even if only C is assessed).

3. **Add calibration factor for overconfidence**
   - ~25 lines. `BeliefState.calibration: float = 1.0`, foil response tracking, discount factor.
   - Expected improvement: Catches overclaiming users. Especially important for the Dunning-Kruger zone (beginners who overestimate).

4. **Add fringe computation from KST**
   - ~20 lines. `inner_fringe(state)` and `outer_fringe(state)` functions returning the "peaks of knowledge" and "ready to learn" sets.
   - Expected improvement: Directly actionable output for learning recommendations.

### Tier 2: Moderate Impact, Moderate Complexity (consider these)

5. **Continuous-score update path**
   - ~15 lines. `update_beliefs_from_score(node_id, score: float)` accepting 0.0-1.0 instead of fixed familiarity levels. Enables LLM-graded assessment.

6. **Batch probe selection**
   - ~20 lines. `next_probe_batch(n)` selecting n diverse, high-EIG nodes. Useful for "assess these 5 at once" UI patterns.

7. **BKT-style noisy observation model**
   - ~20 lines. Instead of directly assigning belief from familiarity level, treat self-report as noisy observation with overclaim/underclaim probabilities. Bayesian update on existing belief.

### Tier 3: High Complexity, Diminishing Returns (probably skip)

8. **Multi-step lookahead (POMDP)**
   - Would need ~100+ lines. Two-step lookahead is feasible but EIG already captures most of the benefit. Only worth it if assessment efficiency is critical (it isn't — we have a patient user doing self-assessment, not a timed test).

9. **Full CDM/IRT integration**
   - Requires calibrated item parameters we don't have. Only relevant if we add actual test items with right/wrong answers.

10. **Deep Knowledge Tracing**
    - Requires training data (interaction sequences) we don't have. Overkill for self-report assessment. Could be relevant for tracking knowledge over time in the broader the application layer system.

11. **LLM-based GIRT model**
    - Wrong layer — belongs in application code, not the shared library. The library should stay LLM-free.

---

## Key Open-Source References

| Library | Language | What It Does | Relevant For |
|---------|----------|-------------|--------------|
| [pyBKT](https://github.com/CAHLR/pyBKT) | Python | Bayesian Knowledge Tracing | Noisy observation model |
| [catsim](https://github.com/douglasrizzo/catsim) | Python | CAT simulation with IRT | Item selection algorithms |
| [EduCAT](https://github.com/bigdata-ustc/EduCAT) | Python | CDM-based CAT framework | CDM concepts, Q-matrix |
| [EduCDM](https://github.com/bigdata-ustc/EduCDM) | Python | Cognitive diagnosis models | DINA/DINO/NeuralCDM |
| [pgmpy](https://pgmpy.org/) | Python | Bayesian networks + inference | Belief propagation |
| [PGMax](https://github.com/google-deepmind/PGMax) | Python/JAX | Factor graph loopy BP | High-performance BP |
| [PFG](https://pypi.org/project/pfg/) | Python | Lightweight factor graph BP | Simple BP without deps |
| [kst (R)](https://cran.r-project.org/web/packages/kst/) | R | Knowledge Space Theory | KST algorithms |
| [kst (Python)](https://github.com/milansegedinac/kst) | Python | KST basic algorithms | KST concepts |

---

## Summary Table

| Algorithm Area | What It Improves | Complexity to Add | Impact | Recommendation |
|---------------|------------------|-------------------|--------|----------------|
| KST fringes | Actionable output ("ready to learn") | Low (~20 lines) | Medium | Do it |
| KST probabilistic assessment | Joint state distribution | High (full rewrite) | High | Skip — too complex for self-report |
| IRT/Fisher Information | Question selection optimality | Medium | Low | Skip — need calibrated items |
| CDM attribute hierarchy | Prerequisite-aware estimation | Medium | Medium | Borrow concepts, not code |
| EIG question selection | Better question ordering | Low (~30 lines) | High | Do it |
| Simplified BP propagation | Correct belief propagation | Medium (~60 lines) | High | Do it |
| BKT noisy observation | Robust to self-report error | Low (~20 lines) | Medium | Do it |
| Multi-step lookahead | Marginally better ordering | High (~100+ lines) | Low | Skip |
| Batch selection | Multiple questions at once | Low (~20 lines) | Medium | Do it |
| Foil-based calibration | Overconfidence detection | Low (~25 lines) | High | Do it |
| LLM assessment | Rich probing questions | N/A (wrong layer) | High | App layer, not library |
| DKT/attention models | Temporal knowledge tracking | Very high | Low for assessment | Skip for this module |
| Full loopy BP library | General inference | High (new dependency) | Low | Overkill — simple BP suffices |
