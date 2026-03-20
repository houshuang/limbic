"""Experiment 4: Convex Combination vs RRF Fusion.

Compares Reciprocal Rank Fusion (current amygdala default, k=60) against
convex combination for hybrid retrieval.

Key design insight: With only ~50 docs and a strong embedding model, vector
search is near-perfect, so fusion methods are indistinguishable. This experiment
uses 200+ documents with deliberately hard retrieval challenges:
- Large clusters of semantically similar documents (15+ per topic)
- Queries requiring disambiguation via specific technical terms
- Paraphrase queries with zero keyword overlap
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex, FTS5Index


# ---------------------------------------------------------------------------
# 1. Large corpus with controlled difficulty
# ---------------------------------------------------------------------------

# Core documents organized in large topical clusters.
# Within each cluster, documents are semantically similar but about
# different specific subtopics.

DOCUMENTS = [
    # ===== DATABASE CLUSTER (15 docs) - many near-duplicates =====
    ("db01", "PostgreSQL is an advanced open-source relational database system with strong SQL compliance and ACID transaction guarantees."),
    ("db02", "MySQL is the most popular open-source relational database, widely used in web applications with LAMP stack deployments."),
    ("db03", "SQLite is an embedded database engine that stores the entire database in a single file without needing a server process."),
    ("db04", "Oracle Database is a proprietary multi-model database management system with advanced features for enterprise workloads."),
    ("db05", "Microsoft SQL Server provides relational database services with integrated business intelligence tools for Windows environments."),
    ("db06", "MongoDB stores data in flexible JSON-like BSON documents, making it popular for applications with evolving schemas."),
    ("db07", "Cassandra is a distributed NoSQL database designed for handling large volumes of data across many commodity servers."),
    ("db08", "Redis is an in-memory key-value store used as a database, cache, and message broker with sub-millisecond latency."),
    ("db09", "CouchDB uses a multi-version concurrency control model and stores data as JSON documents accessible via HTTP REST API."),
    ("db10", "Neo4j is a native graph database that stores and queries relationships using Cypher query language with ACID compliance."),
    ("db11", "InfluxDB is a time-series database optimized for fast ingestion and real-time queries on timestamped measurement data."),
    ("db12", "DynamoDB is a fully managed NoSQL database service from AWS providing consistent single-digit millisecond latency at any scale."),
    ("db13", "CockroachDB is a distributed SQL database built for cloud-native applications with automatic sharding and consensus replication."),
    ("db14", "Elasticsearch provides distributed search and analytics with full-text search capabilities built on the Apache Lucene library."),
    ("db15", "TimescaleDB extends PostgreSQL for time-series data with automatic partitioning and continuous aggregates for analytics."),

    # ===== SORTING ALGORITHM CLUSTER (12 docs) =====
    ("sort01", "Quicksort partitions arrays around a pivot and recursively sorts subarrays with average O(n log n) time complexity."),
    ("sort02", "Merge sort divides arrays in half, recursively sorts each half, then merges them in O(n log n) worst-case time."),
    ("sort03", "Heap sort builds a binary max-heap and repeatedly extracts the maximum to produce a sorted array in-place."),
    ("sort04", "Bubble sort compares adjacent elements and swaps them if out of order, iterating until no swaps are needed."),
    ("sort05", "Insertion sort builds the sorted array one element at a time by finding the correct position for each new element."),
    ("sort06", "Radix sort distributes elements into buckets by individual digits, processing from least to most significant position."),
    ("sort07", "Counting sort determines element positions by counting occurrences, working in O(n+k) for integer keys in range k."),
    ("sort08", "Shell sort improves on insertion sort by comparing elements separated by a gap that decreases over iterations."),
    ("sort09", "Tim sort is a hybrid algorithm combining merge sort and insertion sort, used as the default in Python and Java."),
    ("sort10", "Introsort begins with quicksort and switches to heapsort when recursion depth exceeds a threshold to guarantee O(n log n)."),
    ("sort11", "Bucket sort distributes elements into buckets, sorts each bucket individually, then concatenates for the final sorted order."),
    ("sort12", "Selection sort repeatedly finds the minimum from the unsorted portion and places it at the beginning of the sorted portion."),

    # ===== MACHINE LEARNING CLUSTER (15 docs) =====
    ("ml01", "Linear regression fits a straight line to data by minimizing the sum of squared residuals between predictions and actual values."),
    ("ml02", "Logistic regression models binary classification using the sigmoid function to output probabilities between zero and one."),
    ("ml03", "Decision trees split data recursively on features that maximize information gain or minimize Gini impurity at each node."),
    ("ml04", "Random forests combine many decision trees trained on bootstrapped samples with random feature subsets to reduce variance."),
    ("ml05", "Support vector machines find the hyperplane that maximizes the margin between classes in high-dimensional feature space."),
    ("ml06", "K-means clustering partitions observations into k groups by iteratively assigning points to the nearest centroid and updating."),
    ("ml07", "Principal component analysis reduces dimensionality by projecting data onto orthogonal axes of maximum variance."),
    ("ml08", "Naive Bayes classifiers apply Bayes theorem with the assumption of conditional independence between features given the class."),
    ("ml09", "Gradient boosting builds additive models by sequentially fitting trees to the negative gradient of the loss function."),
    ("ml10", "XGBoost implements gradient boosting with regularization, column subsampling, and an efficient split-finding algorithm."),
    ("ml11", "K-nearest neighbors classifies points by majority vote among the k closest training examples using distance metrics."),
    ("ml12", "Neural networks learn nonlinear functions through layers of neurons with weighted connections trained by backpropagation."),
    ("ml13", "Autoencoders learn compressed representations by training encoder-decoder networks to reconstruct their input."),
    ("ml14", "Bayesian optimization uses a probabilistic surrogate model and acquisition function to efficiently search hyperparameter spaces."),
    ("ml15", "Ensemble methods combine predictions from multiple models to achieve better accuracy than any individual model."),

    # ===== DEEP LEARNING CLUSTER (12 docs) =====
    ("dl01", "Convolutional neural networks apply learnable filters to detect spatial patterns in images through convolution and pooling."),
    ("dl02", "Recurrent neural networks process sequences by maintaining hidden states that carry information across timesteps."),
    ("dl03", "Long short-term memory networks address vanishing gradients with gating mechanisms that control information flow through cells."),
    ("dl04", "Transformer architectures use multi-head self-attention to process all positions in parallel without recurrence."),
    ("dl05", "Generative adversarial networks train a generator and discriminator in a minimax game to produce realistic synthetic data."),
    ("dl06", "Variational autoencoders learn probabilistic latent representations by optimizing evidence lower bound with reparameterization."),
    ("dl07", "Attention mechanisms allow models to focus on relevant parts of the input when producing each output element."),
    ("dl08", "Batch normalization normalizes layer inputs to stabilize training and allow higher learning rates in deep networks."),
    ("dl09", "Dropout randomly deactivates neurons during training to prevent co-adaptation and reduce overfitting."),
    ("dl10", "Residual connections add shortcut paths that skip layers, enabling training of very deep networks by mitigating gradient degradation."),
    ("dl11", "Knowledge distillation transfers learned representations from a large teacher model to a smaller student model."),
    ("dl12", "Graph neural networks propagate information along edges to learn node and graph representations from relational data."),

    # ===== PROGRAMMING LANGUAGE CLUSTER (12 docs) =====
    ("lang01", "Python is an interpreted high-level language with dynamic typing and significant whitespace that emphasizes code readability."),
    ("lang02", "JavaScript is the primary language for web browsers, supporting event-driven programming with first-class functions and prototypal inheritance."),
    ("lang03", "Rust provides memory safety without garbage collection through its ownership and borrowing system with zero-cost abstractions."),
    ("lang04", "Go is a statically typed compiled language with built-in concurrency through goroutines and channels for systems programming."),
    ("lang05", "Java is a platform-independent object-oriented language that compiles to bytecode running on the Java Virtual Machine."),
    ("lang06", "C++ extends C with classes, templates, and the Standard Template Library for high-performance systems and game development."),
    ("lang07", "TypeScript adds static type annotations to JavaScript and compiles to plain JavaScript for safer large-scale web development."),
    ("lang08", "Haskell is a purely functional language with lazy evaluation, strong static typing, and algebraic data types."),
    ("lang09", "Kotlin is a modern JVM language with null safety, coroutines, and full interoperability with existing Java codebases."),
    ("lang10", "Swift combines safety features like optionals and value types with high performance for Apple platform development."),
    ("lang11", "Elixir runs on the BEAM virtual machine and provides fault-tolerant concurrent programming with the actor model."),
    ("lang12", "Clojure is a dynamic functional Lisp dialect on the JVM emphasizing immutable data and persistent data structures."),

    # ===== CLIMATE / ENERGY CLUSTER (12 docs) =====
    ("clim01", "Photovoltaic panels convert sunlight directly into electricity using semiconductor materials, primarily crystalline silicon wafers."),
    ("clim02", "Offshore wind turbines harness stronger ocean winds but face higher installation and maintenance costs than onshore installations."),
    ("clim03", "Geothermal power extracts heat from underground reservoirs to generate steam that drives turbines for baseload electricity."),
    ("clim04", "Nuclear fusion aims to replicate stellar energy production by combining hydrogen isotopes at extreme temperatures and pressures."),
    ("clim05", "Carbon capture and storage removes CO2 from industrial emissions and injects it into geological formations underground."),
    ("clim06", "Tidal energy converts the kinetic energy of ocean tides into electricity using underwater turbines in coastal areas."),
    ("clim07", "Biomass energy derives from burning organic materials like wood chips, agricultural waste, and dedicated energy crops."),
    ("clim08", "Hydroelectric dams convert the potential energy of water stored at height into electricity through turbine generators."),
    ("clim09", "Methane from livestock digestion and manure management accounts for a significant portion of agricultural greenhouse gas emissions."),
    ("clim10", "Permafrost thaw releases stored carbon as CO2 and methane, creating a positive feedback loop that accelerates warming."),
    ("clim11", "Heat pumps transfer thermal energy from ambient air or ground to buildings, achieving efficiencies three to five times higher than electric heaters."),
    ("clim12", "Green hydrogen produced by electrolysis of water using renewable electricity could decarbonize heavy industry and long-distance transport."),

    # ===== HISTORY CLUSTER (10 docs) =====
    ("hist01", "The printing press invented by Gutenberg around 1440 used movable metal type to mass-produce books for the first time."),
    ("hist02", "The Silk Road connected Chinese markets to Mediterranean traders through a network of overland and maritime trade routes."),
    ("hist03", "The Industrial Revolution mechanized production through steam-powered machinery, transforming Britain from agrarian to manufacturing economy."),
    ("hist04", "The Renaissance revived classical Greek and Roman art, philosophy, and science across fourteenth to seventeenth century Europe."),
    ("hist05", "Viking longships enabled Norse explorers to reach North America centuries before Columbus through the North Atlantic route."),
    ("hist06", "The Mongol Empire established the largest contiguous land empire using cavalry tactics and a sophisticated postal relay system."),
    ("hist07", "The French Revolution abolished the monarchy and feudal privileges, establishing principles of popular sovereignty and civil rights."),
    ("hist08", "The Song Dynasty in China invented movable type printing, gunpowder weapons, and the magnetic compass for navigation."),
    ("hist09", "The Roman aqueduct system transported water over long distances using gravity and arched bridges as engineering marvels."),
    ("hist10", "The Ottoman Empire controlled key trade routes between Europe and Asia for over six centuries from its capital Constantinople."),

    # ===== COOKING / FOOD SCIENCE (10 docs) =====
    ("food01", "The Maillard reaction between amino acids and reducing sugars at high temperatures creates hundreds of flavor and aroma compounds."),
    ("food02", "Gluten development through kneading aligns protein strands into an elastic network that traps fermentation gases in bread dough."),
    ("food03", "Sous vide cooking seals food in vacuum bags and heats it in precisely temperature-controlled water for even results."),
    ("food04", "Spherification uses sodium alginate and calcium chloride to form gel membranes around liquid creating caviar-like spheres."),
    ("food05", "Enzymatic browning occurs when polyphenol oxidase reacts with oxygen after cutting fruits, producing brown melanin pigments."),
    ("food06", "Cold smoking at temperatures below 30 degrees preserves food through antimicrobial phenol compounds without cooking it."),
    ("food07", "Lacto-fermentation by Lactobacillus bacteria converts sugars to lactic acid, preserving vegetables with tangy sour flavors."),
    ("food08", "Caramelization occurs when sugar is heated above its melting point, breaking down into hundreds of compounds with nutty brown flavors."),
    ("food09", "Emulsification stabilizes oil and water mixtures using surfactant molecules like lecithin or mustard that reduce surface tension."),
    ("food10", "Tempering chocolate involves precisely cycling temperatures to form stable cocoa butter crystals that give a glossy snap."),

    # ===== CRYPTOGRAPHY CLUSTER (8 docs) =====
    ("crypt01", "RSA encryption bases its security on the computational difficulty of factoring the product of two large prime numbers."),
    ("crypt02", "Elliptic curve cryptography achieves equivalent security to RSA using much smaller key sizes through algebraic structures."),
    ("crypt03", "AES is a symmetric block cipher that encrypts data in 128-bit blocks using 128, 192, or 256-bit keys."),
    ("crypt04", "SHA-256 produces a fixed 256-bit hash from arbitrary input, used in Bitcoin mining and digital signature verification."),
    ("crypt05", "Zero-knowledge proofs allow one party to prove knowledge of information without revealing the information to the verifier."),
    ("crypt06", "Diffie-Hellman key exchange enables two parties to establish a shared secret over an insecure channel using modular arithmetic."),
    ("crypt07", "Homomorphic encryption allows computation on ciphertext that produces encrypted results matching operations performed on the plaintext."),
    ("crypt08", "Post-quantum cryptography develops algorithms resistant to attacks by quantum computers using lattice-based and hash-based constructions."),

    # ===== NEUROSCIENCE / PSYCHOLOGY (8 docs) =====
    ("neuro01", "Hebbian learning strengthens synaptic connections between neurons that fire simultaneously, summarized as cells that fire together wire together."),
    ("neuro02", "The default mode network activates during mind-wandering and self-referential thought when the brain is not task-focused."),
    ("neuro03", "Neuroplasticity enables the brain to reorganize neural pathways throughout life in response to learning and injury."),
    ("neuro04", "Cognitive load theory explains how working memory limitations constrain the amount of new information learners can process."),
    ("neuro05", "Confirmation bias leads people to favor information that confirms existing beliefs while discounting contradictory evidence."),
    ("neuro06", "Dual process theory distinguishes fast automatic System 1 thinking from slow deliberate System 2 analytical reasoning."),
    ("neuro07", "Mirror neurons fire both when an animal performs an action and when it observes the same action performed by another."),
    ("neuro08", "The spacing effect shows that distributed practice over time produces stronger long-term retention than massed practice."),

    # ===== ECOLOGY (8 docs) =====
    ("eco01", "Mycorrhizal networks connect trees underground through fungal hyphae, enabling nutrient transfer and chemical signaling across forests."),
    ("eco02", "Keystone species exert disproportionate influence on their ecosystem relative to their abundance, like sea otters controlling urchin populations."),
    ("eco03", "Trophic cascades occur when changes at one level of the food chain ripple through to affect multiple other levels."),
    ("eco04", "Coral bleaching occurs when stressed corals expel symbiotic zooxanthellae algae, losing their color and primary energy source."),
    ("eco05", "Nitrogen fixation by rhizobium bacteria in legume root nodules converts atmospheric N2 into plant-usable ammonium compounds."),
    ("eco06", "Edge effects create distinct microclimates at habitat boundaries that favor generalist species over interior specialists."),
    ("eco07", "Biological magnification concentrates toxins like DDT and mercury at higher trophic levels through the food chain."),
    ("eco08", "Invasive species outcompete natives through lack of natural predators, rapid reproduction, and broad environmental tolerances."),

    # ===== ECONOMICS (8 docs) =====
    ("econ01", "Quantitative easing increases money supply when central banks purchase government bonds from commercial banks."),
    ("econ02", "Behavioral economics demonstrates that loss aversion makes people weigh potential losses about twice as heavily as equivalent gains."),
    ("econ03", "Network effects create winner-take-all markets where platform value increases with each additional user."),
    ("econ04", "The Gini coefficient measures income inequality from zero representing perfect equality to one representing maximum concentration."),
    ("econ05", "Moral hazard arises when insured parties take greater risks because they do not bear the full cost of those risks."),
    ("econ06", "Comparative advantage shows that trade benefits both parties even when one can produce everything more efficiently."),
    ("econ07", "The Phillips curve suggests an inverse relationship between unemployment and inflation rates in the short run."),
    ("econ08", "Tragedy of the commons occurs when individuals deplete shared resources by acting in their own self-interest."),

    # ===== PHYSICS / SPACE (8 docs) =====
    ("phys01", "Gravitational lensing bends light from distant sources around massive foreground objects, creating arcs and multiple images."),
    ("phys02", "Fast radio bursts are millisecond-duration pulses of radio emission from extragalactic sources with debated origins."),
    ("phys03", "The cosmic microwave background is relic radiation from when the universe cooled enough for photons to decouple from matter."),
    ("phys04", "Neutron star mergers produce heavy elements through rapid neutron capture and emit detectable gravitational waves."),
    ("phys05", "Quantum entanglement links particle states so that measuring one instantly determines the state of its partner."),
    ("phys06", "The Higgs boson gives mass to fundamental particles through their interaction with the Higgs field pervading all space."),
    ("phys07", "Dark matter makes up about 27% of the universe's mass-energy but does not emit or absorb electromagnetic radiation."),
    ("phys08", "Hawking radiation is theoretical thermal radiation emitted by black holes due to quantum effects near the event horizon."),

    # ===== CONFOUNDERS: same words, different domains =====
    ("conf01", "Python is a large nonvenomous constrictor snake native to Africa and Asia that kills prey by squeezing."),
    ("conf02", "Java is the most populous island in Indonesia, home to the capital Jakarta and rich Javanese cultural traditions."),
    ("conf03", "A shell is the hard outer covering of mollusks like clams and snails, composed primarily of calcium carbonate."),
    ("conf04", "The Rust Belt refers to the economic decline of former industrial cities in the northeastern and midwestern United States."),
    ("conf05", "Swift is a common migratory bird known for spending almost its entire life on the wing, even sleeping while flying."),
    ("conf06", "Kotlin Island in the Gulf of Finland served as a Russian naval fortress and is now part of Saint Petersburg."),
    ("conf07", "Gradient descent in hiking means following the steepest downhill path to reach the valley floor most quickly."),
    ("conf08", "A random forest is a natural woodland where tree species grow in unpredictable patterns without deliberate planting."),
    ("conf09", "Docker is a breed of spaniel originally developed for flushing birds from dense undergrowth during hunting."),
    ("conf10", "Containers in shipping are standardized steel boxes measuring twenty or forty feet used for global freight transport."),
]

# Total: 15+12+15+12+12+12+10+10+8+8+8+8+8+10 = 148 documents

# ---------------------------------------------------------------------------
# 2. Query set with relevance judgments
# ---------------------------------------------------------------------------

QUERIES = [
    # ===== KEYWORD QUERIES (FTS should help) =====
    # Queries with specific technical terms that distinguish within a cluster
    ("PostgreSQL open-source relational database ACID transactions", {"db01"}, "keyword"),
    ("Redis in-memory key-value store cache message broker", {"db08"}, "keyword"),
    ("InfluxDB time-series database measurement data", {"db11"}, "keyword"),
    ("CockroachDB distributed SQL automatic sharding consensus", {"db13"}, "keyword"),
    ("quicksort partition pivot recursive O(n log n)", {"sort01"}, "keyword"),
    ("radix sort buckets digits least significant", {"sort06"}, "keyword"),
    ("Tim sort hybrid merge insertion Python Java default", {"sort09"}, "keyword"),
    ("XGBoost gradient boosting regularization column subsampling", {"ml10"}, "keyword"),
    ("Bayesian optimization surrogate model acquisition function hyperparameter", {"ml14"}, "keyword"),
    ("Rust ownership borrowing memory safety zero-cost abstractions", {"lang03"}, "keyword"),
    ("Haskell purely functional lazy evaluation algebraic data types", {"lang08"}, "keyword"),
    ("Elixir BEAM virtual machine fault-tolerant actor model", {"lang11"}, "keyword"),
    ("RSA encryption factoring large prime numbers security", {"crypt01"}, "keyword"),
    ("SHA-256 hash Bitcoin mining digital signature", {"crypt04"}, "keyword"),
    ("Diffie-Hellman key exchange shared secret modular arithmetic", {"crypt06"}, "keyword"),

    # ===== SEMANTIC QUERIES (vector should help; FTS gets zero hits) =====
    ("a data store that keeps everything in RAM for speed", {"db08"}, "semantic"),
    ("database that understands connections between things using graph queries", {"db10"}, "semantic"),
    ("ordering items by looking at each digit from right to left", {"sort06"}, "semantic"),
    ("the simplest sorting approach that repeatedly swaps neighbors", {"sort04"}, "semantic"),
    ("teaching a small model to imitate a bigger more capable one", {"dl11"}, "semantic"),
    ("skipping layers in very deep networks to help gradients flow", {"dl10"}, "semantic"),
    ("the programming language that uses goroutines for parallelism", {"lang04"}, "semantic"),
    ("energy source that uses temperature differences deep underground", {"clim03"}, "semantic"),
    ("proving you know a secret without telling anyone what it is", {"crypt05"}, "semantic"),
    ("why people make bad decisions when afraid of losing money", {"econ02"}, "semantic"),
    ("underground fungal highways connecting trees in a forest", {"eco01"}, "semantic"),
    ("chemical reaction that makes bread crust brown and flavorful", {"food01"}, "semantic"),
    ("your brain rewiring itself after an injury or new learning", {"neuro03"}, "semantic"),
    ("the faint glow left over from the birth of the universe", {"phys03"}, "semantic"),
    ("animals whose disappearance would collapse the ecosystem", {"eco02", "eco03"}, "semantic"),

    # ===== HYBRID QUERIES (both signals needed) =====
    # Disambiguation: keyword helps pick the right homonym
    ("Python programming language dynamic typing readability", {"lang01"}, "hybrid"),
    ("Python constrictor snake nonvenomous Africa Asia", {"conf01"}, "hybrid"),
    ("Rust programming language memory safety ownership", {"lang03"}, "hybrid"),
    ("Rust Belt economic decline industrial cities", {"conf04"}, "hybrid"),
    ("Java programming language JVM bytecode platform-independent", {"lang05"}, "hybrid"),
    ("Java island Indonesia Jakarta Javanese culture", {"conf02"}, "hybrid"),

    # Queries needing both keyword specificity and semantic understanding
    ("distributed NoSQL database large scale no single point of failure", {"db07", "db12"}, "hybrid"),
    ("stable O(n log n) comparison-based sorting with divide and conquer", {"sort02"}, "hybrid"),
    ("attention mechanism that lets all positions interact in parallel", {"dl04", "dl07"}, "hybrid"),
    ("deep learning technique to prevent overfitting by randomly disabling units", {"dl09"}, "hybrid"),
    ("agricultural greenhouse gas emissions from livestock and farming", {"clim09"}, "hybrid"),
    ("post-quantum lattice-based cryptography resistant to quantum attacks", {"crypt08"}, "hybrid"),
    ("learning happens when neurons fire at the same time and connections strengthen", {"neuro01"}, "hybrid"),
    ("economic principle about depleting shared resources through selfish behavior", {"econ08"}, "hybrid"),
    ("what happens when carbon locked in frozen ground thaws and enters atmosphere", {"clim10"}, "hybrid"),
]

assert len(QUERIES) == 45


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

def calc_mrr(ranked, relevant):
    for i, d in enumerate(ranked):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at(ranked, relevant, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(ranked_lists, queries):
    ms = [calc_mrr(ranked_lists[q], rel) for q, rel, _ in queries]
    r5 = [recall_at(ranked_lists[q], rel, 5) for q, rel, _ in queries]
    r10 = [recall_at(ranked_lists[q], rel, 10) for q, rel, _ in queries]
    n5 = [ndcg_at(ranked_lists[q], rel, 5) for q, rel, _ in queries]
    n10 = [ndcg_at(ranked_lists[q], rel, 10) for q, rel, _ in queries]
    return {
        "MRR": round(float(np.mean(ms)), 4),
        "Recall@5": round(float(np.mean(r5)), 4),
        "Recall@10": round(float(np.mean(r10)), 4),
        "nDCG@5": round(float(np.mean(n5)), 4),
        "nDCG@10": round(float(np.mean(n10)), 4),
    }


# ---------------------------------------------------------------------------
# 4. Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(vec_res, fts_res, k=60, limit=60):
    scores = {}
    for rank, r in enumerate(vec_res):
        scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(fts_res):
        scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:limit]


def convex_fuse(vec_res, fts_res, alpha, limit=60):
    def minmax(d):
        if not d:
            return {}
        mn, mx = min(d.values()), max(d.values())
        rng = mx - mn if mx > mn else 1.0
        return {k: (v - mn) / rng for k, v in d.items()}

    vn = minmax({r.id: r.score for r in vec_res})
    fn = minmax({r.id: r.score for r in fts_res})
    scores = {}
    for did in set(vn) | set(fn):
        scores[did] = alpha * vn.get(did, 0.0) + (1 - alpha) * fn.get(did, 0.0)
    return sorted(scores, key=scores.get, reverse=True)[:limit]


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def fmt(label, m, w=24):
    return f"  {label:<{w}} {m['MRR']:>7.4f} {m['Recall@5']:>7.4f} {m['Recall@10']:>7.4f} {m['nDCG@5']:>8.4f} {m['nDCG@10']:>8.4f}"


def main():
    W = 78
    print("=" * W)
    print("Experiment 4: Convex Combination vs RRF Fusion")
    print("=" * W)

    doc_ids = [d[0] for d in DOCUMENTS]
    doc_texts = [d[1] for d in DOCUMENTS]

    print(f"\nCorpus: {len(DOCUMENTS)} documents, {len(QUERIES)} queries "
          f"({sum(1 for _,_,t in QUERIES if t=='keyword')} keyword, "
          f"{sum(1 for _,_,t in QUERIES if t=='semantic')} semantic, "
          f"{sum(1 for _,_,t in QUERIES if t=='hybrid')} hybrid)")

    # Build indices
    print("Indexing...")
    t0 = time.time()
    model = EmbeddingModel()
    embeddings = model.embed_batch(doc_texts)
    vi = VectorIndex()
    vi.add(doc_ids, embeddings)
    fi = FTS5Index()
    fi.add_batch([{"id": d[0], "content": d[1]} for d in DOCUMENTS])
    print(f"  Done in {time.time() - t0:.1f}s")

    # Retrieve
    print("Retrieving...\n")
    FETCH = 60
    vec_res, fts_res = {}, {}
    for qt, _, _ in QUERIES:
        vec_res[qt] = vi.search(model.embed(qt), limit=FETCH)
        fts_res[qt] = fi.search(qt, limit=FETCH)

    vec_only = {q: [r.id for r in vec_res[q]] for q, _, _ in QUERIES}
    fts_only = {q: [r.id for r in fts_res[q]] for q, _, _ in QUERIES}
    R = {}  # all results

    # ---- Baselines ----
    hdr = f"  {'Method':<24} {'MRR':>7} {'R@5':>7} {'R@10':>7} {'nDCG@5':>8} {'nDCG@10':>8}"
    print("-" * W)
    print("BASELINES")
    print("-" * W)
    print(hdr)
    for name, rl in [("Vector only", vec_only), ("FTS5 only", fts_only)]:
        m = evaluate(rl, QUERIES)
        R[name] = m
        print(fmt(name, m))

    for qtype in ["keyword", "semantic", "hybrid"]:
        sub = [(q, r, t) for q, r, t in QUERIES if t == qtype]
        vm = evaluate(vec_only, sub)
        fm = evaluate(fts_only, sub)
        R[f"base_vec_{qtype}"] = vm
        R[f"base_fts_{qtype}"] = fm
        print(f"\n  [{qtype}] ({len(sub)} queries)")
        print(f"    Vector:  MRR={vm['MRR']:.4f}  R@5={vm['Recall@5']:.4f}  R@10={vm['Recall@10']:.4f}  nDCG@10={vm['nDCG@10']:.4f}")
        print(f"    FTS5:    MRR={fm['MRR']:.4f}  R@5={fm['Recall@5']:.4f}  R@10={fm['Recall@10']:.4f}  nDCG@10={fm['nDCG@10']:.4f}")

    # ---- RRF ----
    print(f"\n{'-' * W}")
    print("RRF FUSION (varying k)")
    print("-" * W)
    print(f"  {'k':>5}  {'MRR':>7} {'R@5':>7} {'R@10':>7} {'nDCG@5':>8} {'nDCG@10':>8}")

    K_VALS = [1, 5, 10, 20, 40, 60, 80, 100]
    rrf_by_k = {}
    for k in K_VALS:
        rl = {q: rrf_fuse(vec_res[q], fts_res[q], k=k) for q, _, _ in QUERIES}
        m = evaluate(rl, QUERIES)
        rrf_by_k[k] = m
        R[f"rrf_k{k}"] = m
        tag = " <-- default" if k == 60 else ""
        print(f"  {k:>5}  {m['MRR']:>7.4f} {m['Recall@5']:>7.4f} {m['Recall@10']:>7.4f} {m['nDCG@5']:>8.4f} {m['nDCG@10']:>8.4f}{tag}")

    # ---- Convex ----
    print(f"\n{'-' * W}")
    print("CONVEX COMBINATION (alpha: 1=vector, 0=FTS5)")
    print("-" * W)
    print(f"  {'alpha':>7}  {'MRR':>7} {'R@5':>7} {'R@10':>7} {'nDCG@5':>8} {'nDCG@10':>8}")

    cx_by_a = {}
    for a10 in range(0, 11):
        a = a10 / 10.0
        rl = {q: convex_fuse(vec_res[q], fts_res[q], alpha=a) for q, _, _ in QUERIES}
        m = evaluate(rl, QUERIES)
        cx_by_a[a] = m
        R[f"convex_a{a:.1f}"] = m
        print(f"  {a:>7.1f}  {m['MRR']:>7.4f} {m['Recall@5']:>7.4f} {m['Recall@10']:>7.4f} {m['nDCG@5']:>8.4f} {m['nDCG@10']:>8.4f}")

    # ---- Tuned train/test ----
    n_train = len(QUERIES) // 2
    train_q, test_q = QUERIES[:n_train], QUERIES[n_train:]

    print(f"\n{'-' * W}")
    print(f"TUNED COMPARISON (train: {n_train}, test: {len(QUERIES) - n_train})")
    print("-" * W)

    # Tune alpha at 5% steps
    best_a, best_a_s = 0.5, -1.0
    for a100 in range(0, 101, 5):
        a = a100 / 100.0
        rl = {q: convex_fuse(vec_res[q], fts_res[q], alpha=a) for q, _, _ in train_q}
        s = evaluate(rl, train_q)["MRR"]
        if s > best_a_s:
            best_a_s, best_a = s, a

    # Tune k
    best_k, best_k_s = 60, -1.0
    for k in K_VALS:
        rl = {q: rrf_fuse(vec_res[q], fts_res[q], k=k) for q, _, _ in train_q}
        s = evaluate(rl, train_q)["MRR"]
        if s > best_k_s:
            best_k_s, best_k = s, k

    print(f"  Tuned: convex alpha={best_a:.2f} (train MRR={best_a_s:.4f}), RRF k={best_k} (train MRR={best_k_s:.4f})")

    tc = evaluate({q: convex_fuse(vec_res[q], fts_res[q], alpha=best_a) for q, _, _ in test_q}, test_q)
    tr = evaluate({q: rrf_fuse(vec_res[q], fts_res[q], k=best_k) for q, _, _ in test_q}, test_q)
    tv = evaluate(vec_only, test_q)

    R["tuned_convex"] = {**tc, "alpha": best_a}
    R["tuned_rrf"] = {**tr, "k": best_k}

    print(f"\n  Test set ({len(test_q)} queries):")
    print(hdr)
    print(fmt(f"Convex(a={best_a:.2f})", tc))
    print(fmt(f"RRF(k={best_k})", tr))
    print(fmt("Vector only", tv))

    # ---- Per-type breakdown ----
    best_a_all = max(cx_by_a, key=lambda a: (cx_by_a[a]["MRR"], cx_by_a[a]["nDCG@10"]))
    best_k_all = max(rrf_by_k, key=lambda k: (rrf_by_k[k]["MRR"], rrf_by_k[k]["nDCG@10"]))

    print(f"\n{'-' * W}")
    print(f"PER-TYPE BREAKDOWN (convex a={best_a_all:.1f}, RRF k={best_k_all})")
    print("-" * W)

    for qtype in ["keyword", "semantic", "hybrid"]:
        sub = [(q, r, t) for q, r, t in QUERIES if t == qtype]
        vm = evaluate(vec_only, sub)
        fm = evaluate(fts_only, sub)
        cm = evaluate({q: convex_fuse(vec_res[q], fts_res[q], alpha=best_a_all) for q, _, _ in sub}, sub)
        rm = evaluate({q: rrf_fuse(vec_res[q], fts_res[q], k=best_k_all) for q, _, _ in sub}, sub)
        R[f"type_{qtype}_convex"] = cm
        R[f"type_{qtype}_rrf"] = rm
        print(f"\n  [{qtype}] ({len(sub)} queries)")
        print(f"    {'Vector only':<20} MRR={vm['MRR']:.4f}  R@5={vm['Recall@5']:.4f}  nDCG@10={vm['nDCG@10']:.4f}")
        print(f"    {'FTS5 only':<20} MRR={fm['MRR']:.4f}  R@5={fm['Recall@5']:.4f}  nDCG@10={fm['nDCG@10']:.4f}")
        print(f"    {'Convex(a='+f'{best_a_all:.1f})':<20} MRR={cm['MRR']:.4f}  R@5={cm['Recall@5']:.4f}  nDCG@10={cm['nDCG@10']:.4f}")
        print(f"    {'RRF(k='+f'{best_k_all})':<20} MRR={rm['MRR']:.4f}  R@5={rm['Recall@5']:.4f}  nDCG@10={rm['nDCG@10']:.4f}")

    # ---- Per-query diagnostics ----
    print(f"\n{'-' * W}")
    print("PER-QUERY DIAGNOSTICS")
    print("-" * W)

    n_shown = 0
    for qt, rel, qtype in QUERIES:
        cl = convex_fuse(vec_res[qt], fts_res[qt], alpha=best_a_all)
        rl = rrf_fuse(vec_res[qt], fts_res[qt], k=best_k_all)
        vl = [r.id for r in vec_res[qt]]

        c_m = calc_mrr(cl, rel)
        r_m = calc_mrr(rl, rel)
        v_m = calc_mrr(vl, rel)

        interesting = (abs(c_m - r_m) > 0.001 or v_m < 0.34
                       or (c_m != v_m and abs(c_m - v_m) > 0.01))
        if interesting:
            n_shown += 1
            ft5 = [r.id for r in fts_res[qt][:5]]
            fts_hits = sum(1 for r in fts_res[qt] if r.id in rel)
            print(f"\n  Q: \"{qt[:70]}...\" [{qtype}]" if len(qt) > 70 else f"\n  Q: \"{qt}\" [{qtype}]")
            print(f"    Relevant: {sorted(rel)}")
            print(f"    Vec  top5: {vl[:5]}  MRR={v_m:.3f}")
            print(f"    FTS  top5: {ft5}  hits={fts_hits}")
            print(f"    Convex={c_m:.4f}  RRF={r_m:.4f}  Vec={v_m:.4f}")

    if n_shown == 0:
        print("  (No interesting disagreements found)")

    # ---- Degraded vector quality experiment ----
    # Simulates a weaker/cheaper embedding model by adding Gaussian noise.
    # This is the regime where the TOIS paper found convex > RRF.
    print(f"\n{'=' * W}")
    print("DEGRADED VECTOR QUALITY (simulating weaker embedding model)")
    print("=" * W)

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    degraded_results = {}

    for noise in noise_levels:
        if noise == 0.0:
            noisy_vi = vi
        else:
            rng = np.random.RandomState(42)
            noisy_emb = embeddings + rng.randn(*embeddings.shape).astype(np.float32) * noise
            noisy_vi = VectorIndex()
            noisy_vi.add(doc_ids, noisy_emb)

        # Re-run vector search with noisy index
        noisy_vec_res = {}
        for qt, _, _ in QUERIES:
            q_emb = model.embed(qt)
            if noise > 0:
                q_emb = q_emb + rng.randn(*q_emb.shape).astype(np.float32) * noise * 0.5
            noisy_vec_res[qt] = noisy_vi.search(q_emb, limit=FETCH)

        noisy_vec_only = {q: [r.id for r in noisy_vec_res[q]] for q, _, _ in QUERIES}
        v_m = evaluate(noisy_vec_only, QUERIES)

        # Best convex alpha
        best_cx_mrr, best_cx_a = -1, 0.5
        for a100 in range(0, 101, 5):
            a = a100 / 100.0
            rl = {q: convex_fuse(noisy_vec_res[q], fts_res[q], alpha=a) for q, _, _ in QUERIES}
            s = evaluate(rl, QUERIES)["MRR"]
            if s > best_cx_mrr:
                best_cx_mrr, best_cx_a = s, a
        cx_m = evaluate({q: convex_fuse(noisy_vec_res[q], fts_res[q], alpha=best_cx_a) for q, _, _ in QUERIES}, QUERIES)

        # Best RRF k
        best_rrf_mrr, best_rrf_k = -1, 60
        for k in K_VALS:
            rl = {q: rrf_fuse(noisy_vec_res[q], fts_res[q], k=k) for q, _, _ in QUERIES}
            s = evaluate(rl, QUERIES)["MRR"]
            if s > best_rrf_mrr:
                best_rrf_mrr, best_rrf_k = s, k
        rr_m = evaluate({q: rrf_fuse(noisy_vec_res[q], fts_res[q], k=best_rrf_k) for q, _, _ in QUERIES}, QUERIES)

        degraded_results[noise] = {
            "vector_only": v_m,
            "best_convex": {**cx_m, "alpha": best_cx_a},
            "best_rrf": {**rr_m, "k": best_rrf_k},
        }
        R[f"degraded_{noise}_vec"] = v_m
        R[f"degraded_{noise}_convex"] = {**cx_m, "alpha": best_cx_a}
        R[f"degraded_{noise}_rrf"] = {**rr_m, "k": best_rrf_k}

    print(f"\n  {'noise':>7} | {'Vec MRR':>8} | {'Convex MRR':>10} {'(a)':>6} | {'RRF MRR':>8} {'(k)':>5} | {'Winner':>12}")
    print(f"  {'-'*7}-+-{'-'*8}-+-{'-'*10}-{'-'*6}-+-{'-'*8}-{'-'*5}-+-{'-'*12}")
    for noise in noise_levels:
        d = degraded_results[noise]
        v = d["vector_only"]["MRR"]
        c = d["best_convex"]["MRR"]
        r = d["best_rrf"]["MRR"]
        ca = d["best_convex"]["alpha"]
        rk = d["best_rrf"]["k"]
        if c - r > 0.001:
            w = "Convex"
        elif r - c > 0.001:
            w = "RRF"
        else:
            w = "Tie"
        print(f"  {noise:>7.1f} | {v:>8.4f} | {c:>10.4f} {ca:>5.2f} | {r:>8.4f} {rk:>5} | {w:>12}")

    # Detailed breakdown at noise=0.5 (moderate degradation)
    noise_detail = 0.5
    if noise_detail in degraded_results:
        d = degraded_results[noise_detail]
        print(f"\n  Detailed at noise={noise_detail}:")
        for name, m in [("Vector only", d["vector_only"]),
                        (f"Convex(a={d['best_convex']['alpha']:.2f})", d["best_convex"]),
                        (f"RRF(k={d['best_rrf']['k']})", d["best_rrf"])]:
            print(f"    {name:<22} MRR={m['MRR']:.4f}  R@5={m['Recall@5']:.4f}  R@10={m['Recall@10']:.4f}  nDCG@10={m['nDCG@10']:.4f}")

    # ---- Summary ----
    print(f"\n{'=' * W}")
    print("SUMMARY")
    print("=" * W)

    bc_k = max([k for k in R if k.startswith("convex_a")],
               key=lambda k: (R[k]["MRR"], R[k]["nDCG@10"]))
    br_k = max([k for k in R if k.startswith("rrf_k")],
               key=lambda k: (R[k]["MRR"], R[k]["nDCG@10"]))
    bc, br = R[bc_k], R[br_k]
    vo, fo = R["Vector only"], R["FTS5 only"]

    print(f"\n  {'Method':<28} {'MRR':>7} {'R@5':>7} {'R@10':>7} {'nDCG@10':>8}")
    for label, m in [(f"Best convex ({bc_k})", bc),
                     (f"Best RRF ({br_k})", br),
                     ("Vector only", vo),
                     ("FTS5 only", fo)]:
        print(f"  {label:<28} {m['MRR']:>7.4f} {m['Recall@5']:>7.4f} {m['Recall@10']:>7.4f} {m['nDCG@10']:>8.4f}")

    md = bc["MRR"] - br["MRR"]
    nd = bc["nDCG@10"] - br["nDCG@10"]
    if md > 0.001:
        winner = "Convex combination"
    elif md < -0.001:
        winner = "RRF"
    elif nd > 0.001:
        winner = "Convex (by nDCG@10)"
    elif nd < -0.001:
        winner = "RRF (by nDCG@10)"
    else:
        winner = "Tie"

    print(f"\n  Winner: {winner}")
    print(f"  MRR delta (convex - RRF): {md:+.4f}")
    print(f"  nDCG@10 delta: {nd:+.4f}")
    print(f"  Tuned test: convex(a={best_a:.2f}) MRR={tc['MRR']:.4f} vs RRF(k={best_k}) MRR={tr['MRR']:.4f}")

    # ---- Save ----
    out = {
        "experiment": "exp4_fusion_comparison",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "corpus_size": len(DOCUMENTS),
        "num_queries": len(QUERIES),
        "query_types": {t: sum(1 for _, _, tt in QUERIES if tt == t) for t in ["keyword", "semantic", "hybrid"]},
        "best_convex": bc_k,
        "best_rrf": br_k,
        "tuned_alpha": best_a,
        "tuned_k": best_k,
        "winner": winner,
        "results": R,
    }
    out_path = Path("experiments/results/exp4_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
