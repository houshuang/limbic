# Tutorial & Educational Platform Prerequisite/Skill Graphs

Research into publicly available educational prerequisite graphs and structured curriculum data, conducted 2026-03-19. Motivation: find real-world graph data for testing knowledge mapping algorithms in amygdala.

---

## 1. Khan Academy

### Knowledge Map (Historical)
Khan Academy had an interactive "Knowledge Map" showing prerequisite relationships between exercises, rendered as a node graph similar to a skill tree in an RPG. This feature was **removed in August 2013** and never restored, despite persistent community requests.

### API Status
- The **public API was deprecated in 2020**. Endpoints like `/api/v1/topictree` (which returned a ~10MB JSON of the full content tree) now return 307 redirects.
- Khan Academy moved to an **internal GraphQL API** with safelisted queries. External requests are blocked by CORS and require a security hash, making it impractical for data extraction.
- The old API did expose exercise `prerequisites` as slugs, but this data is no longer accessible.

### Alternative Access Routes
- **Kolibri/sushi-chef-khan-academy** ([GitHub](https://github.com/learningequality/sushi-chef-khan-academy)): Learning Equality's content integration script processes KA TSV exports into topic trees for offline use. The script explicitly lists "Handle `prerequisites` (links to exercises slugs)" as a **TODO** -- prerequisites were never implemented.
- **khan-api** ([khan-api.bhavjit.com](https://khan-api.bhavjit.com/)): Third-party Node.js client. May provide some content metadata access, but docs don't confirm prerequisite data.
- **Wayback Machine**: No confirmed archived snapshot of the full topic tree JSON with prerequisites.
- **Internet Archive**: Has KA videos archived but not structured API data.

### Verdict
**Not currently accessible.** The prerequisite graph data that once existed is locked behind deprecated APIs and internal systems. No known public download exists.

---

## 2. Metacademy

The most promising small-to-medium graph for ML/statistics/linear algebra concepts.

### Overview
[Metacademy](https://metacademy.org) is an open-source "package manager for knowledge" built around a directed prerequisite graph of ML concepts. Each concept has a description, learning goals, time estimate, learning resources, and a list of prerequisite concepts.

### Data
- **141 nodes** (concepts from ML, Statistics, Linear Algebra)
- **331 direct prerequisite edges** (1,586 total including transitive closure)
- Expert-curated prerequisite relationships with optional "reason" annotations

### Access
- **GitHub content repo**: [metacademy/metacademy-content](https://github.com/metacademy/metacademy-content) -- **DEPRECATED** (migrated to relational DB), but the flat-file structure is still clonable. Each concept lives in `content/nodes/{concept}/` with a `dependencies.txt` file listing prerequisites (tag, reason, shortcut flag).
- **Wiki on database format**: [Database format](https://github.com/metacademy/metacademy-content/wiki/Database-format) -- documents the flat-file schema
- **kmap visualization library**: [cjrd/kmap](https://github.com/cjrd/kmap) -- extracted from Metacademy codebase, accepts nodes with `id`, `title`, `summary`, `dependencies` arrays
- **Metacademy-to-Wikipedia mapping**: [Figshare CSV](https://figshare.com/articles/dataset/Metacademy-prereq-pairs-transoformed-to-wiki_csv/7799774) -- 141 concept prerequisite pairs mapped to Wikipedia articles, from the paper "Finding Prerequisite Relations using the Wikipedia Clickstream"
- **Web UI**: Individual concept graphs can be downloaded as JSON via the graph UI (icons in upper right)

### Format
Flat text files (dependencies.txt, title.txt, summary.txt, goals.txt, resources.txt per concept). The Figshare dataset is CSV.

### License
Open source (repo). Figshare dataset terms not specified.

### Learner Performance Data
No.

### Verdict
**Excellent small dataset.** The GitHub content repo can be cloned and parsed to reconstruct the full graph. The Figshare CSV provides a ready-made Wikipedia-mapped version. Good for testing prerequisite inference algorithms.

---

## 3. AL-CPL Dataset (Active Learning for Concept Prerequisite Learning)

### Overview
High-quality expert-annotated concept prerequisite pairs across four domains, built from Wikipedia concepts.

### Data
| Domain | Concepts | Pairs | Prerequisites |
|--------|----------|-------|---------------|
| Data Mining | 120 | 826 | 292 |
| Geometry | 89 | 1,681 | 524 |
| Physics | 153 | 1,962 | 487 |
| Precalculus | 224 | 2,060 | 699 |

**Total**: 586 concepts, 6,529 pairs, 2,002 prerequisites

### Access
- **GitHub**: [harrylclc/AL-CPL-dataset](https://github.com/harrylclc/AL-CPL-dataset)
- `.pairs` files: all concept pairs (positive + negative examples)
- `.preqs` files: prerequisite relations only
- Convention: "second concept (2nd column) is a prerequisite of the first concept (1st column)"
- Feature files in SVM-light format

### Format
CSV/TSV

### License
**CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike)

### Learner Performance Data
No.

### Verdict
**Very useful benchmark.** Clean, multi-domain, expert-validated prerequisite annotations. The four diverse domains (Data Mining, Geometry, Physics, Precalculus) make it good for testing domain-transfer of algorithms. Wikipedia-grounded concepts make it easy to enrich with additional features.

---

## 4. RefD Dataset (Reference Distance)

### Overview
Two datasets for measuring prerequisite relations, from Liang et al. 2015.

### Data
**CrowdComp Dataset**: 5 domains crowdsourced via Amazon Mechanical Turk:
- Meiosis, Public-key Cryptography, Parallel Postulate, Newton's Laws, Global Warming
- ~20 key concepts per domain, up to 400 connected target nodes, ~1,600 KC pairs total

**Course Dataset**: 2 educational domains:
- Computer Science (CS)
- Mathematics (MATH)
- Tab-separated format: "A\tB" means B is prerequisite for A
- Includes both positive (.edges) and negative (.edges_neg) examples

### Access
- **GitHub**: [harrylclc/RefD-dataset](https://github.com/harrylclc/RefD-dataset)
- Two folders: Course/ and CrowdComp/

### Format
TSV (.edges files), CSV (CrowdComp)

### License
**CC BY-NC-SA 4.0**

### Learner Performance Data
No (CrowdComp has crowd annotations but not learning performance).

### Verdict
**Useful smaller benchmark.** Good complement to AL-CPL with different domains. The CrowdComp dataset is interesting for its diverse non-CS topics.

---

## 5. University Course Dataset (CMU / Multi-University)

### Overview
Concept prerequisite pairs extracted from CS course syllabi across US universities. Introduced by Yang et al. (WSDM 2015) in "Concept Graph Learning from Educational Data."

### Data
- 1,008 manually annotated prerequisite concept pairs from CS courses
- Expanded to 2,520 pairs total (1,008 positive + 1,512 negative)
- Courses from MIT, Caltech, Princeton, CMU, and other universities
- Released under Creative Commons license

### Access
- Originally at `http://nyc.lti.cs.cmu.edu/teacher/dataset/` -- **now offline** (DNS not resolving)
- Referenced in many papers; may be obtainable by contacting CMU LTI researchers
- The ACE paper (JEDM 2024) also uses this as a benchmark

### Format
Not confirmed (likely TSV/CSV)

### License
Creative Commons (specific variant not confirmed)

### Learner Performance Data
No.

### Verdict
**Historically important but hard to obtain.** The original download URL is dead. May need to contact researchers or find a mirror.

---

## 6. LectureBank

### Overview
NLP education dataset with prerequisite chain annotations, from Yale LILY Lab (AAAI 2019).

### Data
- **7,499 lecture resources** (slides/PDFs) from 60 university courses
- **320 topics** in hierarchical taxonomy (taxonomy.csv)
- **208 core concepts** with pairwise prerequisite annotations
- **921 positive prerequisite pairs** (out of 42,750 total pairs)
- 5 domains: NLP, Machine Learning, AI, Deep Learning, Information Retrieval
- 1,221 vocabulary terms

### Access
- **GitHub**: [Yale-LILY/LectureBank](https://github.com/Yale-LILY/LectureBank)
- `alldata.tsv`: combined dataset
- `taxonomy.csv`: hierarchical topic tree
- Prerequisite annotations in structured format
- Lecture files via download script (links may be stale)

### Format
TSV, CSV

### License
Not explicitly specified in repo.

### Learner Performance Data
No.

### Verdict
**Good for NLP/ML domain.** The 208-concept prerequisite graph with 921 edges is a reasonable size. The hierarchical taxonomy adds structure. Narrow domain focus (NLP/ML/AI) limits generality.

---

## 7. MOOCCubeX (Tsinghua University)

### Overview
The largest educational knowledge graph dataset available. From Tsinghua's Knowledge Engineering Group, published at ACL 2020 and CIKM 2021.

### Data
- **4,216 courses**, 230,263 videos, 358,265 exercises
- **637,572 fine-grained concepts**
- Concept prerequisite relations for **psychology, mathematics, and computer science**
- **296 million behavioral records** from 3,330,294 students
- Course structures, video watching patterns, exercise attempts, comments

### Access
- **GitHub**: [THU-KEG/MOOCCubeX](https://github.com/THU-KEG/MOOCCubeX)
- Download script: `download_dataset.sh`
- Individual files via direct links to `lfs.aminer.cn`
- File sizes range from 613KB to 21GB

### Format
JSON and text files

### License
**GPL-3.0**

### Learner Performance Data
**Yes** -- extensive behavioral data (video watching, exercise attempts, comments) for 3.3M students.

### Verdict
**The gold standard for scale.** Massive dataset with both graph structure and learner behavior. The prerequisite relations for three domains (psychology, math, CS) are expert-refined. Main downsides: primarily Chinese MOOC content (from XueTangX), very large download, GPL license.

---

## 8. MOOC-Radar (Tsinghua University)

### Overview
Fine-grained knowledge repository for cognitive student modeling, also from THU-KEG.

### Data
- 2,513 exercises
- 14,226 students
- 12+ million behavioral records
- 5,600 fine-grained concepts

### Access
- **GitHub**: [THU-KEG/MOOC-Radar](https://github.com/THU-KEG/MOOC-Radar)

### Verdict
**Smaller complement to MOOCCubeX.** More focused on cognitive modeling with fine-grained concept annotations.

---

## 9. University of Illinois Course Prerequisites

### Overview
Structured course prerequisite data for every course at UIUC.

### Data
- All courses offered at UIUC (~8,589 sections in Fall 2019)
- Prerequisite relationships parsed from course descriptions
- Each row: course code, prerequisite count, prerequisite course codes

### Access
- **GitHub**: [illinois/prerequisites-dataset](https://github.com/illinois/prerequisites-dataset)
- Single CSV file, directly downloadable

### Format
CSV

### License
Not specified.

### Learner Performance Data
No.

### Verdict
**Real-world course-level prerequisite graph.** Unlike concept-level datasets, this captures actual university course dependencies. Good for testing at a different granularity level. Simple format.

---

## 10. freeCodeCamp Curriculum

### Overview
Open-source web development curriculum with hierarchical challenge structure.

### Structure
- **Hierarchy**: Superblocks -> Chapters -> Modules -> Blocks -> Challenges
- `curriculum/structure/curriculum.json` defines superblock ordering
- Each block has a `challengeOrder` array with sequenced challenges
- `meta.json` files define names and ordering within blocks

### Access
- **GitHub**: [freeCodeCamp/freeCodeCamp](https://github.com/freeCodeCamp/freeCodeCamp) (monorepo)
- Curriculum files in `curriculum/` directory
- Contributor docs: [contribute.freecodecamp.org/curriculum-file-structure/](https://contribute.freecodecamp.org/curriculum-file-structure/)

### Prerequisite Relationships
**No explicit prerequisites.** The curriculum uses linear sequencing (challengeOrder arrays) but does NOT define prerequisite relationships between superblocks, blocks, or challenges. The ordering implies progression but isn't modeled as a graph.

### Format
JSON (meta.json, curriculum.json), Markdown (challenges)

### License
BSD-3-Clause

### Learner Performance Data
No.

### Verdict
**Limited for our purposes.** The hierarchy is flat/linear, not a prerequisite graph. Could be used as a source corpus for inferring prerequisite relations, but the structure itself doesn't encode them.

---

## 11. Coursera Skills Graph

### Overview
Coursera has a proprietary "Skills Graph" that maps learners, content, and careers through a common skills taxonomy. A "Career Graph" connects workforce roles, skills, and learning content.

### Data
- Skills mapped to courses via ML model trained on instructor/SME/learner labels
- Structured skill taxonomy connecting skills to each other, to content, to careers, and to learners

### Access
**Not public.** The Skills Graph is proprietary infrastructure. No public API, no data download. Described in a [Medium engineering blog post](https://medium.com/coursera-engineering/courseras-skills-graph-helps-learners-find-the-right-content-to-reach-their-goals-b10418a05214) but not accessible.

### Verdict
**Not usable.** Interesting architecture but entirely proprietary.

---

## 12. edX / Open edX

### Overview
Open edX has a Course Blocks API for accessing course structure (xBlocks hierarchy).

### Access
- Course Blocks API documented at [openedx.atlassian.net](https://openedx.atlassian.net/wiki/spaces/AC/pages/29688043/Course+Blocks+API)
- Individual course structures available via API

### Prerequisite Relationships
The API exposes course structure (sections, subsections, units) but **no cross-course prerequisite graph**. Individual course internal structure is hierarchical but linear.

### Verdict
**Not useful for prerequisite graphs.** Course-internal structure only.

---

## 13. ACM/IEEE CS2023 Curriculum Guidelines

### Overview
The CS2023 Joint Task Force (ACM/IEEE-CS/AAAI) published updated computer science curricular guidelines with 17 knowledge areas and a competency model.

### Data
- 17 knowledge areas with knowledge units and topics
- Competency model framework
- Some inter-area prerequisite relationships mentioned (e.g., "software competency area is in part a prerequisite of other competency areas")

### Access
- Full report PDF: [CS2023 Report](https://ieeecs-media.computer.org/media/education/reports/CS2023.pdf)
- HTML version: [csed.acm.org](https://csed.acm.org/wp-content/uploads/2025/11/CS2023-Report.htm)
- Knowledge model intro: [PDF](https://csed.acm.org/wp-content/uploads/2024/01/Introduction-to-Knowledge-Model-v1.pdf)

### Format
**PDF/HTML only.** No machine-readable structured data (JSON, XML, or graph format) appears to be published.

### Prerequisite Relationships
Mentioned in prose but **not formalized as a graph**.

### Verdict
**Not directly usable.** Would require manual extraction from PDF to create a prerequisite graph. Could be a valuable project but significant effort.

---

## 14. Wikipedia/Wikidata as Knowledge Structure

### Wikipedia Clickstream
- Monthly dataset of (referrer, resource) navigation pairs: [dumps.wikimedia.org](https://dumps.wikimedia.org/other/clickstream/readme.html)
- Shows how users navigate between articles -- can be used as a proxy for prerequisite relationships
- Used in "Finding Prerequisite Relations using the Wikipedia Clickstream" (WWW 2019)
- The Metacademy-to-Wikipedia mapping on Figshare bridges expert prerequisites to Wikipedia navigation data

### Wikidata
- Rich ontology with `subclass of` (P279), `instance of` (P31), `part of` (P361) relationships
- **No dedicated "prerequisite" property** for educational concepts
- `field of study` and academic discipline hierarchies exist but aren't structured as learning prerequisites
- Could be queried via SPARQL for discipline hierarchies as a rough proxy

### DBpedia
- Structured extraction of Wikipedia infoboxes
- 400M+ facts, 3.7M+ entities
- Category hierarchies could serve as concept structure
- Used in educational KG research (DBpedia Spotlight for concept expansion)

### Verdict
**Useful as enrichment and proxy, not as direct prerequisite data.** The clickstream data is interesting for inferring prerequisites statistically. Wikidata/DBpedia provide concept hierarchies but not educational prerequisites.

---

## 15. Bloom's Taxonomy Annotated Learning Objectives

### Overview
21,380 learning objectives from 5,558 courses at an Australian university (Monash), manually labeled by Bloom's taxonomy cognitive level.

### Data
- 21,380 learning objectives
- 6 cognitive levels: Remember, Understand, Apply, Analyze, Evaluate, Create
- Binary annotations for each level

### Access
- **GitHub**: [SteveLEEEEE/EDM2022CLO](https://github.com/SteveLEEEEE/EDM2022CLO)
- CSV data file with text + labels
- Jupyter notebooks for ML/DL classification experiments
- Paper: "Automatic Classification of Learning Objectives Based on Bloom's Taxonomy" (EDM 2022)

### Format
CSV

### License
Not specified.

### Prerequisite Relationships
**No.** Bloom's levels are cognitive complexity levels, not prerequisite chains. However, they could complement a prerequisite graph by indicating the depth of knowledge required.

### Learner Performance Data
No.

### Verdict
**Complementary rather than primary.** Useful for understanding cognitive levels in curriculum design but doesn't provide prerequisite structure.

---

## 16. Open University Learning Analytics Dataset (OULAD)

### Overview
Anonymized data from 22 presentations of 7 modules at the Open University, widely used in learning analytics research.

### Data
- 32,593 students
- 7 modules, 22 presentations
- Assessments, VLE interactions (click data), demographics, registrations

### Access
- [analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset)
- [Kaggle](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)
- [UCI ML Repository](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset)

### Format
CSV tables

### Prerequisite Relationships
**No.** The dataset focuses on student behavior within courses, not between-course prerequisites.

### Learner Performance Data
**Yes** -- extensive assessment results and VLE interaction data.

### Verdict
**Not useful for prerequisite graphs** but excellent for learner modeling research. Could be paired with a prerequisite graph dataset for richer experiments.

---

## Summary: Best Datasets for Testing Knowledge Mapping Algorithms

### Tier 1: Ready to Use (clean prerequisite graphs, downloadable now)

| Dataset | Concepts | Prereq Edges | Domains | URL |
|---------|----------|-------------|---------|-----|
| **AL-CPL** | 586 | 2,002 | Data Mining, Geometry, Physics, Precalculus | [GitHub](https://github.com/harrylclc/AL-CPL-dataset) |
| **Metacademy** | 141 | 331 (direct) | ML, Statistics, Linear Algebra | [GitHub](https://github.com/metacademy/metacademy-content) |
| **RefD** | ~200+ | ~1,600+ | CS, Math, 5 science topics | [GitHub](https://github.com/harrylclc/RefD-dataset) |
| **LectureBank** | 208 | 921 | NLP, ML, AI, DL, IR | [GitHub](https://github.com/Yale-LILY/LectureBank) |

### Tier 2: Larger Scale (more setup required)

| Dataset | Scale | Has Prereqs | Has Learner Data | URL |
|---------|-------|-------------|-----------------|-----|
| **MOOCCubeX** | 637K concepts | Yes (3 domains) | Yes (3.3M students) | [GitHub](https://github.com/THU-KEG/MOOCCubeX) |
| **UIUC Courses** | ~8,500 courses | Yes (course-level) | No | [GitHub](https://github.com/illinois/prerequisites-dataset) |
| **MOOC-Radar** | 5,600 concepts | Unknown | Yes (14K students) | [GitHub](https://github.com/THU-KEG/MOOC-Radar) |

### Tier 3: Complementary / Enrichment

| Dataset | What It Adds | URL |
|---------|-------------|-----|
| **Bloom's Taxonomy CLO** | Cognitive level annotations | [GitHub](https://github.com/SteveLEEEEE/EDM2022CLO) |
| **Wikipedia Clickstream** | Navigation-based prerequisite proxy | [Wikimedia Dumps](https://dumps.wikimedia.org/other/clickstream/) |
| **Metacademy-Wikipedia mapping** | Bridges expert graph to Wikipedia | [Figshare](https://figshare.com/articles/dataset/Metacademy-prereq-pairs-transoformed-to-wiki_csv/7799774) |
| **OULAD** | Learner performance data | [OU Analyse](https://analyse.kmi.open.ac.uk/open-dataset) |

### Recommended Starting Points for the application layer/Amygdala

1. **AL-CPL dataset** -- best balance of size, quality, and domain diversity. Four domains with expert-validated prerequisites in a clean format. Can directly test whether amygdala's embedding similarity correlates with expert-judged prerequisite relationships.

2. **Metacademy content repo** -- small but well-structured graph with rich metadata (descriptions, learning goals, reasons for prerequisites). Clone the repo, parse `dependencies.txt` files, and reconstruct the graph.

3. **UIUC prerequisites** -- real-world course-level prerequisite graph. Different granularity from concept-level datasets, useful for testing at curriculum scale.

4. **MOOCCubeX** -- if we need scale and learner behavior data. The prerequisite relations for math/psychology/CS are expert-refined, and the behavioral data could validate whether prerequisite ordering predicts learning success.

---

## Key Papers

- Yang et al. (2015). "Concept Graph Learning from Educational Data." WSDM 2015. [PDF](https://www.cs.cmu.edu/~hanxiaol/publications/yang-wsdm15.pdf)
- Liang et al. (2015). "Measuring Prerequisite Relations Among Concepts." [GitHub](https://github.com/harrylclc/RefD-dataset)
- Li et al. (2019). "What Should I Learn First: Introducing LectureBank." AAAI 2019. [ar5iv](https://ar5iv.labs.arxiv.org/html/1811.12181)
- Pan et al. (2017). "Prerequisite Relation Learning for Concepts in MOOCs." ACL 2017. [PDF](https://keg.cs.tsinghua.edu.cn/jietang/publications/ACL17-Pan-et-al-Prerequisite-Relationship-MOOCs.pdf)
- Roy et al. (2019). "Inferring Concept Prerequisite Relations from Online Educational Resources." [arXiv](https://arxiv.org/pdf/1811.12640)
- Survey: "Prerequisite Relation Learning: A Survey and Outlook." ACM Computing Surveys, 2025. [ACM DL](https://dl.acm.org/doi/10.1145/3733593)
- ACE paper (JEDM 2024): "AI-Assisted Construction of Educational Knowledge Graphs with Prerequisite Relations." [JEDM](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/737)
