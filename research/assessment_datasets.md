# Datasets for Validating Knowledge Mapping with Prerequisite Graphs

**Date**: 2026-03-19
**Goal**: Find datasets with (1) a concept/skill DAG with prerequisites, (2) learner assessment data, (3) multiple learners — to validate an entropy-based adaptive probing algorithm (~15 questions) with belief propagation on a prerequisite graph.

---

## Tier 1: Excellent Fit (have both prerequisite graph AND learner responses)

### 1. Junyi Academy (2015 dataset via EduData)

**Why it's the best match**: Explicit prerequisite graph between exercises + millions of learner interactions.

- **Scale**: 837 exercises, ~250,000 students, ~26 million interactions (math curriculum, elementary through high school)
- **Prerequisite graph**: YES — each exercise has a `prerequisite` field pointing to its parent in the knowledge map, plus `h_position`/`v_position` for the knowledge map layout. This forms a DAG.
- **Learner data**: Full interaction logs — which exercises each student attempted, correctness, time taken, hints used
- **Format**: CSV files via EduData Python package
- **Download**:
  - `pip install EduData` then `edudata download junyi`
  - Also on Kaggle: https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy (the 2018-2019 version with 72,630 students / 16.2M attempts, but may lack the prerequisite field)
  - PSLC DataShop: https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198 (the 2015 version with prerequisites)
- **Key files**: `junyi_Exercise_table.csv` (exercise metadata + prerequisites), `junyi_ProblemLog_original.csv` (interaction logs)
- **License**: CC BY-NC-SA 4.0. Cite: Chang, Hsu & Chen, "Modeling Exercise Relationships in E-Learning: A Unified Approach," EDM 2015.
- **Suitability**: EXCELLENT. Can reconstruct the DAG from prerequisite field, then use learner interaction logs to determine ground-truth mastery per exercise, then simulate the probing algorithm. 837 exercises is a realistic graph size.

### 2. XES3G5M (NeurIPS 2023)

**Why it fits**: Hierarchical KC tree with explicit prerequisite routes + large learner interaction data.

- **Scale**: 865 leaf knowledge components (KCs) organized in a hierarchical tree, 7,652 questions, 18,066 students, 5,549,635 interactions
- **Prerequisite graph**: YES — KCs are organized in a tree structure with "KC routes" (paths from root to leaf). Each question maps to one or more leaf KCs. The tree encodes prerequisite/subsumption relationships.
- **Learner data**: Per-student sequences of question attempts with correctness, timestamps, and repeat indicators
- **Format**: CSV (interaction sequences), JSON (question metadata, KC route maps, RoBERTa embeddings)
- **Download**: Google Drive link from https://github.com/ai4ed/XES3G5M (MIT license)
- **Domain**: 3rd-grade math (Chinese curriculum)
- **Suitability**: EXCELLENT. The KC tree is a natural prerequisite structure. With 865 KCs and 18K students, this is ideal for testing whether ~15 questions can determine mastery across the tree. The hierarchy means belief propagation upward/downward is meaningful.

### 3. Eedi / NeurIPS 2020 Education Challenge

**Why it fits**: 4-level topic ontology tree + massive learner response data.

- **Scale**: 27,613 questions, 118,971 students, 15,867,850 answers. Questions tagged with a 4-level KC ontology: Subject > Topic > Subtopic > Objective (57 leaf KCs in the Tasks 3/4 subset)
- **Prerequisite graph**: PARTIAL — the 4-level ontology tree provides a hierarchical structure (Subject > Area > Topic > Subtopic). Not explicitly prerequisite-labeled, but the hierarchy is pedagogically meaningful (e.g., "Algebra" > "Factorizing" > "Factorizing into a single bracket"). Won the 2021 Best Publicly Available Educational Data Set Prize.
- **Learner data**: Full response logs — which questions each student answered, which option they chose, correctness
- **Format**: CSV files
- **Download**: https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip
- **Paper**: https://arxiv.org/abs/2007.12061
- **Suitability**: GOOD. The ontology tree provides structure for belief propagation. 57 leaf KCs is manageable — could test whether 15 questions suffice. The full dataset with 27K questions and 119K students gives ample ground truth. Would need to construct explicit prerequisite edges (not provided, but inferrable from the tree + learning patterns).

### 4. FrcSub (Tatsuoka Fraction Subtraction)

**Why it fits**: Classic Q-matrix dataset with expert-defined skill dependencies — small but gold-standard.

- **Scale**: 536 students, 20 questions, 8 underlying cognitive skills (attributes)
- **Prerequisite graph**: IMPLICIT — the 8 skills have logical dependencies (e.g., "simplify before subtracting" requires "convert whole number to fraction"). The Q-matrix maps each question to required skills. Researchers have derived prerequisite orderings.
- **Learner data**: Full binary response matrix (536 x 20)
- **Format**: Built into R packages (`CDM`, `GDINA`)
- **Download**: `install.packages("CDM")` in R, then `data(fraction.subtraction.data)` and `data(fraction.subtraction.qmatrix)`
- **Suitability**: GOOD for initial proof-of-concept. Very small (8 skills, 20 items), but the Q-matrix + response data lets you test belief propagation directly. The explicit skill dependencies are well-studied. Too small for production validation but perfect for algorithm debugging.

### 5. MOOCCubeX (Tsinghua, 2021)

**Why it fits**: Large-scale concept prerequisite graphs + student exercise attempts.

- **Scale**: 637,572 concepts, 358,265 exercises, 3,330,294 students, 4,216 courses
- **Prerequisite graph**: YES — explicit concept prerequisite JSON files for Psychology, Mathematics, and Computer Science domains. Built via prediction + human annotation (F1=0.905 for prerequisite discovery).
- **Learner data**: Exercise attempt logs (21GB), video watching logs (3GB), user profiles
- **Format**: JSON (concepts, prerequisites), tab-separated text (relations), JSON (user behavior)
- **Download**: https://github.com/THU-KEG/MOOCCubeX — automated via `download_dataset.sh`
- **Suitability**: EXCELLENT for scale testing, but the prerequisite graphs are algorithmically derived (not curriculum-designed), so they may be noisier than hand-curated graphs. The sheer size (637K concepts) would require subgraph extraction. Good for testing scalability of the probing algorithm.

---

## Tier 2: Partial Fit (have learner data, but prerequisite graph needs construction)

### 6. ASSISTments (2009, 2012, 2015, 2017)

- **Scale**: Multiple versions. 2009: 4,217 students, 346,860 interactions, 110 skills. 2012: 27,066 students, 2.5M interactions. 2015: 19,917 students, 708,631 interactions, 100 KCs.
- **Prerequisite graph**: NOT PROVIDED. Skills are tagged but no explicit prerequisite structure. The data does include "prerequisite and post-requisite skills" as problem features in some versions.
- **Learner data**: Rich interaction logs — correctness, hints, time, skill tags
- **Download**: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data (and variants)
- **Suitability**: FAIR. Would need to construct a prerequisite graph externally (e.g., from math curriculum standards). The learner data is excellent and widely benchmarked, but without a prerequisite DAG, half the validation target is missing. Could pair with a hand-built math skill graph.

### 7. EdNet (Riiid/Santa, 2019)

- **Scale**: 784,309 students, 131M interactions, 13,169 questions tagged with 293 skill types, 1,021 lectures
- **Prerequisite graph**: NO. Skills are tagged as integer IDs with no explicit dependency structure.
- **Learner data**: Massive — 4 hierarchical levels of granularity (KT1-KT4), from simple Q/A logs to full behavioral traces
- **Format**: CSV files (1.2-6.4GB per level)
- **Download**: Hosted on Google Drive, links from https://github.com/riiid/ednet
- **Domain**: TOEIC English test preparation
- **Suitability**: FAIR. Huge scale but no prerequisite graph. Would need to construct skill dependencies for TOEIC English (e.g., grammar prerequisites). The 293 skill tags could potentially be organized into a DAG by an English teaching expert, but this is non-trivial.

### 8. KDD Cup 2010 (Algebra 2005/2006, Bridge to Algebra 2006/2007)

- **Scale**: Algebra2005: 574 students, 809,694 interactions, 112 KCs. Bridge2006: 1,146 students, 3.7M interactions, 493 KCs.
- **Prerequisite graph**: NOT PROVIDED explicitly, but the KC models map steps to knowledge components. Carnegie Learning's Cognitive Tutor has internal prerequisite structures that aren't published.
- **Learner data**: Detailed step-level interaction data from Cognitive Tutors
- **Download**: https://pslcdatashop.web.cmu.edu/KDDCup/ and https://kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Data
- **Suitability**: FAIR. Well-studied dataset with rich learner data, but prerequisite graph would need to be constructed from math curriculum knowledge.

---

## Tier 3: Concept Prerequisite Graphs Only (no learner assessment data)

### 9. AL-CPL Dataset (Liang et al., 2018)

- **Scale**: 4 domains — Data Mining (120 concepts, 826 pairs), Geometry (89 concepts, 1,681 pairs), Physics (153 concepts, 1,962 pairs), Precalculus (224 concepts, 2,060 pairs)
- **Prerequisite graph**: YES — expert-validated prerequisite pairs forming strict partial orders. Both positive and negative pairs provided (useful for training).
- **Learner data**: NONE — this is purely a concept prerequisite dataset
- **Download**: https://github.com/harrylclc/AL-CPL-dataset (CC BY-NC-SA 4.0)
- **Suitability**: GOOD for graph structure testing only. Could generate synthetic learner data using belief propagation on these real prerequisite graphs (simulate learners who have learned up to various points in the DAG, then test the probing algorithm). The 89-224 concept range is realistic for our use case.

### 10. Metacademy

- **Scale**: 141 concepts (Machine Learning, Statistics, Linear Algebra) with expert-curated prerequisite edges
- **Prerequisite graph**: YES — hand-built by ML researchers, each concept has described prerequisites
- **Learner data**: NONE
- **Download**: Figshare: https://figshare.com/articles/dataset/Metacademy-prereq-pairs-transoformed-to-wiki_csv/7799774
- **Suitability**: USEFUL for graph structure. 141 concepts is close to our curriculum sizes (Greece 67, Rome 55, Sicily 70). Would need synthetic learners. The ML/Stats domain is well-understood enough to validate whether generated prerequisite orderings make sense.

---

## Tier 4: Not Suitable or Not Available

### 11. Khan Academy
- **Status**: Public API shut down July 2020. No public prerequisite graph or learner data available. The Knowledge Map feature (which showed skill connections) was removed in 2013. Internal data is not accessible.

### 12. ALEKS (Knowledge Space Theory)
- **Status**: No public datasets. ALEKS's knowledge spaces are proprietary. The theoretical framework (Doignon & Falmagne) is well-documented in books and papers, but actual ALEKS assessment data has never been published. A 2021 paper discusses "ALEKS and its data" but the data itself remains private.

### 13. Duolingo
- **Published datasets**:
  - Half-Life Regression: 13M learning traces (word-level spaced repetition data). Download: https://github.com/duolingo/halflife-regression
  - SLAM 2018: 7M tokens from 6K+ users learning English/Spanish/French over 30 days. Download: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO
- **Prerequisite graph**: NONE. Neither dataset includes a skill tree or prerequisite structure. The skill tree exists internally but isn't published.
- **Suitability**: NOT SUITABLE for our use case. Good for spaced repetition research but lacks the graph structure we need.

### 14. Open Learning Initiative (OLI, Carnegie Mellon)
- **Status**: OLI courses use learning objectives and Q-matrices internally, but no public dataset with prerequisite structures has been published. DataShop hosts some OLI data (e.g., OLI Engineering Statics: 333 students, 1,223 exercise tags) but without explicit prerequisite graphs.

### 15. Carnegie Learning Cognitive Tutor
- **Status**: Data available via PSLC DataShop with KC models at multiple granularities (KTracedSkills, SubSkills, Rules), but no published prerequisite/dependency structures between KCs. The internal tutor models are proprietary.

---

## Simulation & Tooling Resources

### R Packages for Knowledge Space Theory
- **kst**: Implements knowledge structures, surmise relations, and deterministic assessment. Can construct knowledge spaces from prerequisite orderings and simulate assessment. https://cran.r-project.org/package=kst
- **DAKS**: Data Analysis in Knowledge Spaces. Includes simulation tools for generating response patterns from knowledge structures with BLIM (Basic Local Independence Model). Can simulate learner populations given a prerequisite quasi-order. https://cran.r-project.org/package=DAKS
- **CDM** / **GDINA**: Cognitive Diagnosis Modeling packages. Include FrcSub dataset and Q-matrix. Support simulating learner responses given skill mastery profiles.

### Python Packages
- **EduData** (`pip install EduData`): Download and preprocess educational datasets (Junyi, ASSISTments, EdNet, etc.). https://github.com/bigdata-ustc/EduData
- **EduCAT**: Computerized Adaptive Testing implementations (selection strategies, cognitive diagnosis models). Uses ASSISTments. https://github.com/bigdata-ustc/EduCAT
- **pyKT**: Knowledge Tracing toolkit supporting 10+ datasets with standardized preprocessing. https://pykt-toolkit.readthedocs.io/

### Synthetic Learner Generation
- **DAKS R package**: Can generate synthetic response matrices from any knowledge structure (prerequisite quasi-order) using the BLIM model with configurable careless error and lucky guess rates.
- **DKT Synthetic Data**: Chris Piech's Deep Knowledge Tracing repo includes synthetic data (2,000 simulated learners, 50 exercises). https://github.com/chrispiech/DeepKnowledgeTracing/tree/master/data/synthetic
- **Agent4Edu / LLM-based simulation**: Recent (2025) approaches use LLMs to simulate realistic learner behaviors given persona profiles. More complex than needed but interesting for future work.

---

## Recommended Validation Strategy

### Phase 1: Proof of Concept (small, controlled)
Use **FrcSub** (8 skills, 20 items, 536 real learners) or **AL-CPL Precalculus** (224 concepts, expert prerequisite graph + synthetic learners via DAKS simulation).

### Phase 2: Realistic Scale
Use **Junyi Academy** (837 exercises with explicit prerequisite DAG + 250K real learners). This is the single best dataset for our purpose — real prerequisite graph, real learner responses, manageable scale.

### Phase 3: Large Scale
Use **XES3G5M** (865 KCs in a tree + 18K students) for testing on a larger hierarchical structure, or **MOOCCubeX** (subsample a domain-specific prerequisite graph + learner data).

### Phase 4: Domain Transfer
Test on **Eedi** (57 KCs with topic ontology + 119K students) to validate that the algorithm works on a different kind of hierarchical structure (topic tree rather than prerequisite DAG).

### For each dataset, the validation would be:
1. Extract the prerequisite DAG
2. Determine ground-truth mastery per learner per concept (from their full interaction history)
3. Run the probing algorithm: select ~15 questions via entropy-based selection + belief propagation
4. Compare inferred mastery state to ground truth
5. Measure: accuracy, coverage (% of concepts correctly classified), efficiency (questions needed to reach X% accuracy)

---

## Sources

- Khan Academy API: https://support.khanacademy.org/hc/en-us/community/posts/34654466601741-Khan-Academy-API
- ALEKS Knowledge Space Theory: https://www.aleks.com/about_aleks/knowledge_space_theory
- ASSISTments Data: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
- EdNet GitHub: https://github.com/riiid/ednet
- Junyi Academy Kaggle: https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy
- XES3G5M GitHub: https://github.com/ai4ed/XES3G5M
- Eedi / NeurIPS 2020: https://arxiv.org/abs/2007.12061
- AL-CPL Dataset: https://github.com/harrylclc/AL-CPL-dataset
- Metacademy Figshare: https://figshare.com/articles/dataset/Metacademy-prereq-pairs-transoformed-to-wiki_csv/7799774
- MOOCCubeX: https://github.com/THU-KEG/MOOCCubeX
- Duolingo HLR: https://github.com/duolingo/halflife-regression
- Duolingo SLAM: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO
- EduData: https://github.com/bigdata-ustc/EduData
- EduCAT: https://github.com/bigdata-ustc/EduCAT
- pyKT: https://pykt-toolkit.readthedocs.io/
- PSLC DataShop: https://pslcdatashop.web.cmu.edu/
- KDD Cup 2010: https://pslcdatashop.web.cmu.edu/KDDCup/
- DAKS R Package: https://cran.r-project.org/package=DAKS
- CDM R Package (FrcSub): https://rdrr.io/cran/CDM/man/fraction.subtraction.data.html
- CAT Survey: https://arxiv.org/abs/2404.00712
