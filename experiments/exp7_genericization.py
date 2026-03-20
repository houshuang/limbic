"""Experiment 7: Genericization Quality Evaluation

Tests whether amygdala's text genericization (stripping numbers, dates, URLs)
improves cross-source matching for semantically identical claims that differ
in surface details.

RESEARCH.md question: "Should genericization strip domain-specific terms
(school names, researcher names) before embedding?"

Method:
  1. 50 same-meaning claim pairs with different surface details
     (numbers, dates, proper nouns, URLs, percentages, currency, mixed)
  2. 30 unrelated control pairs
  3. Cosine similarity computed three ways:
     a. Raw (no genericization)
     b. Current regex genericization (strips numbers, dates, URLs, %, $)
     c. NER-based genericization (if spaCy available — skipped if not)
  4. Metrics: mean cosine for same-meaning, mean cosine for unrelated,
     discrimination gap, per-category breakdown.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np


from amygdala import EmbeddingModel

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Test data: 50 same-meaning pairs across 5 categories
# ---------------------------------------------------------------------------

CATEGORIES = {
    "numbers": "Number differences only",
    "dates": "Date differences only",
    "proper_nouns": "Proper noun / institution name differences",
    "urls_refs": "URL / reference differences",
    "mixed": "Mixed differences (realistic)",
}

SAME_MEANING_PAIRS = [
    # --- NUMBERS (10 pairs) ---
    {
        "category": "numbers",
        "a": "The program served 1,500 students across 12 schools.",
        "b": "The initiative reached 2,300 participants across 8 schools.",
    },
    {
        "category": "numbers",
        "a": "Class sizes were reduced to 18 students per teacher.",
        "b": "Class sizes were reduced to 24 students per teacher.",
    },
    {
        "category": "numbers",
        "a": "Reading scores improved by 15% over three years.",
        "b": "Reading scores improved by 23% over five years.",
    },
    {
        "category": "numbers",
        "a": "The budget allocated $4.2 million for teacher training.",
        "b": "The budget allocated $7.8 million for teacher training.",
    },
    {
        "category": "numbers",
        "a": "Dropout rates fell from 12% to 6% in participating districts.",
        "b": "Dropout rates fell from 18% to 9% in participating districts.",
    },
    {
        "category": "numbers",
        "a": "The intervention covered 340 classrooms in 45 buildings.",
        "b": "The intervention covered 520 classrooms in 67 buildings.",
    },
    {
        "category": "numbers",
        "a": "Average test performance rose by 8 percentile points.",
        "b": "Average test performance rose by 14 percentile points.",
    },
    {
        "category": "numbers",
        "a": "Student attendance increased from 85% to 93%.",
        "b": "Student attendance increased from 78% to 91%.",
    },
    {
        "category": "numbers",
        "a": "The scholarship fund distributed $350,000 to 120 recipients.",
        "b": "The scholarship fund distributed $580,000 to 210 recipients.",
    },
    {
        "category": "numbers",
        "a": "Teacher retention improved by 22 percentage points over the period.",
        "b": "Teacher retention improved by 31 percentage points over the period.",
    },

    # --- DATES (10 pairs) ---
    {
        "category": "dates",
        "a": "The curriculum reform was implemented in September 2021.",
        "b": "The curriculum reform was implemented in January 2023.",
    },
    {
        "category": "dates",
        "a": "A longitudinal study from 2018 to 2022 tracked student outcomes.",
        "b": "A longitudinal study from 2020 to 2024 tracked student outcomes.",
    },
    {
        "category": "dates",
        "a": "The new standards were adopted on 2019-03-15.",
        "b": "The new standards were adopted on 2022-09-01.",
    },
    {
        "category": "dates",
        "a": "Since 2015, the district has invested in digital learning tools.",
        "b": "Since 2019, the district has invested in digital learning tools.",
    },
    {
        "category": "dates",
        "a": "The pilot program launched in March 2020 and concluded in June 2021.",
        "b": "The pilot program launched in August 2022 and concluded in November 2023.",
    },
    {
        "category": "dates",
        "a": "Results from the 2017 assessment cycle showed steady improvement.",
        "b": "Results from the 2023 assessment cycle showed steady improvement.",
    },
    {
        "category": "dates",
        "a": "The school board approved the plan in February 2022.",
        "b": "The school board approved the plan in October 2024.",
    },
    {
        "category": "dates",
        "a": "Data collected between January 2019 and December 2020 confirmed the trend.",
        "b": "Data collected between March 2021 and August 2023 confirmed the trend.",
    },
    {
        "category": "dates",
        "a": "The 2016-2019 strategic plan emphasized STEM education.",
        "b": "The 2021-2024 strategic plan emphasized STEM education.",
    },
    {
        "category": "dates",
        "a": "Annual reviews since 2014 have consistently recommended more funding.",
        "b": "Annual reviews since 2020 have consistently recommended more funding.",
    },

    # --- PROPER NOUNS / INSTITUTION NAMES (10 pairs) ---
    {
        "category": "proper_nouns",
        "a": "According to NTNU research, test scores improved significantly.",
        "b": "A University of Bergen study found assessment results rose substantially.",
    },
    {
        "category": "proper_nouns",
        "a": "The Oslo School District implemented a new reading curriculum.",
        "b": "The Trondheim School District implemented a new reading curriculum.",
    },
    {
        "category": "proper_nouns",
        "a": "Professor Hansen's team at UiO developed the assessment framework.",
        "b": "Dr. Johansen's group at NTNU developed the evaluation framework.",
    },
    {
        "category": "proper_nouns",
        "a": "Nordland County allocated additional funds for special education.",
        "b": "Vestland County allocated additional funds for special education.",
    },
    {
        "category": "proper_nouns",
        "a": "The McKinsey report recommended consolidating rural schools.",
        "b": "The Deloitte analysis recommended consolidating rural schools.",
    },
    {
        "category": "proper_nouns",
        "a": "Finland's education model has inspired reforms in Norway.",
        "b": "Singapore's education model has inspired reforms in Denmark.",
    },
    {
        "category": "proper_nouns",
        "a": "Research published in Nature Education confirmed the benefits of active learning.",
        "b": "A study in the Journal of Educational Psychology confirmed the benefits of active learning.",
    },
    {
        "category": "proper_nouns",
        "a": "The PISA rankings showed Scandinavian students excelling in reading.",
        "b": "The TIMSS results showed Nordic students excelling in reading.",
    },
    {
        "category": "proper_nouns",
        "a": "Stavanger municipality introduced free school meals for all primary students.",
        "b": "Kristiansand municipality introduced free school meals for all primary students.",
    },
    {
        "category": "proper_nouns",
        "a": "The Bill & Melinda Gates Foundation funded the literacy initiative.",
        "b": "The Wellcome Trust funded the literacy initiative.",
    },

    # --- URLS / REFERENCES (10 pairs) ---
    {
        "category": "urls_refs",
        "a": "The full report is available at https://education.gov/report-2023.pdf.",
        "b": "The complete findings can be found at https://www.udir.no/research/outcomes.html.",
    },
    {
        "category": "urls_refs",
        "a": "For methodology details, see https://arxiv.org/abs/2301.12345.",
        "b": "For methodology details, see https://doi.org/10.1016/j.edurev.2023.100456.",
    },
    {
        "category": "urls_refs",
        "a": "Contact the team at admin@schoolproject.org for more information.",
        "b": "Contact the team at info@education-reform.no for more information.",
    },
    {
        "category": "urls_refs",
        "a": "The data dashboard is hosted at https://dashboard.school-metrics.com/2023/.",
        "b": "The analytics portal is available at https://stats.utdanning.no/results/.",
    },
    {
        "category": "urls_refs",
        "a": "As documented in Smith et al. (2021), collaborative learning improves outcomes.",
        "b": "As shown by Larsen & Berg (2023), collaborative learning improves outcomes.",
    },
    {
        "category": "urls_refs",
        "a": "The curriculum guide can be downloaded from https://resources.edu.gov/curriculum-v3.pdf.",
        "b": "The teaching framework is posted at https://www.skoleverket.se/frameworks/latest.",
    },
    {
        "category": "urls_refs",
        "a": "Registration is open at https://conference.edtech.io/register?year=2023.",
        "b": "Sign up at https://nkul.ntnu.no/registration/2024.",
    },
    {
        "category": "urls_refs",
        "a": "Source code is available at https://github.com/edu-project/analysis.",
        "b": "The implementation is published at https://gitlab.com/school-tools/evaluation.",
    },
    {
        "category": "urls_refs",
        "a": "See Figure 3 in Johnson (2019) for the trend visualization.",
        "b": "See Table 5 in Olsen (2022) for the trend summary.",
    },
    {
        "category": "urls_refs",
        "a": "Details in the appendix at https://nsd.no/supplements/study-A.pdf.",
        "b": "Supporting materials at https://opendata.no/supplements/research-B.pdf.",
    },

    # --- MIXED (10 pairs) ---
    {
        "category": "mixed",
        "a": "The program served 1,500 students in Oslo in 2023.",
        "b": "The initiative reached 2,300 participants in Bergen in 2024.",
    },
    {
        "category": "mixed",
        "a": "According to NTNU research, test scores improved by 15%.",
        "b": "A university study found assessment results rose by 23%.",
    },
    {
        "category": "mixed",
        "a": "The $3.5 million grant from the Norwegian Research Council funded 12 projects in 2022.",
        "b": "The €2.8 million award from the Swedish Research Council funded 9 projects in 2024.",
    },
    {
        "category": "mixed",
        "a": "A 2019 report by McKinsey (https://mckinsey.com/edu-report) found 78% of schools underperforming.",
        "b": "A 2023 study by BCG (https://bcg.com/school-analysis) found 65% of schools underperforming.",
    },
    {
        "category": "mixed",
        "a": "Oslo Kommune invested $12 million in 2021 to renovate 35 school buildings.",
        "b": "Bergen Kommune invested $8 million in 2023 to renovate 22 school buildings.",
    },
    {
        "category": "mixed",
        "a": "Professor Eriksen at UiT found that 45% of students preferred online exams in a 2020 survey.",
        "b": "Dr. Andersen at UiB found that 62% of students preferred online exams in a 2023 survey.",
    },
    {
        "category": "mixed",
        "a": "The Horizon 2020 project (grant #789456) trained 500 teachers across 6 countries by December 2022.",
        "b": "The Erasmus+ initiative (ref: 2023-EDU-1234) trained 320 teachers across 4 countries by June 2024.",
    },
    {
        "category": "mixed",
        "a": "Vestland County's 2019-2022 plan allocated 18% of the budget to digital infrastructure.",
        "b": "Rogaland County's 2021-2025 plan allocated 24% of the budget to digital infrastructure.",
    },
    {
        "category": "mixed",
        "a": "A randomized trial at Stanford (N=1,200) showed a 0.35 SD improvement (p<0.001) in 2018.",
        "b": "An experimental study at Harvard (N=800) showed a 0.42 SD improvement (p<0.01) in 2022.",
    },
    {
        "category": "mixed",
        "a": "The OECD Education at a Glance 2022 report (https://oecd.org/education/eag-2022.htm) ranked Norway 8th.",
        "b": "The UNESCO Global Education Monitor 2024 (https://unesco.org/gem-2024) ranked Sweden 5th.",
    },
]

# --- 30 UNRELATED control pairs ---
UNRELATED_PAIRS = [
    {
        "a": "The school implemented a new reading curriculum for third graders.",
        "b": "Global shipping costs have risen sharply due to fuel price increases.",
    },
    {
        "a": "Teacher salaries were increased to attract better candidates.",
        "b": "The volcanic eruption disrupted air traffic across northern Europe.",
    },
    {
        "a": "Students who received tutoring showed improved math performance.",
        "b": "New regulations require electric vehicles to have audible warning sounds.",
    },
    {
        "a": "The district invested in modern science laboratory equipment.",
        "b": "Coffee production in Brazil fell due to unexpected frost damage.",
    },
    {
        "a": "Parental involvement programs led to better homework completion rates.",
        "b": "The spacecraft successfully entered orbit around Mars.",
    },
    {
        "a": "After-school programs reduced juvenile crime in the neighborhood.",
        "b": "The recipe calls for two cups of flour and one egg.",
    },
    {
        "a": "Bilingual education programs improved outcomes for immigrant students.",
        "b": "The bridge construction project is three months behind schedule.",
    },
    {
        "a": "Early childhood education reduces the achievement gap for disadvantaged children.",
        "b": "The new smartphone model features a triple-camera system.",
    },
    {
        "a": "Standardized testing has been criticized for narrowing the curriculum.",
        "b": "The fishing industry reported record catches of Atlantic cod.",
    },
    {
        "a": "School meal programs improve concentration and academic performance.",
        "b": "The cryptocurrency market experienced extreme volatility this quarter.",
    },
    {
        "a": "Interactive whiteboards have replaced traditional chalkboards in most classrooms.",
        "b": "The marathon runner completed the course in under three hours.",
    },
    {
        "a": "Professional development workshops help teachers adopt new pedagogical methods.",
        "b": "The hotel chain announced plans to expand into the Asian market.",
    },
    {
        "a": "Smaller class sizes allow for more individualized instruction.",
        "b": "The new highway bypass will reduce traffic congestion downtown.",
    },
    {
        "a": "School libraries are transitioning to digital-first resource collections.",
        "b": "The pharmaceutical company announced positive trial results for the vaccine.",
    },
    {
        "a": "Music education has been linked to improved spatial reasoning skills.",
        "b": "Drought conditions have severely impacted wheat yields in the region.",
    },
    {
        "a": "Online learning platforms expanded access to rural communities.",
        "b": "The fashion industry is moving toward sustainable fabric sourcing.",
    },
    {
        "a": "Student loan debt has become a significant barrier to homeownership.",
        "b": "The new telescope captured images of a previously unknown galaxy.",
    },
    {
        "a": "Physical education requirements promote lifelong health habits.",
        "b": "The railway company ordered fifty new high-speed train cars.",
    },
    {
        "a": "Gifted education programs identify and nurture exceptional talent.",
        "b": "The archaeological dig uncovered pottery fragments from the Bronze Age.",
    },
    {
        "a": "Inclusive classroom practices benefit both disabled and non-disabled students.",
        "b": "The airline introduced new direct routes to three Asian cities.",
    },
    {
        "a": "Vocational training programs address the skills gap in manufacturing.",
        "b": "The zoo welcomed a pair of endangered giant pandas.",
    },
    {
        "a": "Peer mentoring programs improve social skills and reduce bullying.",
        "b": "The renewable energy sector added more jobs than fossil fuels this year.",
    },
    {
        "a": "School counselors play a critical role in student mental health support.",
        "b": "The art auction set a new record for a contemporary painting.",
    },
    {
        "a": "Flipped classroom models allow more time for hands-on activities.",
        "b": "The deep-sea expedition discovered three new species of jellyfish.",
    },
    {
        "a": "STEM education initiatives aim to close the gender gap in engineering.",
        "b": "The city council approved the construction of a new sports arena.",
    },
    {
        "a": "Homework load has been debated as a factor in student stress.",
        "b": "The orbiting satellite detected unusual radio emissions from a distant star.",
    },
    {
        "a": "Charter schools offer an alternative governance structure within public education.",
        "b": "The bakery introduced a line of gluten-free pastries.",
    },
    {
        "a": "Language immersion programs accelerate second-language acquisition.",
        "b": "The mining company received approval to extract lithium deposits.",
    },
    {
        "a": "Outdoor education programs foster environmental awareness and resilience.",
        "b": "The new CPU architecture delivers a forty percent performance improvement.",
    },
    {
        "a": "Gap year programs can improve student motivation and academic focus.",
        "b": "The cargo ship ran aground blocking the canal for several days.",
    },
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def compute_pair_similarities(pairs, model):
    """Compute cosine similarity for each pair using the given model."""
    texts_a = [p["a"] for p in pairs]
    texts_b = [p["b"] for p in pairs]
    embs_a = model.embed_batch(texts_a)
    embs_b = model.embed_batch(texts_b)
    return [cosine_sim(embs_a[i], embs_b[i]) for i in range(len(pairs))]


def try_ner_genericize(text: str, nlp) -> str:
    """Replace named entities with their type labels using spaCy."""
    doc = nlp(text)
    result = text
    for ent in reversed(doc.ents):
        result = result[:ent.start_char] + f"[{ent.label_}]" + result[ent.end_char:]
    return result


def main():
    print("=" * 90)
    print("EXPERIMENT 7: GENERICIZATION QUALITY EVALUATION")
    print("=" * 90)
    print(f"Same-meaning pairs: {len(SAME_MEANING_PAIRS)}")
    print(f"Unrelated pairs:    {len(UNRELATED_PAIRS)}")
    print(f"Categories: {', '.join(CATEGORIES.keys())}")

    # Check spaCy availability
    spacy_available = False
    nlp = None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        spacy_available = True
        print("spaCy: available (en_core_web_sm loaded)")
    except Exception:
        print("spaCy: not available, skipping NER-based genericization")

    methods = ["raw", "genericized"]
    if spacy_available:
        methods.append("ner_genericized")

    # --- Build models ---
    print("\nLoading embedding model...")
    t0 = time.perf_counter()
    model_raw = EmbeddingModel(genericize=False)
    model_gen = EmbeddingModel(genericize=True)
    # Force model load
    model_raw.embed("warmup")
    model_gen.embed("warmup")
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # --- Compute similarities ---
    results_by_method = {}

    # Raw
    print("\nComputing RAW similarities...")
    t0 = time.perf_counter()
    raw_same = compute_pair_similarities(SAME_MEANING_PAIRS, model_raw)
    raw_unrelated = compute_pair_similarities(UNRELATED_PAIRS, model_raw)
    results_by_method["raw"] = {
        "same_meaning": raw_same,
        "unrelated": raw_unrelated,
        "time": time.perf_counter() - t0,
    }

    # Genericized (regex)
    print("Computing GENERICIZED similarities...")
    t0 = time.perf_counter()
    gen_same = compute_pair_similarities(SAME_MEANING_PAIRS, model_gen)
    gen_unrelated = compute_pair_similarities(UNRELATED_PAIRS, model_gen)
    results_by_method["genericized"] = {
        "same_meaning": gen_same,
        "unrelated": gen_unrelated,
        "time": time.perf_counter() - t0,
    }

    # NER-based (if spaCy available)
    if spacy_available:
        print("Computing NER-GENERICIZED similarities...")
        t0 = time.perf_counter()

        ner_same_pairs = []
        for p in SAME_MEANING_PAIRS:
            ner_same_pairs.append({
                "a": try_ner_genericize(p["a"], nlp),
                "b": try_ner_genericize(p["b"], nlp),
            })
        ner_unrelated_pairs = []
        for p in UNRELATED_PAIRS:
            ner_unrelated_pairs.append({
                "a": try_ner_genericize(p["a"], nlp),
                "b": try_ner_genericize(p["b"], nlp),
            })

        ner_same = compute_pair_similarities(ner_same_pairs, model_raw)
        ner_unrelated = compute_pair_similarities(ner_unrelated_pairs, model_raw)
        results_by_method["ner_genericized"] = {
            "same_meaning": ner_same,
            "unrelated": ner_unrelated,
            "time": time.perf_counter() - t0,
        }

    # --- Aggregate metrics ---
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)

    summary = {}
    for method in methods:
        same = results_by_method[method]["same_meaning"]
        unrel = results_by_method[method]["unrelated"]
        mean_same = float(np.mean(same))
        mean_unrel = float(np.mean(unrel))
        gap = mean_same - mean_unrel
        summary[method] = {
            "mean_same_meaning": mean_same,
            "std_same_meaning": float(np.std(same)),
            "mean_unrelated": mean_unrel,
            "std_unrelated": float(np.std(unrel)),
            "discrimination_gap": gap,
            "time_sec": results_by_method[method]["time"],
        }

    # Overall table
    print(f"\n{'Method':<20s} {'Same (mean)':>12s} {'Same (std)':>11s} {'Unrel (mean)':>13s} {'Unrel (std)':>12s} {'Gap':>8s}")
    print("-" * 80)
    for method in methods:
        s = summary[method]
        print(f"{method:<20s} {s['mean_same_meaning']:>12.4f} {s['std_same_meaning']:>11.4f} {s['mean_unrelated']:>13.4f} {s['std_unrelated']:>12.4f} {s['discrimination_gap']:>8.4f}")

    # --- Per-category breakdown ---
    print(f"\n{'='*90}")
    print("PER-CATEGORY BREAKDOWN")
    print(f"{'='*90}")

    per_category = {}
    for cat, desc in CATEGORIES.items():
        cat_indices = [i for i, p in enumerate(SAME_MEANING_PAIRS) if p["category"] == cat]
        per_category[cat] = {"description": desc, "n_pairs": len(cat_indices)}

        print(f"\n--- {desc} ({len(cat_indices)} pairs) ---")
        print(f"  {'Method':<20s} {'Mean cosine':>12s} {'Std':>8s} {'vs Raw':>8s}")
        print(f"  {'-'*52}")

        for method in methods:
            same = results_by_method[method]["same_meaning"]
            cat_sims = [same[i] for i in cat_indices]
            mean_val = float(np.mean(cat_sims))
            std_val = float(np.std(cat_sims))

            # Improvement vs raw
            raw_cat_sims = [results_by_method["raw"]["same_meaning"][i] for i in cat_indices]
            raw_mean = float(np.mean(raw_cat_sims))
            delta = mean_val - raw_mean

            per_category[cat][method] = {
                "mean": mean_val,
                "std": std_val,
                "delta_vs_raw": delta,
                "similarities": cat_sims,
            }

            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            if method == "raw":
                delta_str = "  ---"
            print(f"  {method:<20s} {mean_val:>12.4f} {std_val:>8.4f} {delta_str:>8s}")

    # --- Per-pair details (biggest wins and losses from genericization) ---
    print(f"\n{'='*90}")
    print("BIGGEST WINS AND LOSSES FROM GENERICIZATION (same-meaning pairs)")
    print(f"{'='*90}")

    deltas = []
    for i in range(len(SAME_MEANING_PAIRS)):
        raw_sim = results_by_method["raw"]["same_meaning"][i]
        gen_sim = results_by_method["genericized"]["same_meaning"][i]
        deltas.append({
            "index": i,
            "category": SAME_MEANING_PAIRS[i]["category"],
            "raw": raw_sim,
            "genericized": gen_sim,
            "delta": gen_sim - raw_sim,
            "text_a": SAME_MEANING_PAIRS[i]["a"][:80],
            "text_b": SAME_MEANING_PAIRS[i]["b"][:80],
        })

    deltas.sort(key=lambda x: x["delta"], reverse=True)

    print("\nTop 5 improvements (genericization helped most):")
    for d in deltas[:5]:
        print(f"  [{d['category']}] delta={d['delta']:+.4f}  raw={d['raw']:.4f}  gen={d['genericized']:.4f}")
        print(f"    A: {d['text_a']}")
        print(f"    B: {d['text_b']}")

    print("\nTop 5 regressions (genericization hurt most):")
    for d in deltas[-5:]:
        print(f"  [{d['category']}] delta={d['delta']:+.4f}  raw={d['raw']:.4f}  gen={d['genericized']:.4f}")
        print(f"    A: {d['text_a']}")
        print(f"    B: {d['text_b']}")

    # --- Show what genericization actually does to the text ---
    print(f"\n{'='*90}")
    print("GENERICIZATION EXAMPLES (what the regex strips)")
    print(f"{'='*90}")
    for cat in CATEGORIES:
        idx = next(i for i, p in enumerate(SAME_MEANING_PAIRS) if p["category"] == cat)
        pair = SAME_MEANING_PAIRS[idx]
        print(f"\n  [{cat}]")
        print(f"    Original A: {pair['a']}")
        print(f"    Generic A:  {model_gen._genericize(pair['a'])}")
        print(f"    Original B: {pair['b']}")
        print(f"    Generic B:  {model_gen._genericize(pair['b'])}")

    # --- Proper nouns analysis ---
    print(f"\n{'='*90}")
    print("ANALYSIS: PROPER NOUNS (not stripped by current regex)")
    print(f"{'='*90}")
    pn_indices = [i for i, p in enumerate(SAME_MEANING_PAIRS) if p["category"] == "proper_nouns"]
    pn_raw = [results_by_method["raw"]["same_meaning"][i] for i in pn_indices]
    pn_gen = [results_by_method["genericized"]["same_meaning"][i] for i in pn_indices]
    print(f"  Proper noun pairs: raw mean={np.mean(pn_raw):.4f}, genericized mean={np.mean(pn_gen):.4f}")
    print(f"  Delta: {np.mean(pn_gen) - np.mean(pn_raw):+.4f}")
    print(f"  Note: current regex does NOT strip proper nouns.")
    print(f"  These pairs differ mainly in names/institutions — genericization has minimal effect.")
    if spacy_available:
        pn_ner = [results_by_method["ner_genericized"]["same_meaning"][i] for i in pn_indices]
        print(f"  NER-based: mean={np.mean(pn_ner):.4f}, delta vs raw={np.mean(pn_ner) - np.mean(pn_raw):+.4f}")
        print(f"  NER genericization DOES help proper nouns by replacing names with type labels.")

    # --- Answer the research question ---
    print(f"\n{'='*90}")
    print("CONCLUSIONS")
    print(f"{'='*90}")

    gen_gap = summary["genericized"]["discrimination_gap"]
    raw_gap = summary["raw"]["discrimination_gap"]
    gap_improvement = gen_gap - raw_gap

    print(f"\n1. Overall discrimination gap (same-meaning minus unrelated):")
    print(f"   Raw:         {raw_gap:.4f}")
    print(f"   Genericized: {gen_gap:.4f}")
    print(f"   Improvement: {gap_improvement:+.4f}")

    print(f"\n2. Per-category impact on same-meaning cosine (genericized vs raw):")
    for cat in CATEGORIES:
        delta = per_category[cat]["genericized"]["delta_vs_raw"]
        print(f"   {cat:<16s}: {delta:+.4f}")

    print(f"\n3. Research question: Should genericization strip domain-specific terms?")
    pn_delta = float(np.mean(pn_gen)) - float(np.mean(pn_raw))
    if pn_delta < 0.005:
        print(f"   Current regex has minimal effect on proper nouns (delta={pn_delta:+.4f}).")
        if spacy_available:
            ner_delta = float(np.mean(pn_ner)) - float(np.mean(pn_raw))
            if ner_delta > 0.01:
                print(f"   NER-based genericization helps (delta={ner_delta:+.4f}) but requires spaCy.")
            else:
                print(f"   NER-based genericization also has minimal effect (delta={ner_delta:+.4f}).")
        else:
            print(f"   NER-based approach (spaCy) could not be tested — not installed.")
            print(f"   Recommendation: install spaCy and test, or accept that proper nouns")
            print(f"   contribute useful semantic signal that should not be stripped.")
    else:
        print(f"   Current regex unexpectedly affects proper noun pairs (delta={pn_delta:+.4f}).")

    # --- Save JSON ---
    # Clean per_category for JSON (remove numpy arrays)
    per_category_json = {}
    for cat, data in per_category.items():
        per_category_json[cat] = {
            "description": data["description"],
            "n_pairs": data["n_pairs"],
        }
        for method in methods:
            m = data[method]
            per_category_json[cat][method] = {
                "mean": m["mean"],
                "std": m["std"],
                "delta_vs_raw": m["delta_vs_raw"],
            }

    output = {
        "experiment": "exp7_genericization",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "n_same_meaning_pairs": len(SAME_MEANING_PAIRS),
        "n_unrelated_pairs": len(UNRELATED_PAIRS),
        "methods": methods,
        "summary": summary,
        "per_category": per_category_json,
        "pair_deltas": [
            {
                "index": d["index"],
                "category": d["category"],
                "raw": d["raw"],
                "genericized": d["genericized"],
                "delta": d["delta"],
            }
            for d in deltas
        ],
        "research_question": "Should genericization strip domain-specific terms (school names, researcher names)?",
        "spacy_available": spacy_available,
    }

    out_path = RESULTS_DIR / "exp7_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
