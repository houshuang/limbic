import os
"""Experiment 16: Cross-lingual retrieval — Norwegian query → English document.

Amygdala uses paraphrase-multilingual-MiniLM-L12-v2 (384d) which scored 0.84 on
Norwegian similarity pairs. But cross-lingual retrieval hasn't been tested.
Otak currently translates Norwegian to English before embedding — if cross-lingual
works well enough, that translation step is unnecessary.

Tests:
  1. Cross-lingual similarity: EN-NO matching vs non-matching pairs
  2. Cross-lingual retrieval: Norwegian query → English VectorIndex (R@1, R@3, R@5, MRR)
  3. Translation baseline: does translating NO→EN before embedding help?
  4. Real domain data: Norwegian queries against actual domain claims
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex

OTAK_DB = Path(os.environ.get("AMYGDALA_EVAL_DB", "eval_claims.db"))
RESULTS_PATH = Path("experiments/results/exp16_results.json")
RNG = np.random.default_rng(42)


# ── Data: 30 concept pairs (EN + NO) ────────────────────────────────

CONCEPT_PAIRS = [
    # Education domain (domain core)
    {
        "en": "Students learn better when they receive formative feedback rather than just grades",
        "no": "Elever lærer bedre når de får formativ tilbakemelding i stedet for bare karakterer",
        "type": "translation",
    },
    {
        "en": "Play-based learning in early childhood education supports social and cognitive development",
        "no": "Lekbasert læring i barnehagen støtter sosial og kognitiv utvikling",
        "type": "translation",
    },
    {
        "en": "Dialogic reading interventions improve children's vocabulary acquisition",
        "no": "Dialogisk lesing styrker barns ordforråd og språkutvikling",
        "type": "paraphrase",
    },
    {
        "en": "Teachers need more time for professional development and collaboration",
        "no": "Lærere trenger mer tid til faglig utvikling og samarbeid",
        "type": "translation",
    },
    {
        "en": "Standardized testing narrows the curriculum and encourages teaching to the test",
        "no": "Nasjonale prøver innsnevrer pensum og fører til at man underviser mot testen",
        "type": "paraphrase",
    },
    {
        "en": "Outdoor education promotes both physical health and environmental awareness in children",
        "no": "Uteskole fremmer både fysisk helse og miljøbevissthet hos barn",
        "type": "translation",
    },
    {
        "en": "Digital tools in schools must be balanced with analog learning experiences",
        "no": "Digitale verktøy i skolen må balanseres med analoge læringsopplevelser",
        "type": "translation",
    },
    {
        "en": "Inclusive education requires adapting teaching methods to individual student needs",
        "no": "Inkluderende opplæring krever tilpasning av undervisning til den enkeltes behov",
        "type": "translation",
    },
    {
        "en": "Class size has a significant impact on teaching quality and student outcomes",
        "no": "Klassestørrelse har stor betydning for undervisningskvaliteten og elevenes læringsutbytte",
        "type": "paraphrase",
    },
    {
        "en": "Early intervention programs for reading difficulties reduce long-term academic problems",
        "no": "Tidlig innsats for lesevansker reduserer langsiktige faglige utfordringer",
        "type": "translation",
    },
    # Climate and environment
    {
        "en": "Climate change increases the frequency of extreme weather events",
        "no": "Klimaendringer øker hyppigheten av ekstreme værhendelser",
        "type": "translation",
    },
    {
        "en": "Renewable energy sources must replace fossil fuels to meet emission targets",
        "no": "Fornybare energikilder må erstatte fossile brensler for å nå utslippsmålene",
        "type": "translation",
    },
    {
        "en": "Biodiversity loss threatens ecosystem stability and human food security",
        "no": "Tap av biologisk mangfold truer stabiliteten i økosystemer og matsikkerhet",
        "type": "paraphrase",
    },
    {
        "en": "Norway's data center expansion conflicts with nature conservation goals",
        "no": "Norges utbygging av datasentre er i konflikt med mål om naturvern",
        "type": "translation",
    },
    {
        "en": "Carbon capture and storage technology is not yet viable at scale",
        "no": "Karbonfangst og lagring er ennå ikke gjennomførbart i stor skala",
        "type": "translation",
    },
    # Politics and policy
    {
        "en": "Universal basic income could reduce poverty without discouraging employment",
        "no": "Borgerlønn kan redusere fattigdom uten å svekke arbeidsmotivasjonen",
        "type": "paraphrase",
    },
    {
        "en": "Democratic participation declines when citizens feel their vote doesn't matter",
        "no": "Demokratisk deltakelse synker når innbyggerne føler at stemmen deres ikke betyr noe",
        "type": "translation",
    },
    {
        "en": "Public health policy must balance individual freedom with collective responsibility",
        "no": "Folkehelsepolitikk må balansere individuell frihet med kollektivt ansvar",
        "type": "translation",
    },
    {
        "en": "Income inequality has increased significantly in OECD countries since the 1980s",
        "no": "Inntektsulikhet har økt betydelig i OECD-land siden 1980-tallet",
        "type": "translation",
    },
    {
        "en": "Municipal mergers in Norway have not consistently improved public services",
        "no": "Kommunesammenslåinger i Norge har ikke konsekvent forbedret offentlige tjenester",
        "type": "translation",
    },
    # Technology and society
    {
        "en": "Artificial intelligence will transform the labor market within the next decade",
        "no": "Kunstig intelligens vil forandre arbeidsmarkedet i løpet av det neste tiåret",
        "type": "translation",
    },
    {
        "en": "Screen time for young children should be limited according to health guidelines",
        "no": "Skjermtid for små barn bør begrenses i tråd med helseanbefalinger",
        "type": "translation",
    },
    {
        "en": "Social media algorithms create filter bubbles that reinforce existing beliefs",
        "no": "Algoritmer i sosiale medier skaper filterbobler som forsterker eksisterende overbevisninger",
        "type": "translation",
    },
    {
        "en": "Video conferencing fatigue results from the extra cognitive load of interpreting cues on screen",
        "no": "Utmattelse fra videokonferanser skyldes den ekstra kognitive belastningen ved å tolke signaler på skjerm",
        "type": "paraphrase",
    },
    {
        "en": "Privacy regulations need to keep pace with rapid advances in data collection technology",
        "no": "Personvernreguleringer må holde tritt med raske fremskritt innen datainnsamlingsteknologi",
        "type": "translation",
    },
    # Research and methodology
    {
        "en": "Qualitative research methods capture nuances that quantitative approaches miss",
        "no": "Kvalitative forskningsmetoder fanger nyanser som kvantitative tilnærminger overser",
        "type": "translation",
    },
    {
        "en": "Replication failures in psychology have undermined confidence in many established findings",
        "no": "Replikasjonskrisen i psykologien har svekket tilliten til mange etablerte funn",
        "type": "paraphrase",
    },
    {
        "en": "Randomized controlled trials are the gold standard for establishing causal effects",
        "no": "Randomiserte kontrollerte studier er gullstandarden for å fastslå årsakssammenhenger",
        "type": "translation",
    },
    {
        "en": "Interdisciplinary research produces more innovative solutions to complex problems",
        "no": "Tverrfaglig forskning gir mer innovative løsninger på komplekse problemer",
        "type": "translation",
    },
    {
        "en": "Open access publishing accelerates scientific progress by removing paywalls",
        "no": "Åpen tilgang til forskning fremskynder vitenskapelig fremgang ved å fjerne betalingsmurer",
        "type": "paraphrase",
    },
]

# Translation baseline: manual EN translations of Norwegian texts for a subset (10 pairs)
TRANSLATION_BASELINE = [
    {
        "no": "Elever lærer bedre når de får formativ tilbakemelding i stedet for bare karakterer",
        "translated_en": "Students learn better when they receive formative feedback instead of just grades",
    },
    {
        "no": "Dialogisk lesing styrker barns ordforråd og språkutvikling",
        "translated_en": "Dialogic reading strengthens children's vocabulary and language development",
    },
    {
        "no": "Nasjonale prøver innsnevrer pensum og fører til at man underviser mot testen",
        "translated_en": "National tests narrow the curriculum and lead to teaching to the test",
    },
    {
        "no": "Klimaendringer øker hyppigheten av ekstreme værhendelser",
        "translated_en": "Climate change increases the frequency of extreme weather events",
    },
    {
        "no": "Tap av biologisk mangfold truer stabiliteten i økosystemer og matsikkerhet",
        "translated_en": "Loss of biodiversity threatens ecosystem stability and food security",
    },
    {
        "no": "Borgerlønn kan redusere fattigdom uten å svekke arbeidsmotivasjonen",
        "translated_en": "Basic income can reduce poverty without weakening work motivation",
    },
    {
        "no": "Kunstig intelligens vil forandre arbeidsmarkedet i løpet av det neste tiåret",
        "translated_en": "Artificial intelligence will change the labor market within the next decade",
    },
    {
        "no": "Algoritmer i sosiale medier skaper filterbobler som forsterker eksisterende overbevisninger",
        "translated_en": "Social media algorithms create filter bubbles that reinforce existing beliefs",
    },
    {
        "no": "Kvalitative forskningsmetoder fanger nyanser som kvantitative tilnærminger overser",
        "translated_en": "Qualitative research methods capture nuances that quantitative approaches miss",
    },
    {
        "no": "Replikasjonskrisen i psykologien har svekket tilliten til mange etablerte funn",
        "translated_en": "The replication crisis in psychology has weakened confidence in many established findings",
    },
]

# Real domain data: 10 hand-crafted Norwegian queries for actual domain claims
OTAK_QUERIES = [
    {
        "no_query": "Formativ vurdering er bedre enn karakterer for elevenes læring",
        "expected_keywords": ["formative", "feedback", "grades"],
    },
    {
        "no_query": "Lekbasert læring i barnehagen fremmer utvikling",
        "expected_keywords": ["play", "early childhood", "learning"],
    },
    {
        "no_query": "Lærere trenger mer tid til faglig samarbeid og utvikling",
        "expected_keywords": ["teacher", "professional development", "collaboration"],
    },
    {
        "no_query": "Standardiserte tester fører til at lærere underviser mot prøven",
        "expected_keywords": ["test", "exam", "teaching to the test"],
    },
    {
        "no_query": "Uteskole er bra for barns fysiske helse",
        "expected_keywords": ["outdoor", "physical", "health", "nature"],
    },
    {
        "no_query": "Tidlig innsats for lesevansker hjelper barn på lang sikt",
        "expected_keywords": ["reading", "early", "intervention", "phonological"],
    },
    {
        "no_query": "Klassestørrelse påvirker kvaliteten på undervisningen",
        "expected_keywords": ["class size", "teaching quality"],
    },
    {
        "no_query": "Digitale verktøy i skolen må brukes med omtanke",
        "expected_keywords": ["digital", "school", "analog"],
    },
    {
        "no_query": "Barns nysgjerrighet bør styrkes i barnehagen",
        "expected_keywords": ["curiosity", "children", "ECEC", "kindergarten"],
    },
    {
        "no_query": "Skjermtid for små barn bør begrenses",
        "expected_keywords": ["screen time", "children", "limit"],
    },
]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Part 1: Cross-lingual Similarity ────────────────────────────────

def test_cross_lingual_similarity(model):
    print("\n" + "=" * 80)
    print("PART 1: Cross-Lingual Similarity (EN-NO pair similarity)")
    print("=" * 80)

    en_texts = [p["en"] for p in CONCEPT_PAIRS]
    no_texts = [p["no"] for p in CONCEPT_PAIRS]
    types = [p["type"] for p in CONCEPT_PAIRS]

    en_vecs = model.embed_batch(en_texts)
    no_vecs = model.embed_batch(no_texts)

    # Matching pair similarities
    match_sims = [cosine_sim(en_vecs[i], no_vecs[i]) for i in range(len(CONCEPT_PAIRS))]
    translation_sims = [s for s, t in zip(match_sims, types) if t == "translation"]
    paraphrase_sims = [s for s, t in zip(match_sims, types) if t == "paraphrase"]

    # Non-matching pair similarities (all off-diagonal)
    non_match_sims = []
    for i in range(len(CONCEPT_PAIRS)):
        for j in range(len(CONCEPT_PAIRS)):
            if i != j:
                non_match_sims.append(cosine_sim(en_vecs[i], no_vecs[j]))

    gap = np.mean(match_sims) - np.mean(non_match_sims)

    print(f"\nMatching pairs (n={len(match_sims)}):")
    print(f"  Mean cosine:   {np.mean(match_sims):.4f}")
    print(f"  Min:           {np.min(match_sims):.4f}")
    print(f"  Max:           {np.max(match_sims):.4f}")
    print(f"  Std:           {np.std(match_sims):.4f}")
    print(f"\n  Translations (n={len(translation_sims)}): mean={np.mean(translation_sims):.4f}")
    print(f"  Paraphrases  (n={len(paraphrase_sims)}):  mean={np.mean(paraphrase_sims):.4f}")

    print(f"\nNon-matching pairs (n={len(non_match_sims)}):")
    print(f"  Mean cosine:   {np.mean(non_match_sims):.4f}")
    print(f"  Std:           {np.std(non_match_sims):.4f}")
    print(f"  95th pctile:   {np.percentile(non_match_sims, 95):.4f}")

    print(f"\nDiscrimination gap: {gap:.4f}")

    # Show worst matching pairs (lowest similarity)
    sorted_pairs = sorted(enumerate(match_sims), key=lambda x: x[1])
    print("\nLowest cross-lingual similarity (potential trouble):")
    for idx, sim in sorted_pairs[:5]:
        print(f"  {sim:.4f}  EN: {en_texts[idx][:60]}...")
        print(f"          NO: {no_texts[idx][:60]}...")

    return {
        "match_mean": float(np.mean(match_sims)),
        "match_min": float(np.min(match_sims)),
        "match_max": float(np.max(match_sims)),
        "match_std": float(np.std(match_sims)),
        "translation_mean": float(np.mean(translation_sims)),
        "paraphrase_mean": float(np.mean(paraphrase_sims)),
        "nonmatch_mean": float(np.mean(non_match_sims)),
        "nonmatch_std": float(np.std(non_match_sims)),
        "nonmatch_p95": float(np.percentile(non_match_sims, 95)),
        "discrimination_gap": float(gap),
        "all_match_sims": [float(s) for s in match_sims],
    }


# ── Part 2: Cross-lingual Retrieval ─────────────────────────────────

def test_cross_lingual_retrieval(model):
    print("\n" + "=" * 80)
    print("PART 2: Cross-Lingual Retrieval (NO query → EN index)")
    print("=" * 80)

    en_texts = [p["en"] for p in CONCEPT_PAIRS]
    no_texts = [p["no"] for p in CONCEPT_PAIRS]

    en_vecs = model.embed_batch(en_texts)
    no_vecs = model.embed_batch(no_texts)

    # Build EN index
    vi = VectorIndex()
    ids = [f"en_{i}" for i in range(len(en_texts))]
    vi.add(ids, en_vecs)

    # Search with each NO query
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_ranks = []

    print(f"\nSearching {len(no_texts)} Norwegian queries against {len(en_texts)} English documents:")
    failures = []

    for i, no_vec in enumerate(no_vecs):
        results = vi.search(no_vec, limit=10)
        result_ids = [r.id for r in results]
        target_id = f"en_{i}"

        if target_id in result_ids[:1]:
            recall_at_1 += 1
        if target_id in result_ids[:3]:
            recall_at_3 += 1
        if target_id in result_ids[:5]:
            recall_at_5 += 1

        if target_id in result_ids:
            rank = result_ids.index(target_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            failures.append((i, no_texts[i], en_texts[i]))

    n = len(no_texts)
    r1 = recall_at_1 / n
    r3 = recall_at_3 / n
    r5 = recall_at_5 / n
    mrr = np.mean(reciprocal_ranks)

    print(f"\n  Recall@1:  {r1:.3f} ({recall_at_1}/{n})")
    print(f"  Recall@3:  {r3:.3f} ({recall_at_3}/{n})")
    print(f"  Recall@5:  {r5:.3f} ({recall_at_5}/{n})")
    print(f"  MRR:       {mrr:.4f}")

    if failures:
        print(f"\n  Failures (not in top 10):")
        for idx, no, en in failures:
            print(f"    [{idx}] NO: {no[:60]}...")
            print(f"         EN: {en[:60]}...")

    # Show examples where match is not rank 1
    non_rank1 = [(i, reciprocal_ranks[i]) for i in range(n)
                 if reciprocal_ranks[i] > 0 and reciprocal_ranks[i] < 1.0]
    if non_rank1:
        print(f"\n  Found at rank > 1 ({len(non_rank1)} cases):")
        for idx, rr in non_rank1:
            rank = int(1 / rr)
            results = vi.search(no_vecs[idx], limit=5)
            print(f"    [{idx}] Rank {rank} — NO: {no_texts[idx][:50]}...")
            print(f"         True EN: {en_texts[idx][:50]}...")
            print(f"         Top hit: {en_texts[int(results[0].id.split('_')[1])][:50]}...")

    return {
        "recall_at_1": float(r1),
        "recall_at_3": float(r3),
        "recall_at_5": float(r5),
        "mrr": float(mrr),
        "n_queries": n,
        "n_failures": len(failures),
    }


# ── Part 3: Translation Baseline ────────────────────────────────────

def test_translation_baseline(model):
    print("\n" + "=" * 80)
    print("PART 3: Translation Baseline (NO→EN translation vs direct cross-lingual)")
    print("=" * 80)

    # Build index of all EN texts
    en_texts = [p["en"] for p in CONCEPT_PAIRS]
    en_vecs = model.embed_batch(en_texts)
    vi = VectorIndex()
    ids = [f"en_{i}" for i in range(len(en_texts))]
    vi.add(ids, en_vecs)

    # For each baseline pair, compare:
    #   A) direct: embed NO text, search EN index
    #   B) translated: embed translated-EN text, search EN index
    direct_ranks = []
    translated_ranks = []
    direct_sims = []
    translated_sims = []

    for tb in TRANSLATION_BASELINE:
        # Find which concept pair this corresponds to
        no_text = tb["no"]
        translated_text = tb["translated_en"]

        pair_idx = next(i for i, p in enumerate(CONCEPT_PAIRS) if p["no"] == no_text)
        target_id = f"en_{pair_idx}"
        en_target = CONCEPT_PAIRS[pair_idx]["en"]

        # Direct cross-lingual
        no_vec = model.embed(no_text)
        en_target_vec = en_vecs[pair_idx]
        direct_sim = cosine_sim(no_vec, en_target_vec)
        direct_sims.append(direct_sim)

        results_direct = vi.search(no_vec, limit=30)
        direct_ids = [r.id for r in results_direct]
        if target_id in direct_ids:
            direct_ranks.append(direct_ids.index(target_id) + 1)
        else:
            direct_ranks.append(999)

        # Translated
        tr_vec = model.embed(translated_text)
        translated_sim = cosine_sim(tr_vec, en_target_vec)
        translated_sims.append(translated_sim)

        results_translated = vi.search(tr_vec, limit=30)
        translated_ids = [r.id for r in results_translated]
        if target_id in translated_ids:
            translated_ranks.append(translated_ids.index(target_id) + 1)
        else:
            translated_ranks.append(999)

    print(f"\n{'Pair':<4} {'Direct sim':>11} {'Transl sim':>11} {'Dir rank':>9} {'Tr rank':>8}")
    print("-" * 50)
    for i, tb in enumerate(TRANSLATION_BASELINE):
        pair_idx = next(j for j, p in enumerate(CONCEPT_PAIRS) if p["no"] == tb["no"])
        print(f"{pair_idx:<4} {direct_sims[i]:>11.4f} {translated_sims[i]:>11.4f} "
              f"{direct_ranks[i]:>9} {translated_ranks[i]:>8}")

    direct_mrr = np.mean([1.0 / r if r < 999 else 0 for r in direct_ranks])
    translated_mrr = np.mean([1.0 / r if r < 999 else 0 for r in translated_ranks])

    print(f"\nDirect cross-lingual:")
    print(f"  Mean similarity: {np.mean(direct_sims):.4f}")
    print(f"  MRR:             {direct_mrr:.4f}")
    print(f"  Mean rank:       {np.mean([r for r in direct_ranks if r < 999]):.1f}")

    print(f"\nTranslated NO→EN:")
    print(f"  Mean similarity: {np.mean(translated_sims):.4f}")
    print(f"  MRR:             {translated_mrr:.4f}")
    print(f"  Mean rank:       {np.mean([r for r in translated_ranks if r < 999]):.1f}")

    sim_delta = np.mean(translated_sims) - np.mean(direct_sims)
    print(f"\nTranslation advantage (sim): {sim_delta:+.4f}")
    print(f"Translation advantage (MRR): {translated_mrr - direct_mrr:+.4f}")

    return {
        "direct_mean_sim": float(np.mean(direct_sims)),
        "translated_mean_sim": float(np.mean(translated_sims)),
        "direct_mrr": float(direct_mrr),
        "translated_mrr": float(translated_mrr),
        "sim_delta": float(sim_delta),
        "mrr_delta": float(translated_mrr - direct_mrr),
        "per_pair_direct_sims": [float(s) for s in direct_sims],
        "per_pair_translated_sims": [float(s) for s in translated_sims],
        "per_pair_direct_ranks": direct_ranks,
        "per_pair_translated_ranks": translated_ranks,
    }


# ── Part 4: Real Otak Data ──────────────────────────────────────────

def test_domain_retrieval(model):
    print("\n" + "=" * 80)
    print("PART 4: Real Domain Data (NO query → domain claims)")
    print("=" * 80)

    conn = sqlite3.connect(str(OTAK_DB))
    cur = conn.execute(
        "SELECT DISTINCT n.name FROM idx_knowledge_item_claim_type k "
        "JOIN nodes n ON n.id = k.node_id "
        "WHERE n.deleted_at IS NULL AND n.name IS NOT NULL "
        "AND length(n.name) > 20 AND length(n.name) < 500 "
        "ORDER BY RANDOM() LIMIT 2000"
    )
    domain_claims = [row[0] for row in cur.fetchall()]
    conn.close()

    if not domain_claims:
        print("  No domain claims found, skipping Part 4")
        return None

    print(f"  Loaded {len(domain_claims)} domain claims")

    # Embed and index domain claims
    t0 = time.time()
    domain_vecs = model.embed_batch(domain_claims)
    embed_time = time.time() - t0
    print(f"  Embedded in {embed_time:.1f}s")

    vi = VectorIndex()
    domain_ids = [f"domain_{i}" for i in range(len(domain_claims))]
    vi.add(domain_ids, domain_vecs)

    results_data = []

    for q in OTAK_QUERIES:
        no_query = q["no_query"]
        keywords = q["expected_keywords"]

        query_vec = model.embed(no_query)
        results = vi.search(query_vec, limit=10)

        print(f"\n  Query: \"{no_query}\"")
        print(f"  Expected keywords: {keywords}")

        hits = []
        keyword_found_at = None
        for rank, r in enumerate(results[:5], 1):
            idx = int(r.id.split("_")[1])
            claim = domain_claims[idx]
            claim_lower = claim.lower()
            has_keyword = any(kw.lower() in claim_lower for kw in keywords)
            marker = " <<" if has_keyword else ""
            print(f"    #{rank} ({r.score:.3f}) {claim[:80]}...{marker}")
            hits.append({
                "rank": rank,
                "score": float(r.score),
                "claim": claim,
                "keyword_match": has_keyword,
            })
            if has_keyword and keyword_found_at is None:
                keyword_found_at = rank

        results_data.append({
            "query": no_query,
            "expected_keywords": keywords,
            "keyword_found_at_rank": keyword_found_at,
            "top_hits": hits,
        })

    # Aggregate keyword-match stats
    found_ranks = [r["keyword_found_at_rank"] for r in results_data if r["keyword_found_at_rank"] is not None]
    keyword_recall_at_5 = len(found_ranks) / len(results_data)

    print(f"\n  Keyword-relevant result in top 5: {len(found_ranks)}/{len(results_data)} ({keyword_recall_at_5:.0%})")
    if found_ranks:
        print(f"  Mean rank of first keyword match: {np.mean(found_ranks):.1f}")

    return {
        "n_claims": len(domain_claims),
        "n_queries": len(OTAK_QUERIES),
        "keyword_recall_at_5": float(keyword_recall_at_5),
        "mean_keyword_rank": float(np.mean(found_ranks)) if found_ranks else None,
        "per_query": results_data,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("EXPERIMENT 16: Cross-Lingual Retrieval (Norwegian → English)")
    print("=" * 80)
    print(f"Model: paraphrase-multilingual-MiniLM-L12-v2 (384d)")
    print(f"Concept pairs: {len(CONCEPT_PAIRS)}")

    t0 = time.time()
    model = EmbeddingModel()

    results = {}
    results["part1_similarity"] = test_cross_lingual_similarity(model)
    results["part2_retrieval"] = test_cross_lingual_retrieval(model)
    results["part3_translation_baseline"] = test_translation_baseline(model)

    try:
        results["part4_domain"] = test_domain_retrieval(model)
    except Exception as e:
        print(f"\n  Part 4 failed: {e}")
        results["part4_domain"] = {"error": str(e)}

    total_time = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    p1 = results["part1_similarity"]
    p2 = results["part2_retrieval"]
    p3 = results["part3_translation_baseline"]

    print(f"\n1. Cross-lingual similarity:")
    print(f"   Matching pairs:     {p1['match_mean']:.4f} (min {p1['match_min']:.4f})")
    print(f"   Non-matching pairs: {p1['nonmatch_mean']:.4f}")
    print(f"   Discrimination gap: {p1['discrimination_gap']:.4f}")
    print(f"   Translations:       {p1['translation_mean']:.4f}")
    print(f"   Paraphrases:        {p1['paraphrase_mean']:.4f}")

    print(f"\n2. Cross-lingual retrieval (NO→EN, n={p2['n_queries']}):")
    print(f"   Recall@1: {p2['recall_at_1']:.3f}")
    print(f"   Recall@3: {p2['recall_at_3']:.3f}")
    print(f"   Recall@5: {p2['recall_at_5']:.3f}")
    print(f"   MRR:      {p2['mrr']:.4f}")

    print(f"\n3. Translation baseline (n={len(TRANSLATION_BASELINE)}):")
    print(f"   Direct cross-lingual MRR:  {p3['direct_mrr']:.4f}")
    print(f"   Translated NO→EN MRR:      {p3['translated_mrr']:.4f}")
    print(f"   Translation advantage:     {p3['mrr_delta']:+.4f} MRR, {p3['sim_delta']:+.4f} sim")

    if results.get("part4_domain") and not results["part4_domain"].get("error"):
        p4 = results["part4_domain"]
        print(f"\n4. Real domain data ({p4['n_claims']} claims, {p4['n_queries']} queries):")
        print(f"   Keyword-relevant in top 5: {p4['keyword_recall_at_5']:.0%}")
        if p4["mean_keyword_rank"] is not None:
            print(f"   Mean rank of keyword hit:  {p4['mean_keyword_rank']:.1f}")

    # Verdict
    print(f"\n{'=' * 80}")
    if p2["recall_at_1"] >= 0.90 and p3["mrr_delta"] < 0.05:
        print("VERDICT: Cross-lingual retrieval is strong. Translation step is unnecessary.")
    elif p2["recall_at_1"] >= 0.80:
        print("VERDICT: Cross-lingual retrieval is good. Translation helps marginally,")
        print("         but the added latency/complexity is likely not worth it.")
    elif p2["recall_at_1"] >= 0.60:
        print("VERDICT: Cross-lingual retrieval is moderate. Translation provides a")
        print("         meaningful boost — consider keeping it for precision-critical uses.")
    else:
        print("VERDICT: Cross-lingual retrieval is weak. Translation step is recommended.")
    print(f"{'=' * 80}")

    print(f"\nTotal time: {total_time:.1f}s")

    # Save
    results["metadata"] = {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "n_concept_pairs": len(CONCEPT_PAIRS),
        "n_translation_baseline": len(TRANSLATION_BASELINE),
        "total_time_s": float(total_time),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
