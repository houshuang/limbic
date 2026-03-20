# Application Threshold Calibration: Nomic 768d → MiniLM 384d

**Date**: 2026-03-19
**Context**: Application's knowledge index uses cosine similarity thresholds to classify claim pairs as KNOWN/EXTENDS/NEW. Thresholds were 0.78/0.68, calibrated for Nomic 768d. After migrating to `paraphrase-multilingual-MiniLM-L12-v2` (384d) via amygdala, thresholds need recalibration.

## Corpus

- 4,500 atomic claims from 237 articles (mostly Sicilian history, AI/tech, mixed topics)
- 10.1M total pairs, 120K cross-article pairs above cosine 0.50

## Score Distribution Comparison

MiniLM produces a **much tighter similarity distribution** than Nomic — fewer pairs reach high cosine:

| Threshold | Nomic 768d (858 claims) | MiniLM 384d (4,500 claims) | MiniLM cross-article |
|-----------|------------------------|---------------------------|---------------------|
| ≥ 0.95 | 29 | 19 | 15 |
| ≥ 0.90 | 258 | 79 | 47 |
| ≥ 0.85 | 879 | 278 | 137 |
| ≥ 0.80 | 2,263 | 865 | 438 |
| ≥ 0.78 | 3,074 | 1,279 | 657 |
| ≥ 0.72 | 6,731 | 4,073 | 2,340 |
| ≥ 0.68 | 14,305 | 8,128 | 5,181 |

Even accounting for corpus size difference (858 vs 4,500), MiniLM is ~4× more discriminating at high thresholds.

## NLI Cross-Validation

Sampled 15 cross-article pairs per band, classified with `nli_classify()` (cross-encoder/nli-deberta-v3-base):

| Cosine band | Entailment | Neutral | Contradiction | Interpretation |
|-------------|-----------|---------|---------------|----------------|
| 0.88–1.00 | 47% | 40% | 13% | True duplicates + near-duplicates |
| 0.82–0.88 | 13% | 87% | 0% | Related but DIFFERENT claims |
| 0.75–0.82 | 7% | 80% | 13% | Different claims, same domain |
| 0.68–0.75 | 7% | 53% | 40% | Contradictions spike — danger zone |
| 0.60–0.68 | 0% | 93% | 7% | Same topic, genuinely different |
| 0.50–0.60 | 0% | 87% | 13% | Loosely related at best |

### Key findings

1. **Entailment cliff at 0.88**: Drops from 47% to 13%. This is the natural KNOWN boundary.
2. **Contradiction spike at 0.72**: Jumps from 13% to 40%. Below 0.72, cosine conflates related claims with contradictory ones.
3. **Clean neutral zone 0.72–0.88**: 80-87% neutral (related, non-contradictory). Perfect EXTENDS zone, but needs NLI to filter the occasional entailment (promote to KNOWN) or contradiction (demote to NEW).

## Chosen Thresholds

| Classification | Old (Nomic) | New (MiniLM) | Rationale |
|---------------|-------------|--------------|-----------|
| KNOWN | ≥ 0.78 | ≥ 0.88 | Above entailment cliff |
| NLI zone | 0.68–0.78 | 0.72–0.88 | NLI disambiguates (entailment→KNOWN, contradiction→NEW, neutral→EXTENDS) |
| EXTENDS | ≥ 0.68 | ≥ 0.72 | Below 0.72, 40% contradictions |
| NEW | < 0.68 | < 0.72 | — |

NLI runtime: 2,271 pairs in the ambiguous zone × 7ms/pair = ~16 seconds (free, local).

## Generalizable Insights for Amygdala Users

1. **Don't transfer thresholds between models.** Nomic 0.78 ≠ MiniLM 0.78. Always recalibrate.
2. **MiniLM's tighter distribution is a feature.** Higher scores are more meaningful — fewer false matches.
3. **NLI as cascade works extremely well.** Cosine handles easy cases (>0.88 = same, <0.72 = different), NLI resolves the ambiguous middle. This is the recommended pattern for any dedup/knowledge-matching use case.
4. **Watch for contradictions.** Cosine similarity cannot distinguish "agrees with" from "contradicts" — both get high scores if claims share vocabulary. Below ~0.72 (MiniLM), contradictions become frequent enough to be a problem.
5. **Calibration methodology**: Sample ~15 pairs per cosine band, run `nli_classify_batch()`, look for the entailment cliff and contradiction spike. Takes minutes, not hours.

## Manual Pair Inspection (Reference)

### Cosine ~0.90 (true duplicates)
- "Sicilian language has vocabulary influenced by Greek, Catalan, Norman, French" ↔ "Languages that have influenced Sicilian include Latin, Ancient Greek, Byzantine"
- "Proof tracks the identity of the author for every character" ↔ "Proof tracks authorship for every character written"

### Cosine ~0.80 (related but different)
- "Operation Sicilian Vespers (1992–98) aimed to fight the Sicilian mafia" ↔ "War of the Sicilian Vespers began with the revolt" (different centuries!)
- "Large part of population speaks Sicilian" ↔ "Sicilian is primarily a home language among peers"

### Cosine ~0.65 (topically linked, semantically distant)
- "About half the population of Sicily was Muslim at the Norman Conquest" ↔ "Norman lord Roger II conquered Sicily and established the Kingdom"
