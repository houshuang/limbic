# `limbic.hippocampus.resolve` — retrieve candidates in code, never by prompt

Entity resolution against a local knowledge base, as plain functions over a
SQLite sidecar. The point is not fuzzy matching; it is **taking away the
model's ability to say the wrong identifier at all**. Code retrieves a short
candidate list, the schema offers only those slots, and the runner maps a slot
back to a real id — so a hallucinated identifier is unrepresentable rather than
merely discouraged.

## Ten lines

```python
from limbic.hippocampus.resolve import build_index, candidates, slot_enum, unslot

index = build_index("kb.idx", persons, kind="person")   # rows: id, name, aliases, summary, rank
build_index(index.conn, works, kind="work")             # several kinds, one sidecar

cards = candidates(index, "Bjoernson", k=10, hints={"birth_year": 1832})
fragment, slot_map = slot_enum(cards, n_slots=40)       # identical schema for the whole batch
# ... send `fragment` as the ref field of your response schema, `cards` in the packet body
items, problems = unslot(model_items, slot_map)         # c07 -> "1832-bjornson"; refuses c41
```

## What the pieces do

| Function | Returns | Notes |
|---|---|---|
| `fold(text, lang="nb", expand=False)` | folded string | Two spellings: drop (`bjornson`) and expand (`bjoernson`). The character map runs **before** NFKD — reversed, `Zauberflöte` could never produce `zauberfloete`. Coerces non-strings; a numeric title used to crash a production lookup. |
| `name_keys(name, lang="nb")` | list of keys | Both spellings, `Last, First` inversion, parenthetical strip, internal particles (`Ludwig van Beethoven` → `ludwig beethoven`), Norwegian genitive (`Ibsens` → `ibsen`). |
| `build_index(conn_or_path, rows, kind)` | `Index` | Folded key, order-independent token key, inverted token index, FTS5, first-letter/length blocking key. Rebuilding one kind leaves the others alone. |
| `candidates(index, query, k, kind, hints)` | `list[Card]` | `hints` rerank on any card field; they never filter, because the KB is often the one that is wrong. |
| `text_candidates(index, text, k)` | `list[Card]` | Entities whose *every* name token occurs in the passage, plus genitive stems. |
| `slot_enum(cards, n_slots)` | `(fragment, slot_map)` | Lists every slot `c01..cNN` whether or not it is filled. |
| `unslot(items, slot_map)` | `(resolved, problems)` | A slot the packet never supplied is a problem, not a guess. |

## Match types, in descending trust

| type | meaning | score |
|---|---|---|
| `exact` | byte-identical to an indexed name | 1.00 |
| `folded` | equal after folding, inversion or de-parenthesising | 0.93–0.97 |
| `token` | identical token set, any order | 0.88 |
| `token_overlap` | ≥2 shared tokens, scored by Jaccard | 0.60–0.85 |
| `fuzzy` | near-miss spelling, **off by default** | ≤ 0.84 |
| `fts` | FTS5 hit on some token | 0.30–0.65 |

`fuzzy` is capped below the 0.85 confidence threshold on purpose: a
one-character difference is a candidate for review, never an automatic link.
It is off by default because it is the layer that manufactures
plausible-looking wrong answers — and folding plus token-set matching recovered
only **5%** of one campaign's 5,194 held unmatched names, so it is usually not
where your recall gap is. A card is `confident` only at score ≥ `min_score`
**and** with no near-tied rival; ties are marked `ambiguous`, and an ambiguous
top hit is the only case that needs a model at all.

## Why the index is a sidecar

Not a table in the shipped database. The shipped file is read by a web app and
two native apps, so adding tables drags every consumer into a parity check for
an index only tooling reads, and the build step would have to run before any
lookup. A sidecar also works against an already-sealed release candidate, which
is what most workers actually have.

## Measure recall before you trust it

The reference implementation (`kb_resolve.py`, 20 Sep 2026) measured recall@1
0.93 / recall@5 0.995 on 200 held-out person pairs, with a **0.000** false-match
rate on 200 negatives, median 2.21 ms per query. Those numbers came from a gold
set mined out of the project's own applied *merge* proposals — a reviewed human
statement that this string denotes that entity — with no model calls at all. If
your project has merge history, you already have a gold set; if it has none,
you have never measured matching recall, which was true of every project in the
audit.

Two things that gold set taught, worth copying:

- **Filter the negatives.** Without it, "Erik Solheim (lektor)" and
  "Les Misérables" counted as names the catalogue does not hold, and the
  resolver was scored down for answering them correctly.
- **Report semantic reassignments separately.** Some applied merges are
  editorial, not name variants ("Tibor Vargas kammerorkester" → "Fiolinkonsert
  nr. 5 i A-dur"). No string matcher can reach those; averaging them in just
  hides the lexical number.

## Gotchas

- **The token minimum is shared.** The inverted index and the passage scan both
  use three characters. They must agree: indexing "Et dukkehjem" as two tokens
  while the scan only looks up tokens of three or more makes the work
  unfindable in its own text.
- **`rank` breaks ties**, so a lone surname returns the best-attested bearer
  first — while still refusing to be confident about it.
- **A hit from `text_candidates` is a coding candidate, never coverage.** The
  entity appearing on the page is not evidence that the page is about it.
