"""Deterministic entity resolution over a local knowledge base.

Retrieve candidates in code, then let the model pick one from a fixed list —
never let it type an identifier from memory. The 20 Sep 2026 llm-pipeline-audit
found the opposite pattern in production: an import prompt telling the model
"the full list is too large, use LIKE queries", credits resolved by exact
string match and otherwise *creating a new person*, 23 copies of
`normalize_name` with 17 distinct bodies, and no gold set, so matching recall
had never been measured at all. The repair bill was 636 merges and 574
duplicate proposals.

The reference implementation this generalises (`kb_resolve.py`, Kulturbase,
20 Sep 2026) measured recall@1 0.93 / recall@5 0.995 on 200 held-out person
pairs mined from merge history, with a **0.000** false-match rate on 200
negatives, at a median 2.21 ms per query.

Usage — build a sidecar index once, then resolve against it:

    from limbic.hippocampus.resolve import build_index, candidates, slot_enum

    index = build_index("kb.idx", persons, kind="person")   # rows: id, name, ...
    cards = candidates(index, "Bjoernson", k=10)
    fragment, slot_map = slot_enum(cards, n_slots=40)       # byte-identical schema
    # ... send `fragment` as the schema's ref field, `cards` as the packet body
    items, problems = unslot(model_items, slot_map)         # slots -> real ids

Why slots: an `enum` of *this* item's candidate IDs changes the JSON schema on
every call, and the schema is rendered ahead of the input in the provider's
cache prefix — measured 0% cached input across six calls with an otherwise
identical 5,962-token prefix. Fixed slots `c01..cNN` keep the schema
byte-identical (measured 58% cached input on the same batch) *and* remove the
last place the model handles an identifier of ours.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from limbic.amygdala import connect

__all__ = [
    "Card",
    "Index",
    "build_index",
    "candidates",
    "fold",
    "name_keys",
    "slot_enum",
    "text_candidates",
    "unslot",
]

# Nordic letters do not decompose under NFKD, so they need an explicit map, and
# the map must run *before* NFKD: NFKD splits o-umlaut into "o" + a combining
# mark, so folding afterwards would turn "Zauberflöte" into "zauberflote" and
# the transliterated "zauberfloete" spelling could never be produced — which is
# exactly the "Tryllefløjten = Die Zauberflöte" case a production import prompt
# was asking a model to guess at. Two spellings are indexed and generated per
# query: a *drop* spelling (the way an English-language source writes it,
# "Bjornson") and an *expand* spelling (the way a transliterating source does,
# "Bjoernson", "Naess", "Haaklev"). Either query spelling then meets either
# indexed one.
_NORDIC_DROP = {
    "ø": "o", "æ": "a", "å": "a", "ä": "a", "ö": "o", "ü": "u", "ß": "ss",
    "ð": "d", "þ": "th", "đ": "d", "ł": "l", "ŧ": "t", "ŋ": "n", "œ": "o",
}
_NORDIC_EXPAND = {
    "ø": "oe", "æ": "ae", "å": "aa", "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "ð": "d", "þ": "th", "đ": "d", "ł": "l", "ŧ": "t", "ŋ": "ng", "œ": "oe",
}
_GERMAN_DROP = {"ä": "a", "ö": "o", "ü": "u", "ß": "ss"}
_GERMAN_EXPAND = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}

# lang -> (drop map, expand map). An unknown language gets NFKD only, which is
# correct for Latin-script languages whose diacritics do decompose.
_LANG_MAPS: dict[str, tuple[dict[str, str], dict[str, str]]] = {
    "nb": (_NORDIC_DROP, _NORDIC_EXPAND),
    "nn": (_NORDIC_DROP, _NORDIC_EXPAND),
    "no": (_NORDIC_DROP, _NORDIC_EXPAND),
    "da": (_NORDIC_DROP, _NORDIC_EXPAND),
    "sv": (_NORDIC_DROP, _NORDIC_EXPAND),
    "is": (_NORDIC_DROP, _NORDIC_EXPAND),
    "de": (_GERMAN_DROP, _GERMAN_EXPAND),
    "": ({}, {}),
}

# Particles that sit inside a name. Stripping them gives a second key, so
# "Ludwig van Beethoven" also meets "Ludwig Beethoven".
_PARTICLES = frozenset(
    {"av", "von", "van", "de", "del", "della", "den", "der", "di", "du",
     "la", "le", "of", "ten", "ter", "van't", "zu"}
)

# Generational suffixes are not given names, so "Krogh, d.y." must not invert.
_SUFFIXES = frozenset({"jr", "sr", "d e", "d y", "den eldre", "den yngre", "ii", "iii", "iv"})

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")

# A trailing "s" is the Norwegian genitive ("Ibsens hus"), only stripped from a
# token long enough that the stem is still a plausible name.
_GENITIVE_MIN = 4

# Near-miss layer bounds. It is capped below the default confidence threshold
# on purpose: a one-character difference is a candidate for review, never an
# automatic link.
_FUZZY_POOL = 300
_FUZZY_MIN_RATIO = 0.80
_FUZZY_MAX = 0.84
_FUZZY_SCAN_LIMIT = 3000
_FUZZY_MIN_QUERY_CHARS = 4

# Minimum token length for the inverted index and for scanning a passage. The
# two must agree: indexing "Et dukkehjem" as two tokens while a passage scan
# only ever looks up tokens of three characters or more makes the work
# unfindable in its own text, because "all its tokens are present" can never
# become true. Particles below this length carry no identifying signal anyway.
_TOKEN_MIN = 3

DEFAULT_MIN_SCORE = 0.85
NONE_SLOT = "none"


# ---------------------------------------------------------------------------
# Folding and name keys
# ---------------------------------------------------------------------------

def _apply(text: str, table: Mapping[str, str]) -> str:
    return "".join(table.get(ch, ch) for ch in text) if table else text


def fold(text: Any, lang: str = "nb", *, expand: bool = False) -> str:
    """Casefold, strip diacritics, normalise punctuation and whitespace.

    `expand=False` gives the drop spelling (Bjørnson -> bjornson);
    `expand=True` gives the transliterated one (Bjørnson -> bjoernson). Index
    both and generate both for a query and either spelling matches.

    Non-strings are coerced rather than rejected: a catalogue title can be
    numeric-looking, and `'int' object has no attribute 'lower'` is a real
    crash this replaces.
    """
    if not isinstance(text, str):
        text = str(text)
    drop, expand_map = _LANG_MAPS.get(lang, _LANG_MAPS[""])
    out = _apply(text.casefold(), expand_map if expand else drop)
    out = unicodedata.normalize("NFKD", out)
    out = "".join(ch for ch in out if not unicodedata.combining(ch))
    # Hyphens, apostrophes and periods separate name parts, so they become
    # spaces rather than vanishing: "Jean-Luc" and "Jean Luc" agree.
    out = _PUNCT.sub(" ", out)
    return _SPACE.sub(" ", out).strip()


def _invert(name: str) -> str | None:
    """"Andre, Bjørn Tore" -> "Bjørn Tore Andre". None when not invertible."""
    if name.count(",") != 1:
        return None
    last, first = (part.strip() for part in name.split(","))
    if not last or not first or fold(first) in _SUFFIXES:
        return None
    return f"{first} {last}"


def _strip_parenthetical(name: str) -> str | None:
    """"Ibsen, Henrik (1828-1906)" -> "Ibsen, Henrik". None when unchanged."""
    if "(" not in name:
        return None
    stripped = re.sub(r"\s*\([^)]*\)", "", name).strip()
    return stripped if stripped and stripped != name.strip() else None


def genitive_stem(token: str) -> str | None:
    """Norwegian genitive: "Ibsens" -> "Ibsen". None when it does not apply."""
    if len(token) >= _GENITIVE_MIN and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return None


def tokens(name: Any, lang: str = "nb", *, minimum: int = 2) -> list[str]:
    return [t for t in fold(name, lang).split() if len(t) >= minimum]


def token_key(name: Any, lang: str = "nb") -> str:
    """Order-independent key. Empty below two tokens — matching a lone surname
    would make "Hansen" resolve confidently to whichever Hansen sorts first."""
    parts = sorted(set(tokens(name, lang)))
    return " ".join(parts) if len(parts) >= 2 else ""


def name_keys(name: Any, lang: str = "nb") -> list[str]:
    """Every folded spelling this name should be findable under.

    Covers both diacritic spellings, "Last, First" inversion, a parenthetical
    qualifier, an internal particle ("Ludwig van Beethoven" -> "ludwig
    beethoven") and the genitive ("Ibsens" -> "ibsen"). Deduplicated, most
    canonical first.
    """
    if not isinstance(name, str):
        name = str(name)
    surfaces = [name]
    for extra in (_invert(name), _strip_parenthetical(name)):
        if extra:
            surfaces.append(extra)
    inverted_stripped = _strip_parenthetical(name)
    if inverted_stripped:
        deeper = _invert(inverted_stripped)
        if deeper:
            surfaces.append(deeper)

    keys: list[str] = []
    for surface in surfaces:
        for expand in (False, True):
            folded = fold(surface, lang, expand=expand)
            if not folded:
                continue
            keys.append(folded)
            parts = folded.split()
            without_particles = [p for p in parts if p not in _PARTICLES]
            if without_particles and len(without_particles) != len(parts):
                keys.append(" ".join(without_particles))
            stemmed = [genitive_stem(p) or p for p in parts]
            if stemmed != parts:
                keys.append(" ".join(stemmed))
    return list(dict.fromkeys(k for k in keys if k))


def text_tokens(text: Any, lang: str = "nb", *, minimum: int = 3) -> set[str]:
    """Tokens of a free-text blob, with genitive stems and both spellings."""
    found: set[str] = set()
    for expand in (False, True):
        for token in fold(text, lang, expand=expand).split():
            if len(token) < minimum:
                continue
            found.add(token)
            stem = genitive_stem(token)
            if stem and len(stem) >= minimum:
                found.add(stem)
    return found


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

@dataclass
class Card:
    """One compact candidate card — what the model sees instead of the KB."""

    kind: str
    id: str
    label: str
    match_type: str
    score: float
    matched_on: str = ""
    summary: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        """One line for a prompt body: id, label, signals, then the summary."""
        head = f"[{self.id}] {self.label} — {self.match_type} {self.score:.2f}"
        tail = [self.summary] if self.summary else []
        if self.matched_on and fold(self.matched_on) != fold(self.label):
            tail.append(f"matched on: {self.matched_on}")
        tail.extend(self.notes)
        return head + ("\n      " + "\n      ".join(tail) if tail else "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "label": self.label,
            "match_type": self.match_type, "score": self.score,
            "summary": self.summary, **self.fields,
        }


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS resolve_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS resolve_name (
    kind TEXT NOT NULL, entity_id TEXT NOT NULL, variant TEXT NOT NULL,
    raw TEXT NOT NULL, folded TEXT NOT NULL, token_key TEXT NOT NULL,
    head TEXT NOT NULL, flen INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS resolve_name_folded ON resolve_name (kind, folded);
CREATE INDEX IF NOT EXISTS resolve_name_block ON resolve_name (kind, head, flen);
CREATE INDEX IF NOT EXISTS resolve_name_tokens ON resolve_name (kind, token_key);
CREATE INDEX IF NOT EXISTS resolve_name_entity ON resolve_name (kind, entity_id);
CREATE TABLE IF NOT EXISTS resolve_token (
    kind TEXT NOT NULL, token TEXT NOT NULL, entity_id TEXT NOT NULL,
    variant_no INTEGER NOT NULL, n_tokens INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS resolve_token_lookup ON resolve_token (kind, token);
CREATE TABLE IF NOT EXISTS resolve_card (
    kind TEXT NOT NULL, entity_id TEXT NOT NULL, payload TEXT NOT NULL,
    rank INTEGER DEFAULT 0, PRIMARY KEY (kind, entity_id)
);
"""

_FTS = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS resolve_fts USING fts5("
    "folded, kind UNINDEXED, entity_id UNINDEXED, tokenize='unicode61')"
)


@dataclass
class Index:
    """A sidecar SQLite index. Deliberately not a table in the source database.

    Kulturbase's reason generalises: the shipped database is read by a web app
    and two native apps, so adding tables to it drags every consumer into a
    parity check for an index only tooling reads, and the build step would have
    to run before any lookup. A sidecar also works against an already-sealed
    release candidate, which is what most workers actually have.
    """

    conn: sqlite3.Connection
    lang: str = "nb"
    has_fts: bool = True

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def kinds(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT kind FROM resolve_card ORDER BY kind")]

    def card(self, kind: str, entity_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload, rank FROM resolve_card WHERE kind=? AND entity_id=?",
            (kind, str(entity_id)),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        payload["_rank"] = row["rank"]
        return payload


def _open(conn_or_path: str | Path | sqlite3.Connection) -> tuple[sqlite3.Connection, bool]:
    if isinstance(conn_or_path, sqlite3.Connection):
        conn, owned = conn_or_path, False
    else:
        conn, owned = connect(str(conn_or_path)), True
    conn.row_factory = sqlite3.Row
    return conn, owned


def build_index(
    conn_or_path: str | Path | sqlite3.Connection,
    rows: Iterable[Mapping[str, Any]],
    kind: str,
    *,
    lang: str = "nb",
    name_field: str = "name",
    id_field: str = "id",
    alias_field: str = "aliases",
    summary_field: str = "summary",
    rank_field: str = "rank",
) -> Index:
    """Build (or rebuild) the index for one `kind` of entity.

    `rows` are mappings with at least an id and a name. `aliases` (a list of
    strings) are indexed as lower-trust variants; `summary` (keep it under
    ~200 characters) is what the model reads on the card; `rank` breaks ties
    between equal-scoring names by how attested the entity is, so a lone
    surname returns the best-attested bearer first while still refusing to be
    confident about it. Every other key is carried through to the card, where
    `hints` can rerank on it.

    Rebuilding one kind leaves the others alone, so several kinds share one
    sidecar file. Call it again to refresh after the source changes.
    """
    conn, _ = _open(conn_or_path)
    conn.executescript(_SCHEMA)
    has_fts = True
    try:
        conn.execute(_FTS)
    except sqlite3.OperationalError:  # FTS5 not compiled in
        has_fts = False

    for table in ("resolve_name", "resolve_token", "resolve_card"):
        conn.execute(f"DELETE FROM {table} WHERE kind=?", (kind,))
    if has_fts:
        conn.execute("DELETE FROM resolve_fts WHERE kind=?", (kind,))

    names: list[tuple] = []
    token_rows: list[tuple] = []
    cards: list[tuple] = []
    fts_rows: list[tuple] = []

    for row in rows:
        entity_id = str(row[id_field])
        label = row[name_field]
        aliases = [a for a in (row.get(alias_field) or []) if a]
        payload = {
            k: v for k, v in row.items()
            if k not in {id_field, name_field, alias_field, rank_field}
        }
        payload["id"] = entity_id
        payload["label"] = label if isinstance(label, str) else str(label)
        payload.setdefault(summary_field, row.get(summary_field) or "")
        cards.append((kind, entity_id, json.dumps(payload, ensure_ascii=False),
                      int(row.get(rank_field) or 0)))

        surfaces: list[tuple[str, Any]] = [("name", label)]
        surfaces += [("alias", a) for a in aliases]
        variant_no = 0
        seen_keys: set[str] = set()
        for variant, surface in surfaces:
            raw = surface if isinstance(surface, str) else str(surface)
            for key in name_keys(raw, lang):
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                names.append((kind, entity_id, variant, raw, key,
                              token_key(key, lang), key[:1], len(key)))
                if has_fts:
                    fts_rows.append((key, kind, entity_id))
                parts = {t for t in key.split() if len(t) >= _TOKEN_MIN}
                for token in parts:
                    token_rows.append((kind, token, entity_id, variant_no, len(parts)))
                variant_no += 1

    conn.executemany(
        "INSERT INTO resolve_name (kind, entity_id, variant, raw, folded, token_key, head, flen)"
        " VALUES (?,?,?,?,?,?,?,?)", names)
    conn.executemany(
        "INSERT INTO resolve_token (kind, token, entity_id, variant_no, n_tokens)"
        " VALUES (?,?,?,?,?)", token_rows)
    conn.executemany(
        "INSERT INTO resolve_card (kind, entity_id, payload, rank) VALUES (?,?,?,?)", cards)
    if has_fts and fts_rows:
        conn.executemany("INSERT INTO resolve_fts (folded, kind, entity_id) VALUES (?,?,?)", fts_rows)
    conn.execute("INSERT OR REPLACE INTO resolve_meta (key, value) VALUES (?, ?)",
                 (f"lang:{kind}", lang))
    conn.commit()
    return Index(conn=conn, lang=lang, has_fts=has_fts)


def open_index(path: str | Path, *, lang: str = "nb") -> Index:
    """Open an index built earlier. `build_index` returns one directly."""
    conn, _ = _open(path)
    has_fts = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='resolve_fts'").fetchone())
    return Index(conn=conn, lang=lang, has_fts=has_fts)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def _hits(index: Index, kind: str, query: str, *, fuzzy: bool) -> dict[str, tuple[str, float, str]]:
    """entity_id -> (match_type, base score, matched-on string)."""
    lang = index.lang
    found: dict[str, tuple[str, float, str]] = {}

    def offer(entity_id: Any, match_type: str, score: float, matched: str) -> None:
        entity_id = str(entity_id)
        current = found.get(entity_id)
        if current is None or score > current[1]:
            found[entity_id] = (match_type, score, matched)

    raw = query.strip()

    # 1. Byte-identical.
    for row in index.conn.execute(
            "SELECT entity_id, raw FROM resolve_name WHERE kind=? AND raw=?", (kind, raw)):
        offer(row["entity_id"], "exact", 1.0, row["raw"])

    # 2. Folded — either diacritic spelling, either name orientation.
    keys = name_keys(raw, lang)
    if keys:
        marks = ",".join("?" * len(keys))
        for row in index.conn.execute(
                f"SELECT entity_id, raw, variant FROM resolve_name"
                f" WHERE kind=? AND folded IN ({marks})", (kind, *keys)):
            offer(row["entity_id"], "folded",
                  0.97 if row["variant"] == "name" else 0.93, row["raw"])

    query_tokens = {t for key in keys for t in key.split() if len(t) >= 2}

    # 3. Identical token set, any order.
    key = " ".join(sorted(query_tokens)) if len(query_tokens) >= 2 else ""
    if key:
        for row in index.conn.execute(
                "SELECT entity_id, raw FROM resolve_name WHERE kind=? AND token_key=?",
                (kind, key)):
            offer(row["entity_id"], "token", 0.88, row["raw"])

    # 4. Token overlap through the inverted index.
    lookup = [t for t in query_tokens if len(t) >= _TOKEN_MIN]
    if len(query_tokens) >= 2 and lookup:
        marks = ",".join("?" * len(lookup))
        best: dict[str, float] = {}
        for row in index.conn.execute(
                f"SELECT entity_id, n_tokens, count(DISTINCT token) AS shared"
                f" FROM resolve_token WHERE kind=? AND token IN ({marks})"
                f" GROUP BY entity_id, variant_no, n_tokens", (kind, *lookup)):
            shared, total = row["shared"], row["n_tokens"]
            if shared < 2:
                continue
            jaccard = shared / max(1, len(query_tokens) + total - shared)
            entity_id = str(row["entity_id"])
            best[entity_id] = max(best.get(entity_id, 0.0), jaccard)
        for entity_id, jaccard in best.items():
            offer(entity_id, "token_overlap", 0.60 + 0.25 * jaccard, "")

    # 5. Near-miss spellings. Off by default: it is the layer that manufactures
    #    plausible-looking wrong answers, and folding plus token-set matching
    #    recovered only 5% of one campaign's 5,194 held unmatched names — the
    #    gap there was create-policy and missing authority IDs, not fuzziness.
    #    Capped below the confidence threshold when it is on.
    query_folded = fold(raw, lang)
    if fuzzy and len(query_folded) >= _FUZZY_MIN_QUERY_CHARS:
        best_so_far = max((s for _t, s, _m in found.values()), default=0.0)
        if len(query_tokens) >= 2 and best_so_far < 0.9 and lookup:
            marks = ",".join("?" * len(lookup))
            pool = [str(r["entity_id"]) for r in index.conn.execute(
                f"SELECT entity_id, count(DISTINCT token) AS shared FROM resolve_token"
                f" WHERE kind=? AND token IN ({marks})"
                f" GROUP BY entity_id ORDER BY shared DESC LIMIT ?",
                (kind, *lookup, _FUZZY_POOL))]
            if pool:
                marks = ",".join("?" * len(pool))
                for row in index.conn.execute(
                        f"SELECT entity_id, raw, folded FROM resolve_name"
                        f" WHERE kind=? AND entity_id IN ({marks})", (kind, *pool)):
                    ratio = SequenceMatcher(None, query_folded, row["folded"]).ratio()
                    if ratio >= _FUZZY_MIN_RATIO:
                        offer(row["entity_id"], "fuzzy",
                              min(_FUZZY_MAX, 0.50 + 0.38 * ratio), row["raw"])
        best_so_far = max((s for _t, s, _m in found.values()), default=0.0)
        if best_so_far < 0.6 and 0 < len(query_tokens) <= 2:
            # A one- or two-word near-miss ("Galgemannen"/"Galgmannen") shares
            # no whole token, so no inverted index can reach it. Blocked on
            # first letter and length to stay a few hundred comparisons; a typo
            # in the first character is the accepted limit of a fallback.
            slack = max(2, len(query_folded) // 4)
            for row in index.conn.execute(
                    "SELECT entity_id, raw, folded FROM resolve_name"
                    " WHERE kind=? AND head=? AND flen BETWEEN ? AND ? LIMIT ?",
                    (kind, query_folded[:1], len(query_folded) - slack,
                     len(query_folded) + slack, _FUZZY_SCAN_LIMIT)):
                ratio = SequenceMatcher(None, query_folded, row["folded"]).ratio()
                if ratio >= _FUZZY_MIN_RATIO:
                    offer(row["entity_id"], "fuzzy",
                          min(_FUZZY_MAX, 0.50 + 0.38 * ratio), row["raw"])

    # 6. FTS fallback, only when the cheap layers found little.
    if index.has_fts and len(found) < 25 and query_tokens:
        expr = " OR ".join(f'"{t}"' for t in sorted(query_tokens) if len(t) >= _TOKEN_MIN)
        if expr:
            try:
                for row in index.conn.execute(
                        "SELECT entity_id, folded FROM resolve_fts"
                        " WHERE resolve_fts MATCH ? AND kind=? ORDER BY rank LIMIT 200",
                        (expr, kind)):
                    hit = set(row["folded"].split())
                    if not hit:
                        continue
                    jaccard = len(hit & query_tokens) / len(hit | query_tokens)
                    offer(row["entity_id"], "fts", 0.30 + 0.35 * jaccard, "")
            except sqlite3.OperationalError:
                pass
    return found


def _apply_hints(payload: Mapping[str, Any], score: float, hints: Mapping[str, Any],
                 tolerance: int, notes: list[str]) -> float:
    """Rerank on extra evidence; never filter on it.

    A hint that disagrees is a demotion, not a veto: the catalogue is often the
    one that is wrong, and a hard filter hides that.
    """
    for key, given in hints.items():
        if given is None:
            continue
        have = payload.get(key)
        if have is None:
            notes.append(f"{key} unknown in the index")
            continue
        if isinstance(given, (int, float)) and isinstance(have, (int, float)):
            agrees = abs(float(have) - float(given)) <= tolerance
        else:
            agrees = fold(have) == fold(given)
        if agrees:
            score = min(1.0, score + 0.03)
        else:
            score *= 0.35
            notes.append(f"{key} {have!r} != {given!r}")
    return score


def candidates(
    index: Index,
    query: str,
    k: int = 10,
    kind: str | None = None,
    hints: Mapping[str, Any] | None = None,
    *,
    fuzzy: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
    tolerance: int = 1,
) -> list[Card]:
    """The k best candidate cards for `query`, best first.

    `kind=None` searches every indexed kind. `hints` rerank on any card field
    (`{"birth_year": 1828}`); they never filter. `fuzzy=False` keeps the
    near-miss layer off — turn it on only once you have measured that you need
    it. Each card's `notes` say `confident` or `below threshold`, and a card
    within 0.02 of the next one is marked `ambiguous`: an ambiguous top hit is
    the case that needs a model (or a human), and it is the only one that does.
    """
    if not query or not str(query).strip():
        return []
    kinds = [kind] if kind else index.kinds()
    hints = hints or {}
    out: list[Card] = []
    for one in kinds:
        for entity_id, (match_type, score, matched) in _hits(index, one, str(query), fuzzy=fuzzy).items():
            payload = index.card(one, entity_id)
            if payload is None:
                continue
            notes: list[str] = []
            score = _apply_hints(payload, score, hints, tolerance, notes)
            if score <= 0:
                continue
            extras = {kk: vv for kk, vv in payload.items()
                      if kk not in {"id", "label", "summary", "_rank"}}
            out.append(Card(
                kind=one, id=str(payload["id"]), label=str(payload["label"]),
                match_type=match_type, score=round(min(1.0, score), 4),
                matched_on=matched, summary=str(payload.get("summary") or ""),
                fields=extras, rank=int(payload.get("_rank") or 0), notes=notes))
    out.sort(key=lambda c: (-c.score, -c.rank, str(c.id)))
    if len(out) > 1 and out[0].score - out[1].score <= 0.02:
        out[0].notes.append("ambiguous: tied with the next candidate")
    for card in out:
        confident = card.score >= min_score and not any(
            n.startswith("ambiguous") for n in card.notes)
        card.notes.append("confident" if confident else "below threshold")
    return out[:k]


def text_candidates(index: Index, text: str, k: int = 25, kind: str | None = None) -> list[Card]:
    """Every entity whose full name literally occurs in `text`, best first.

    All of an entity's name tokens must be present — the precision rule a
    production person campaign settled on — plus genitive stems, so "Ibsens"
    finds Ibsen. This is the candidate list a packet carries: a hit is a
    *coding candidate*, never by itself evidence that the passage is about it.
    """
    present = text_tokens(text, index.lang)
    lookup = [t for t in present if len(t) >= _TOKEN_MIN]
    if not lookup:
        return []
    kinds = [kind] if kind else index.kinds()
    out: list[Card] = []
    marks = ",".join("?" * len(lookup))
    for one in kinds:
        matched: list[tuple[int, str]] = []
        for row in index.conn.execute(
                f"SELECT entity_id, n_tokens, count(DISTINCT token) AS shared"
                f" FROM resolve_token WHERE kind=? AND token IN ({marks})"
                f" GROUP BY entity_id, variant_no, n_tokens", (one, *lookup)):
            if row["shared"] >= row["n_tokens"]:
                matched.append((row["n_tokens"], str(row["entity_id"])))
        matched.sort(key=lambda x: (-x[0], x[1]))
        seen: set[str] = set()
        for n_tokens, entity_id in matched[: k * 2]:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            payload = index.card(one, entity_id)
            if payload is None:
                continue
            extras = {kk: vv for kk, vv in payload.items()
                      if kk not in {"id", "label", "summary", "_rank"}}
            out.append(Card(
                kind=one, id=str(payload["id"]), label=str(payload["label"]),
                match_type="token", score=round(min(0.95, 0.55 + 0.1 * n_tokens), 4),
                matched_on=str(payload["label"]), summary=str(payload.get("summary") or ""),
                fields=extras, rank=int(payload.get("_rank") or 0),
                notes=[f"{n_tokens} name tokens present"]))
    out.sort(key=lambda c: (-c.score, -c.rank, str(c.id)))
    return out[:k]


# ---------------------------------------------------------------------------
# Slot enums
# ---------------------------------------------------------------------------

def slot_enum(
    cards: Sequence[Card] | Sequence[Mapping[str, Any]],
    n_slots: int,
    *,
    prefix: str = "c",
    none_slot: str = NONE_SLOT,
    description: str = "",
) -> tuple[dict[str, Any], dict[str, str]]:
    """A fixed-width slot enum and the map back to real ids.

    The enum lists **every** slot `c01..cNN` regardless of how many candidates
    this packet actually has, so the JSON schema is byte-identical across the
    batch and the provider's prefix cache engages (measured 0% -> 58% cached
    input on one batch when the per-packet enum was replaced by slots). Slots
    beyond the supplied cards are unused and are refused by `unslot`.

    Returns `(schema_fragment, slot_map)`. Put the fragment in your response
    schema; keep the slot map beside the packet; never send the map.
    """
    if n_slots < len(cards):
        raise ValueError(
            f"{len(cards)} candidates exceed {n_slots} slots — cut the packet "
            "or raise n_slots for the whole batch (not for this packet alone, "
            "which would vary the schema again)")
    width = max(2, len(str(n_slots)))
    slot_map: dict[str, str] = {}
    for ordinal, card in enumerate(cards, start=1):
        slot = f"{prefix}{ordinal:0{width}d}"
        slot_map[slot] = str(card.id if isinstance(card, Card) else card["id"])
    all_slots = [f"{prefix}{i:0{width}d}" for i in range(1, n_slots + 1)]
    fragment = {
        "type": "string",
        "enum": [none_slot, *all_slots],
        "description": description or (
            f"Exactly one candidate slot supplied in this packet, or "
            f"{none_slot!r} when none of them is right. Never invent an "
            f"identifier; only slots this packet lists may be used."),
    }
    return fragment, slot_map


def unslot(
    items: Iterable[Mapping[str, Any]],
    slot_map: Mapping[str, str],
    *,
    field: str = "ref",
    id_field: str | None = None,
    none_slot: str = NONE_SLOT,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Map slots back to real ids; refuse items citing a slot this packet never
    supplied.

    Returns `(resolved, problems)`. An item that used `none_slot` is kept with
    its id set to `None` — "none of these" is a wanted answer, not a failure,
    and scoring coverage without it is how 130 fabricated bios reached
    production.
    """
    id_field = id_field or f"{field}_id"
    resolved: list[dict[str, Any]] = []
    problems: list[str] = []
    for position, item in enumerate(items):
        slot = item.get(field)
        if slot == none_slot or slot is None:
            resolved.append({**item, id_field: None})
            continue
        if slot not in slot_map:
            problems.append(
                f"item {position}: {field}={slot!r} is not a slot this packet supplied")
            continue
        resolved.append({**item, id_field: slot_map[slot]})
    return resolved, problems
