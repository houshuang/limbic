"""Temporal data utilities for historical/knowledge workloads.

Two concerns live here:

1. Parsing uncertain date expressions into integer year ranges — "10th century",
   "940s", "circa 942", "c. 942", "942-996", "4th century BC", plus full EDTF
   strings when the optional `edtf` package is installed.

2. Allen interval relations and soft temporal plausibility scoring — "was X
   alive when Y happened?", "is candidate date consistent with context?".

The DateRange shape uses integer years (negative for BC). This is coarser than
full EDTF but indexes trivially and covers every query the entity resolver
actually needs (century-level plausibility, "lived during X", "overlapping
reigns"). Precise month/day handling is out of scope; add at the call site
where it's needed.

Year-numbering convention: we use the **historical** convention — there is
no year zero. `1 AD` → `1`, `1 BC` → `-1`, `44 BC` → `-44`. This mismatches
the astronomical convention used by Wikidata (`1 BC` = `0`, `2 BC` = `-1`).
Convert at the Wikidata integration boundary if needed: astronomical_year
`-44` = historical `-43 BC`, and historical `-44` = astronomical `-43`. For
century-scale plausibility scoring this one-year offset is well below the
decay scale and ignorable; for exact birth/death dates, convert explicitly.

Design note — soft temporal filtering (DELICATE, 2025): historical texts
reference people across eras ("Machiavelli wrote about Caesar"). Never hard-
exclude by date; use `plausibility_score` as a decaying scoring signal instead.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


def _try_parse_edtf(text: str):
    """Attempt to parse text as EDTF. Returns EDTF object or None.

    Optional dependency: `pip install limbic[temporal]` installs `edtf`.
    When unavailable or parsing fails, returns None and the caller falls back
    to the regex-based parsers below.
    """
    try:
        from edtf import parse_edtf  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return parse_edtf(text)
    except Exception:
        return None


@dataclass
class DateRange:
    """Inclusive integer-year range with optional precision flags.

    For "940s": start=940, end=949.
    For "circa 942": start=942, end=942, approximate=True.
    For "-44" (Caesar's death): start=-44, end=-44.
    For "4th century BC": start=-400, end=-301.

    `original` preserves the source string for debugging/display.
    """

    start: int
    end: int
    approximate: bool = False
    uncertain: bool = False
    original: str | None = None

    def contains(self, year: int) -> bool:
        return self.start <= year <= self.end

    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    def span_years(self) -> int:
        return self.end - self.start + 1


# Order matters: BC patterns must match before AD, "circa" before plain year.
_CENTURY_BC_RE = re.compile(
    r"^\s*(\d+)(?:st|nd|rd|th)?\s*century\s*(?:BC|BCE)\s*$", re.IGNORECASE
)
_CENTURY_AD_RE = re.compile(
    r"^\s*(\d+)(?:st|nd|rd|th)?\s*century(?:\s*(?:AD|CE))?\s*$", re.IGNORECASE
)
_DECADE_RE = re.compile(r"^\s*(\d{3,4})s\s*$")
_CIRCA_RE = re.compile(
    r"^\s*(?:c\.?|ca\.?|circa|~|≈)\s*(-?\d+)\s*$", re.IGNORECASE
)
_RANGE_RE = re.compile(r"^\s*(-?\d+)\s*[-–—/]\s*(-?\d+)\s*$")
_YEAR_BC_RE = re.compile(r"^\s*(\d+)\s*(?:BC|BCE)\s*$", re.IGNORECASE)
_YEAR_AD_RE = re.compile(r"^\s*(\d+)\s*(?:AD|CE)\s*$", re.IGNORECASE)


def parse_date(text: str) -> DateRange | None:
    """Parse a date expression into a DateRange. Returns None on failure.

    Handles: plain year ("1095", "-44"), AD/BC suffix ("44 BC", "1095 AD"),
    ranges ("942-996", "942/996"), decades ("940s"), centuries ("10th century",
    "4th century BC"), circa ("circa 942", "c. 942", "~942"), and EDTF strings
    when the optional `edtf` package is installed.
    """
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None

    # BC/BCE century first — otherwise the AD regex would swallow it
    m = _CENTURY_BC_RE.match(t)
    if m:
        c = int(m.group(1))
        # Nth century BC: years -(c*100) through -((c-1)*100 + 1)
        # e.g. 4th century BC = 400 BC to 301 BC = -400 to -301
        return DateRange(start=-(c * 100), end=-((c - 1) * 100 + 1), original=t)

    m = _CENTURY_AD_RE.match(t)
    if m:
        c = int(m.group(1))
        # Nth century AD: years (c-1)*100 + 1 through c*100
        # e.g. 10th century = 901 to 1000
        return DateRange(start=(c - 1) * 100 + 1, end=c * 100, original=t)

    m = _DECADE_RE.match(t)
    if m:
        base = int(m.group(1))
        return DateRange(start=base, end=base + 9, original=t)

    m = _CIRCA_RE.match(t)
    if m:
        y = int(m.group(1))
        return DateRange(start=y, end=y, approximate=True, original=t)

    m = _RANGE_RE.match(t)
    if m:
        return DateRange(start=int(m.group(1)), end=int(m.group(2)), original=t)

    m = _YEAR_BC_RE.match(t)
    if m:
        return DateRange(start=-int(m.group(1)), end=-int(m.group(1)), original=t)

    m = _YEAR_AD_RE.match(t)
    if m:
        return DateRange(start=int(m.group(1)), end=int(m.group(1)), original=t)

    # Bare integer (possibly negative for BC)
    try:
        y = int(t)
        return DateRange(start=y, end=y, original=t)
    except ValueError:
        pass

    # EDTF fallback
    edtf_obj = _try_parse_edtf(t)
    if edtf_obj is not None:
        try:
            lo = edtf_obj.lower_strict().tm_year
            hi = edtf_obj.upper_strict().tm_year
            return DateRange(start=lo, end=hi, original=t)
        except Exception:
            pass

    return None


# --- Allen interval relations (subset) ---
# The full Allen algebra has 13 relations; we implement the 6 that appear in
# knowledge-graph queries. Add more when a concrete query needs them.


def before(a: DateRange, b: DateRange) -> bool:
    """a is entirely before b (a.end < b.start)."""
    return a.end < b.start


def after(a: DateRange, b: DateRange) -> bool:
    """a is entirely after b."""
    return a.start > b.end


def during(a: DateRange, b: DateRange) -> bool:
    """a is fully contained within b (inclusive)."""
    return a.start >= b.start and a.end <= b.end


def overlaps(a: DateRange, b: DateRange) -> bool:
    """a and b share at least one year."""
    return not (a.end < b.start or b.end < a.start)


def meets(a: DateRange, b: DateRange) -> bool:
    """a ends exactly where b begins (a.end + 1 == b.start)."""
    return a.end + 1 == b.start


def equals(a: DateRange, b: DateRange) -> bool:
    """Allen's temporal equals: same start and end years.

    Intentionally ignores `approximate`, `uncertain`, and `original` — those
    are metadata about how the range was derived, not about the temporal
    extent itself. For full data-equality use the dataclass `==` operator.
    """
    return a.start == b.start and a.end == b.end


def plausibility_score(
    candidate: DateRange,
    context: DateRange,
    penalty_scale_years: float = 100.0,
) -> float:
    """Soft temporal consistency score in [0, 1].

    Returns 1.0 when candidate and context overlap, decaying exponentially as
    the gap between them grows. `penalty_scale_years` sets the decay — a gap
    of `penalty_scale_years` years produces a score of 1/e ≈ 0.37.

    This is a scoring signal, not a filter. Historical texts routinely
    reference entities across eras ("Machiavelli wrote about Caesar"). Hard
    filtering by date would break these cases; soft scoring preserves them
    while still preferring date-consistent candidates.

    Following the DELICATE (2025) approach for historical entity linking.
    """
    if penalty_scale_years <= 0:
        raise ValueError("penalty_scale_years must be positive")
    if overlaps(candidate, context):
        return 1.0
    if candidate.end < context.start:
        gap = context.start - candidate.end
    else:
        gap = candidate.start - context.end
    return math.exp(-gap / penalty_scale_years)
