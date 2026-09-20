"""Hippocampus - proposal system, cascade merges, deduplication, and data validation."""

from .apply import (
    MISSING, apply_proposal, enum_member, regex, wikidata_exists, wikidata_type_is,
)
from .proposals import Proposal, Change, ProposalStore
from .resolve import (
    Card, Index, build_index, candidates, fold, name_keys, open_index,
    slot_enum, text_candidates, unslot,
)
from .cascade import ReferenceSpec, ReferenceGraph, find_references, apply_merge, apply_delete
from .dedup import (
    VetoGate, CandidatePair, ExclusionList, VetoMatcher,
    exact_field, initial_match, no_conflict, gender_check, reference_ratio,
)
from .validate import ValidationResult, Rule, Validator, required_field, valid_values, reference_exists, no_orphans, conditional_required
from .store import YAMLStore
from .wikidata_resolve import (
    WikidataResolver,
    Resolution,
    ScoredCandidate,
    validate_chosen_qid,
    TYPE_HINT_P31,
    COHERENCE_PROPERTIES,
    DEFAULT_WEIGHTS,
)

__all__ = [
    "MISSING",
    "apply_proposal",
    "enum_member",
    "regex",
    "wikidata_exists",
    "wikidata_type_is",
    "Card",
    "Index",
    "build_index",
    "open_index",
    "candidates",
    "text_candidates",
    "fold",
    "name_keys",
    "slot_enum",
    "unslot",
    "Proposal",
    "Change",
    "ProposalStore",
    "ReferenceSpec",
    "ReferenceGraph",
    "find_references",
    "apply_merge",
    "apply_delete",
    "VetoGate",
    "CandidatePair",
    "ExclusionList",
    "VetoMatcher",
    "exact_field",
    "initial_match",
    "no_conflict",
    "gender_check",
    "reference_ratio",
    "ValidationResult",
    "Rule",
    "Validator",
    "required_field",
    "valid_values",
    "reference_exists",
    "no_orphans",
    "conditional_required",
    "YAMLStore",
    "WikidataResolver",
    "Resolution",
    "ScoredCandidate",
    "validate_chosen_qid",
    "TYPE_HINT_P31",
    "COHERENCE_PROPERTIES",
    "DEFAULT_WEIGHTS",
]
