"""Hippocampus - proposal system, cascade merges, deduplication, and data validation."""

from .proposals import Proposal, Change, ProposalStore
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
