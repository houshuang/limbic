"""Hippocampus - proposal system, cascade merges, deduplication, and data validation.

Names resolve on first access, so importing one submodule (`hippocampus.resolve`
is standard-library only) does not load yaml, numpy or the Wikidata client.
"""

from importlib import import_module

_EXPORTS = {
    "apply": (
        "MISSING", "apply_proposal", "enum_member", "regex", "wikidata_exists",
        "wikidata_type_is",
    ),
    "proposals": (
        "Proposal", "Change", "ProposalStore",
    ),
    "resolve": (
        "Card", "Index", "build_index", "candidates", "fold", "name_keys", "open_index",
        "slot_enum", "text_candidates", "unslot", "invert_name", "strip_parenthetical",
        "text_tokens", "tokens", "token_key", "genitive_stem",
    ),
    "cascade": (
        "ReferenceSpec", "ReferenceGraph", "find_references", "apply_merge", "apply_delete",
    ),
    "dedup": (
        "VetoGate", "CandidatePair", "ExclusionList", "VetoMatcher", "exact_field",
        "initial_match", "no_conflict", "gender_check", "reference_ratio",
    ),
    "validate": (
        "ValidationResult", "Rule", "Validator", "required_field", "valid_values",
        "reference_exists", "no_orphans", "conditional_required",
    ),
    "store": (
        "YAMLStore",
    ),
    "wikidata_resolve": (
        "WikidataResolver", "Resolution", "ScoredCandidate", "validate_chosen_qid",
        "TYPE_HINT_P31", "COHERENCE_PROPERTIES", "DEFAULT_WEIGHTS",
    ),
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = list(_MODULE_OF)


def __getattr__(name: str):
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
