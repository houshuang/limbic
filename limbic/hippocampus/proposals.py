"""Generalized proposal system for managing data changes with human review.

Proposals capture intended modifications (modify, merge, delete) as serializable
records that move through a lifecycle: pending -> approved -> applied (or rejected).
Each proposal records current state, proposed state, reasoning, and cascade scope.
"""

from __future__ import annotations

import ast
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Change:
    """A single entity-level change within a proposal."""
    entity_type: str
    entity_id: str
    action: str  # modify | merge | delete
    current_state: dict[str, Any] = field(default_factory=dict)
    proposed_state: dict[str, Any] = field(default_factory=dict)
    changed_fields: list[dict[str, Any]] = field(default_factory=list)
    merge_target: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.merge_target is None:
            d.pop("merge_target", None)
        if not self.changed_fields:
            d.pop("changed_fields", None)
        if not self.proposed_state:
            d.pop("proposed_state", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Change:
        return cls(
            entity_type=data["entity_type"],
            entity_id=str(data["entity_id"]),
            action=data["action"],
            current_state=data.get("current_state", {}),
            proposed_state=data.get("proposed_state", {}),
            changed_fields=data.get("changed_fields", []),
            merge_target=str(data["merge_target"]) if data.get("merge_target") is not None else None,
        )


@dataclass
class Proposal:
    """A reviewable data-change proposal."""
    id: str
    created_at: str
    status: str  # pending | approved | applied | rejected
    title: str
    reasoning: str
    changes: list[Change] = field(default_factory=list)
    cascade_scope: dict[str, Any] = field(default_factory=dict)
    category: str = "data_quality"
    created_by: str = "system"

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.id,
            "created_at": self.created_at,
            "status": self.status,
            "title": self.title,
            "reasoning": self.reasoning,
            "category": self.category,
            "created_by": self.created_by,
            "changes": [c.to_dict() for c in self.changes],
            "cascade_scope": self.cascade_scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        changes = [Change.from_dict(c) for c in data.get("changes", [])]
        return cls(
            id=data["proposal_id"],
            created_at=data.get("created_at", ""),
            status=data.get("status", "pending"),
            title=data.get("title", ""),
            reasoning=data.get("reasoning", ""),
            changes=changes,
            cascade_scope=data.get("cascade_scope", {}),
            category=data.get("category", "data_quality"),
            created_by=data.get("created_by", "system"),
        )


# ---------------------------------------------------------------------------
# Field value parsing
# ---------------------------------------------------------------------------

def parse_field_value(raw: str) -> Any:
    """Parse a field value string into the appropriate Python type.

    Handles arrays/dicts via ast.literal_eval, integers, booleans, and null.
    Falls back to the raw string if nothing else matches.
    """
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
    if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
        return int(raw)
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null" or low == "none":
        return None
    return raw


# ---------------------------------------------------------------------------
# YAML helpers (local, thin wrappers)
# ---------------------------------------------------------------------------

class _YAMLDumper(yaml.SafeDumper):
    pass

def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

_YAMLDumper.add_representer(str, _str_representer)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, Dumper=_YAMLDumper, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Proposal store
# ---------------------------------------------------------------------------

def _generate_id() -> str:
    return f"prop_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _safe_filename(proposal_id: str, title: str) -> str:
    slug = title.lower().replace(" ", "_")[:40]
    # Remove characters that are unsafe in filenames
    slug = "".join(c for c in slug if c.isalnum() or c in ("_", "-"))
    return f"{proposal_id}_{slug}.yaml"


class ProposalStore:
    """Manages proposals on disk across lifecycle directories.

    This is the filing cabinet, not the lock: it has no preimage check, so an
    `approved` status here is a string in a file, not a guarantee that the
    record still looks the way the proposer saw it. Use
    `limbic.hippocampus.apply.apply_proposal` for the write itself.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base = Path(base_dir)
        self._dirs = {
            "pending": self.base / "pending",
            "approved": self.base / "approved",
            "applied": self.base / "applied",
            "rejected": self.base / "rejected",
        }
        for d in self._dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # -- creation helpers ---------------------------------------------------

    def create_modify(
        self,
        entity_ref: str,
        field_changes: dict[str, str],
        title: str,
        reasoning: str,
        *,
        current_state: dict[str, Any] | None = None,
        category: str = "data_quality",
        created_by: str = "system",
    ) -> Proposal:
        """Create a modify proposal for a single entity."""
        entity_type, entity_id = _parse_ref(entity_ref)
        state = dict(current_state) if current_state else {}
        changed: list[dict[str, Any]] = []
        proposed = dict(state)
        for fname, raw_value in field_changes.items():
            value = parse_field_value(raw_value) if isinstance(raw_value, str) else raw_value
            changed.append({"field": fname, "old_value": state.get(fname), "new_value": value})
            proposed[fname] = value

        change = Change(
            entity_type=entity_type,
            entity_id=entity_id,
            action="modify",
            current_state=state,
            proposed_state=proposed,
            changed_fields=changed,
        )
        proposal = Proposal(
            id=_generate_id(),
            created_at=datetime.now().isoformat(),
            status="pending",
            title=title,
            reasoning=reasoning,
            changes=[change],
            category=category,
            created_by=created_by,
            cascade_scope={"description": f"Modify {entity_ref}"},
        )
        self._write(proposal)
        return proposal

    def create_merge(
        self,
        source_ref: str,
        target_ref: str,
        title: str,
        reasoning: str,
        *,
        source_state: dict[str, Any] | None = None,
        category: str = "duplicates",
        created_by: str = "system",
    ) -> Proposal:
        """Create a merge proposal (source into target)."""
        source_type, source_id = _parse_ref(source_ref)
        target_type, target_id = _parse_ref(target_ref)
        if source_type != target_type:
            raise ValueError(f"Cannot merge different types: {source_type} and {target_type}")

        change = Change(
            entity_type=source_type,
            entity_id=source_id,
            action="merge",
            current_state=source_state or {},
            merge_target=target_id,
        )
        proposal = Proposal(
            id=_generate_id(),
            created_at=datetime.now().isoformat(),
            status="pending",
            title=title,
            reasoning=reasoning,
            changes=[change],
            category=category,
            created_by=created_by,
            cascade_scope={"description": f"Merge {source_ref} into {target_ref}"},
        )
        self._write(proposal)
        return proposal

    def create_delete(
        self,
        entity_ref: str,
        title: str,
        reasoning: str,
        *,
        current_state: dict[str, Any] | None = None,
        category: str = "cleanup",
        created_by: str = "system",
    ) -> Proposal:
        """Create a delete proposal."""
        entity_type, entity_id = _parse_ref(entity_ref)
        change = Change(
            entity_type=entity_type,
            entity_id=entity_id,
            action="delete",
            current_state=current_state or {},
        )
        proposal = Proposal(
            id=_generate_id(),
            created_at=datetime.now().isoformat(),
            status="pending",
            title=title,
            reasoning=reasoning,
            changes=[change],
            category=category,
            created_by=created_by,
            cascade_scope={"description": f"Delete {entity_ref}"},
        )
        self._write(proposal)
        return proposal

    # -- lifecycle ----------------------------------------------------------

    def list_pending(self) -> list[Proposal]:
        """Return all pending proposals."""
        return self._list_dir("pending")

    def list_approved(self) -> list[Proposal]:
        """Return all approved proposals."""
        return self._list_dir("approved")

    def load(self, proposal_id: str) -> Proposal | None:
        """Load a proposal by ID from any lifecycle directory."""
        for status_dir in self._dirs.values():
            for path in status_dir.glob(f"{proposal_id}_*.yaml"):
                return Proposal.from_dict(_load_yaml(path))
        return None

    def approve(self, proposal_id: str) -> Proposal:
        """Move a proposal from pending to approved."""
        return self._move(proposal_id, "pending", "approved", new_status="approved")

    def reject(self, proposal_id: str) -> Proposal:
        """Move a proposal from pending to rejected."""
        return self._move(proposal_id, "pending", "rejected", new_status="rejected")

    def mark_applied(self, proposal_id: str) -> Proposal:
        """Move a proposal from approved to applied."""
        return self._move(proposal_id, "approved", "applied", new_status="applied")

    # -- internal -----------------------------------------------------------

    def _write(self, proposal: Proposal) -> Path:
        filename = _safe_filename(proposal.id, proposal.title)
        path = self._dirs[proposal.status] / filename
        _save_yaml(path, proposal.to_dict())
        return path

    def _list_dir(self, status: str) -> list[Proposal]:
        proposals: list[Proposal] = []
        for path in sorted(self._dirs[status].glob("*.yaml")):
            try:
                proposals.append(Proposal.from_dict(_load_yaml(path)))
            except Exception:
                continue
        return proposals

    def _move(self, proposal_id: str, from_status: str, to_status: str, *, new_status: str) -> Proposal:
        src_dir = self._dirs[from_status]
        matches = list(src_dir.glob(f"{proposal_id}_*.yaml"))
        if not matches:
            raise FileNotFoundError(f"No {from_status} proposal with id {proposal_id}")
        src_path = matches[0]
        data = _load_yaml(src_path)
        data["status"] = new_status
        dst_path = self._dirs[to_status] / src_path.name
        _save_yaml(dst_path, data)
        src_path.unlink()
        return Proposal.from_dict(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ref(ref: str) -> tuple[str, str]:
    """Parse 'type/id' entity reference into (type, id)."""
    if "/" not in ref:
        raise ValueError(f"Invalid entity reference '{ref}': expected 'type/id'")
    entity_type, entity_id = ref.split("/", 1)
    return entity_type, entity_id
