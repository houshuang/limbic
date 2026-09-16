# limbic.hippocampus

**Proposal-based data change management with cascade merges, deduplication, and validation.**

Hippocampus is the change management layer of limbic. It handles the messy reality of maintaining datasets where entities reference each other: you can't just rename a person without updating every performance that credits them, can't merge two duplicate records without relinking all references, can't apply bulk changes without a review trail.

These patterns were extracted from kulturperler (a Nordic performing arts archive with 10,000+ entities across persons, works, performances, and episodes) and otak/alif (a claims-first knowledge system), where every data improvement — fixing a name, merging duplicates found by amygdala's clustering, deleting orphaned records — needed to be proposed, reviewed, and applied with full cascade handling.

## Install

```bash
pip install "limbic[hippocampus]"
```

**Requirements:** pyyaml >= 6.0 (on top of limbic core).

---

## Modules

| Module | What it does |
|--------|-------------|
| **proposals** | `Proposal`, `Change`, `ProposalStore` — lifecycle management for data changes (pending → approved → applied/rejected) |
| **cascade** | `ReferenceGraph`, `apply_merge`, `apply_delete` — declarative reference relinking when entities merge or get deleted |
| **dedup** | `VetoMatcher`, `VetoGate`, `ExclusionList` — composable veto-gate filtering for candidate duplicate pairs |
| **validate** | `Validator`, `Rule` — composable validation rules that check entities and produce errors/warnings |
| **store** | `YAMLStore` — file-locked, atomic YAML storage with typed entity access |
| **wikidata_resolve** | `WikidataResolver`, `Resolution`, `validate_chosen_qid` — deterministic mention → QID resolution scored across five weighted heuristics, returning an audit record or `status="ambiguous"` rather than a guess |

---

## Proposals (`proposals.py`)

Every data change goes through a proposal lifecycle: **pending → approved → applied** (or **rejected**). This creates an audit trail and enables human review before changes are applied.

### Creating proposals

```python
from limbic.hippocampus import ProposalStore

store = ProposalStore("data/proposals")
# Creates pending/, approved/, applied/, rejected/ directories automatically

# Modify an entity
store.create_modify(
    "person/42",
    field_changes={"name": "Henrik Ibsen", "birth_year": "1828"},
    title="Fix Ibsen birth year",
    reasoning="Was incorrectly listed as 1829",
    current_state={"name": "Henrik Ibsen", "birth_year": 1829},
)

# Merge two entities (source into target)
store.create_merge(
    "person/99", "person/42",
    title="Merge duplicate Ibsen",
    reasoning="Same person, different records from two import batches",
)

# Delete an entity
store.create_delete(
    "work/879",
    title="Remove orphaned work",
    reasoning="No performances reference this work",
)
```

### Lifecycle management

```python
# List and review
proposals = store.list_pending()
proposal = store.load("prop_20260322_143000_a1b2c3")

# Approve, reject, or mark applied
store.approve(proposal.id)      # pending → approved
store.reject(proposal.id)       # pending → rejected
store.mark_applied(proposal.id) # approved → applied
```

### Data model

**`Proposal`** — A reviewable data-change proposal:
- `id` — Auto-generated: `prop_{YYYYMMDD_HHMMSS}_{random_hex}`
- `status` — `"pending"`, `"approved"`, `"applied"`, `"rejected"`
- `title`, `reasoning` — Human-readable description
- `changes` — List of `Change` objects
- `cascade_scope` — Describes what the change affects
- `category` — `"data_quality"`, `"duplicates"`, `"cleanup"`, etc.
- `created_by` — User or system that created it

**`Change`** — A single entity-level change within a proposal:
- `entity_type`, `entity_id` — What's being changed
- `action` — `"modify"`, `"merge"`, or `"delete"`
- `current_state`, `proposed_state` — Before/after snapshots
- `changed_fields` — List of `{field, old, new}` dicts
- `merge_target` — For merge actions, the target entity ID

### Storage format

Proposals are stored as YAML files: `{proposal_id}_{slug}.yaml`. The slug is derived from the title (40 chars max, alphanumeric + underscores/dashes). Each lifecycle status has its own directory, and moving a proposal between statuses moves the file.

---

## Cascade merges (`cascade.py`)

When merging duplicate entities, all references must be relinked. The cascade module handles this **declaratively** — you describe how entity types reference each other, and it handles the relinking automatically.

### Declaring reference relationships

```python
from limbic.hippocampus import ReferenceSpec, ReferenceGraph

graph = ReferenceGraph([
    # A performance has a work_id field referencing a work
    ReferenceSpec("performance", "work_id", "work"),

    # A performance has a credits array, each item has a person_id
    ReferenceSpec("performance", "credits", "person", is_array=True, sub_field="person_id"),

    # A work has a playwrights array of person IDs
    ReferenceSpec("work", "playwrights", "person", is_array=True),

    # An episode references a performance
    ReferenceSpec("episode", "performance_id", "performance"),
])
```

### Merging entities

```python
from limbic.hippocampus import apply_merge

# Merge person/99 into person/42
# Automatically relinks ALL performances, works, episodes that referenced person/99
changes = apply_merge(
    graph,
    source_id="99", target_id="42", entity_type="person",
    data_loader=my_loader, data_writer=my_writer, data_deleter=my_deleter,
)
# changes: ["Relinked performance/301.credits: 99 -> 42", "Deleted person/99"]
```

### Deleting entities

```python
from limbic.hippocampus import apply_delete

# Delete with safety check — raises ValueError if anything still references it
changes = apply_delete(graph, "879", "work", my_loader, my_deleter)

# Force delete even with dangling references
changes = apply_delete(graph, "879", "work", my_loader, my_deleter, force=True)
```

### Finding references

```python
from limbic.hippocampus import find_references

# Find everything that references person/42
refs = find_references(graph, "person", "42", my_loader)
# -> [Reference(source_type="performance", source_id="301", field="credits", target_id="42"), ...]
```

### Storage-agnostic design

The cascade engine uses three callback functions — it doesn't care where your data lives:

- `data_loader(entity_type) -> Iterator[(id, data)]` — Iterate all entities of a type
- `data_writer(entity_type, entity_id, data)` — Write an entity
- `data_deleter(entity_type, entity_id)` — Delete an entity

For YAML files, use the built-in `YAMLStore`. For SQLite, databases, or APIs — write your own callbacks.

### Reference types supported

| Pattern | Declaration | Example |
|---------|------------|---------|
| Scalar reference | `ReferenceSpec("perf", "work_id", "work")` | `{"work_id": "42"}` |
| Array of IDs | `ReferenceSpec("work", "playwrights", "person", is_array=True)` | `{"playwrights": ["1", "2"]}` |
| Array of dicts | `ReferenceSpec("perf", "credits", "person", is_array=True, sub_field="person_id")` | `{"credits": [{"person_id": "1", "role": "actor"}]}` |

Type preservation: if the original reference was an integer, the relinked reference stays an integer.

---

## Deduplication (`dedup.py`)

Candidate duplicate pairs (from fuzzy matching, embedding distance, etc.) pass through a chain of **veto gates**. Any gate can reject a pair. This keeps false-positive control explicit and auditable.

### Building a gate chain

```python
from limbic.hippocampus import VetoMatcher, CandidatePair, ExclusionList
from limbic.hippocampus import exact_field, initial_match, no_conflict, gender_check, reference_ratio

matcher = VetoMatcher(
    gates=[
        initial_match("name"),           # first letter must match (case-insensitive)
        exact_field("birth_year"),       # if both have birth_year, must agree
        no_conflict("wikidata_id"),      # conflicting external IDs = not same entity
        gender_check("name", male_names={"erik", "hans"}, female_names={"anna", "grete"}),
    ],
    exclusions=ExclusionList(),  # known false positives
)
```

### Filtering candidates

```python
pair = CandidatePair(
    id_a="42", id_b="99",
    fields_a={"name": "Henrik Ibsen", "birth_year": 1828},
    fields_b={"name": "Henrik J. Ibsen", "birth_year": 1828},
    score=0.95,
)

result = matcher.check_pair(pair)
# result.accepted = True, result.reason = "passed all gates"

# Batch filtering
results = matcher.filter(all_candidates)
accepted = [r for r in results if r.accepted]
```

### Built-in gates

| Gate | What it checks | Rejects when |
|------|---------------|--------------|
| `exact_field(field)` | Field values match (if both present) | Values differ |
| `initial_match(field)` | First character matches (case-insensitive) | First chars differ |
| `no_conflict(field)` | No conflicting values | Both have values and they differ |
| `gender_check(name_field, male, female)` | Gender consistency from first names | One name is male, other is female |
| `reference_ratio(min_ratio, max_minor)` | One entity dominates in references | Neither dominates sufficiently |

### Custom gates

A gate is just a function returning `(accepted: bool, reason: str)`:

```python
from limbic.hippocampus import VetoGate

def my_gate(fields_a, fields_b):
    if fields_a.get("country") != fields_b.get("country"):
        return False, "different countries"
    return True, ""

matcher = VetoMatcher(gates=[VetoGate("country_check", my_gate)])
```

### Exclusion list

Known false positives that should never merge:

```python
exclusions = ExclusionList()
exclusions.add("42", "99")  # These are definitely different people
# Order doesn't matter — uses frozenset internally
```

### Design notes

- **Short-circuit evaluation:** Checking stops at the first rejection — no need to run all gates if the first one says no.
- **Exclusion list checked first** before any gate runs.
- **Pairs are unordered:** `(42, 99)` and `(99, 42)` are the same pair.

---

## Validation (`validate.py`)

Composable rules that check entities and produce errors or warnings. Run a `Validator` against your full dataset to find all issues at once.

```python
from limbic.hippocampus import Validator, required_field, valid_values, reference_exists, no_orphans, conditional_required

validator = Validator([
    required_field("work", "title"),
    valid_values("work", "category", {"teater", "opera", "konsert", "film"}),
    reference_exists("performance", "work_id", "work"),
    no_orphans("person", [("work", "playwrights"), ("performance", "credits")]),
    conditional_required("work", lambda d: d.get("category") == "opera", "composers",
                         condition_label="category is opera"),
])

# entities: {type: {id: data}}
result = validator.validate(entities)
print(result.summary())   # "3 errors, 1 warnings"
print(result.ok)           # False (errors present)
for msg in result.errors:
    print(msg)             # "work/42: missing required field 'title'"
```

### Built-in rules

| Rule | Purpose | Default severity |
|------|---------|-----------------|
| `required_field(type, field)` | Field must be present and non-empty | error |
| `valid_values(type, field, allowed)` | Field value must be in allowed set | error |
| `reference_exists(source, field, target)` | Referenced entity must exist | error |
| `no_orphans(type, referenced_by)` | Entity must be referenced by at least one other | warning |
| `conditional_required(type, cond_fn, field)` | Field required only when condition is true | error |

### Custom rules

A rule is a `Rule(name, check_fn, entity_type, severity)` where `check_fn(etype, eid, data, all_entities) -> list[str]` returns a list of messages (empty if valid):

```python
from limbic.hippocampus import Rule

def check_year_range(etype, eid, data, all_entities):
    year = data.get("year")
    if year and (year < 1800 or year > 2030):
        return [f"year {year} is out of expected range"]
    return []

rule = Rule("year_range", check_year_range, entity_type="work", severity="warning")
```

### ValidationResult

- `result.errors` — List of error messages
- `result.warnings` — List of warning messages
- `result.ok` — True if no errors (warnings are acceptable)
- `result.summary()` — `"3 errors, 1 warnings"`
- `result.merge(other)` — Combine two results

---

## YAML store (`store.py`)

File-locked, atomic YAML storage with typed entity access. Each entity type maps to a subdirectory, each entity is a single YAML file.

```python
from limbic.hippocampus import YAMLStore

store = YAMLStore("data/", schema={
    "person": "persons",     # person entities in data/persons/
    "work": "plays",         # work entities in data/plays/
    "performance": "performances",
})

# CRUD
data = store.load("person", "42")        # -> dict or None
store.save("person", "42", data)         # atomic write with advisory lock
store.delete("person", "99")             # -> True if existed
ids = store.all_ids("person")            # -> {"42", "43", ...}

# Iteration
for pid, pdata in store.iter_type("person"):
    pass  # iterate all persons

# Backup
store.backup("person", "42")             # timestamped backup
# -> data/.backups/20260322_143000/persons/42.yaml
```

### Concurrency safety

- **Atomic writes:** Writes to a `.yaml.tmp` file first, then atomically renames. No partial writes on crash.
- **Advisory file locking:** Uses `fcntl.flock()` with a `.lock` file per entity. Raises `RuntimeError` if another process holds the lock.
- **Graceful iteration:** `iter_type()` skips malformed YAML files instead of crashing.

### YAML formatting

- Multi-line strings use literal block style (`|`) for readability
- Unicode fully supported (`allow_unicode=True`)
- Keys preserve insertion order (`sort_keys=False`)
- Block-style output for human readability

---

## Integration with other limbic packages

### amygdala → hippocampus

Use amygdala's clustering and embedding to find duplicate candidates, then hippocampus's veto gates to filter false positives and proposals to manage the merges:

```python
from limbic.amygdala import EmbeddingModel, greedy_centroid_cluster
from limbic.hippocampus import VetoMatcher, ProposalStore, exact_field

# Find candidate duplicates with embeddings
model = EmbeddingModel()
embeddings = model.embed_batch([p["name"] for p in persons])
clusters = greedy_centroid_cluster(embeddings, threshold=0.85)

# Filter with veto gates
matcher = VetoMatcher(gates=[exact_field("birth_year")])
# ... create proposals for accepted pairs ...
```

### cerebellum → hippocampus

Use cerebellum's audit findings to automatically create proposals for human review:

```python
# LLM verification flags issues → create proposals
for result in audit_results:
    if result.status == "flagged":
        store.create_modify(
            f"work/{result.item_id}",
            field_changes=result.metadata["suggested_fixes"],
            title=f"Fix flagged issues in {result.item_id}",
            reasoning="; ".join(result.findings),
        )
```

---

## What's NOT in hippocampus

- **Database migrations.** Hippocampus manages data-level changes (entities), not schema-level changes. Use Alembic or similar for schema migrations.
- **Conflict resolution.** If two pending proposals touch the same entity, hippocampus doesn't detect or resolve the conflict. This is a known limitation (see IDEAS.md).
- **Multi-file transactions.** Each proposal applies independently. Atomic application of multiple proposals as a group is not yet supported.
- **Version control.** The backup mechanism creates timestamped snapshots, but there's no diff/revert/branch model. For full versioning, use git on the YAML directory.
