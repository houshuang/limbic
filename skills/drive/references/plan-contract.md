# Direction card contract

The machine-readable plan uses schema `limbic-drive-plan-v0` and these fields:

```json
{
  "schema": "limbic-drive-plan-v0",
  "mode": "research | improve",
  "dry_run": true,
  "request_summary": "...",
  "desired_outcome": "...",
  "definition_of_better": ["..."],
  "assumptions": ["..."],
  "non_goals": ["..."],
  "open_questions": [],
  "precedent_search": {
    "queries": ["one to three searches"],
    "inspected": ["zero to three paths, task ids, or URLs"],
    "result": "found | none-found"
  },
  "precedents": [{"source": "...", "lesson": "..."}],
  "pilot": {
    "description": "...",
    "unit_count": 1,
    "artifact": "...",
    "validation_kind": "human-use | manual-test | source-check",
    "evidence": ["..."],
    "user_checkpoint": "..."
  },
  "budget": {
    "max_workers": 0,
    "max_additional_model_calls": 0,
    "max_additional_premium_calls": 0
  },
  "delegation": {"enabled": false, "workers_may_delegate": false},
  "scale": {"allowed": false, "gate": "..."},
  "stop_conditions": ["..."],
  "next_action": "..."
}
```

The visible card may use natural language rather than JSON. Validate the JSON
form first when possible, then translate it without weakening the constraints.
