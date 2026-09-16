# limbic.drive

**Choose the first move before the swarm.**

`skills/drive` is a shared Codex/Claude planning skill for open-ended voice dumps
such as “research this” and “improve this.” It retrieves the nearest local
precedents, proposes one representative pilot, and stops for human judgment. The
v0 policy deliberately permits no worker calls or project mutation.

```bash
python -m limbic.drive calibrate
python -m limbic.drive validate /path/to/drive-plan.json
```

The three bundled calibration cases capture costly failure modes from the NRK
apps, the Otak/Hirsch investigation, and the Codex/Claude workflow research.
`calibrate` replays all of them; a policy change that would have re-allowed a
past mistake fails there rather than in a live session.

The same checks are available as a library, so a host that builds plans itself
can gate them without shelling out:

```python
from limbic.drive import validate_plan, check_calibrations, SCHEMA_VERSION

violations = validate_plan(plan)      # [] means the plan is allowed to run
if violations:
    raise ValueError(violations)

failures = [c for c in check_calibrations() if c.errors]
for f in failures:
    print(f.case_id, f.errors)     # CalibrationResult
```

`load_calibration_cases()` returns the bundled cases as plain dicts if you want
to extend the set or inspect what a case actually asserts. `SCHEMA_VERSION`
identifies the plan shape `validate_plan` expects, so a host that stores plans
can tell an old card from a current one.

`validate_plan` returns *every* violation rather than the first, so a plan gets
one round of correction instead of one per rule.

## Install

```bash
pip install git+https://github.com/houshuang/limbic.git
```

`limbic.drive` needs nothing beyond the standard library.

## What's NOT in drive

- **No planning itself.** The model decides what the user means, which precedent
  matters, and what a representative pilot looks like. This package only refuses
  plans that skip the pilot.
- **No execution.** It never spawns a worker, calls a model, or touches a
  project. It returns a list of violations.

---

Part of [limbic](../../README.md).
