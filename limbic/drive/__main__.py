"""Command-line policy checks for Drive plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .policy import check_calibrations, validate_plan


def _load_json(path: str) -> object:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate(path: str) -> int:
    try:
        plan = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    errors = validate_plan(plan)
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        return 1
    print("VALID: bounded dry-run plan")
    return 0


def _calibrate() -> int:
    results = check_calibrations()
    for result in results:
        label = "PASS" if result.passed else "FAIL"
        print(f"{label} {result.case_id}")
        for error in result.errors:
            print(f"  - {error}")
    passed = sum(result.passed for result in results)
    print(f"{passed}/{len(results)} calibration cases passed")
    return 0 if passed == len(results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a $drive direction card without invoking a model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a plan JSON file or stdin")
    validate.add_argument("path", help="JSON path, or - for stdin")
    subparsers.add_parser("calibrate", help="run the historical policy cases")
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.path)
    return _calibrate()


if __name__ == "__main__":
    raise SystemExit(main())
