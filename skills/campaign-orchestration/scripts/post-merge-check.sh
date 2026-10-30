#!/bin/bash
# Run after every data merge into main, before the next merge. Refuses (exit 1) on any failure.
set -uo pipefail
cd /Users/stian/src/skard
fail=0
timeout 2400 python3 tests/run_parallel.py -j 2 --modules test_evidence_spine test_acquisition_integrity test_axis_guards test_holds3 > /tmp/pmc-tests.log 2>&1 || { echo "FAIL data tests (see /tmp/pmc-tests.log)"; fail=1; }
python3 scripts/preflight.py > /tmp/pmc-preflight.log 2>&1 || { echo "FAIL preflight"; fail=1; }
[ $fail = 0 ] && echo "post-merge check PASS" || { echo "post-merge check FAILED: do not merge further"; exit 1; }
