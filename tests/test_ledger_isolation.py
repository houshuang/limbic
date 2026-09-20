"""The guard in conftest.py is itself load-bearing, so it is tested.

On 20 September 2026 two agents polluted the real cost ledger by running the
test suite. A comment saying "don't do that" is what was in place at the time.
"""

from __future__ import annotations

import sqlite3

import pytest

from limbic.cerebellum.cost_log import cost_log

from .conftest import PRODUCTION_DIR


def test_cost_log_singleton_points_at_a_temp_file(tmp_path):
    assert cost_log.db_path == tmp_path / "llm_costs.db"
    assert PRODUCTION_DIR not in cost_log.db_path.parents


def test_opening_the_production_ledger_fails_the_test():
    with pytest.raises(AssertionError, match="production limbic database"):
        sqlite3.connect(str(PRODUCTION_DIR / "llm_costs.db"))


def test_the_guard_also_covers_uri_connections():
    with pytest.raises(AssertionError, match="production limbic database"):
        sqlite3.connect(f"file:{PRODUCTION_DIR / 'llm_cache.db'}?mode=ro", uri=True)


def test_an_ordinary_temp_database_still_opens(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "fine.db"))
    conn.close()


def test_in_memory_still_opens():
    sqlite3.connect(":memory:").close()


def test_each_test_gets_a_fresh_ledger():
    """If this row leaked from another test's ledger, isolation is broken."""
    assert cost_log.query() == []
    cost_log.log(project="isolation", model="haiku", purpose="probe", cost_usd=0.0)
    assert len(cost_log.query()) == 1
