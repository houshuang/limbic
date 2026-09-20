"""Test-suite guards: no test may touch a production database.

On 20 September 2026 two agents polluted the real cost ledger
(`~/.local/share/limbic/llm_costs.db`) by running the test suite: `cost_log`
is a module-level singleton whose path is resolved at *import* time, so a
fixture that sets `COST_LOG_DB` later is already too late. Hence two layers
here, in this order:

1. Environment variables are set at **module import time**, before pytest
   imports any test module (and therefore before `limbic.cerebellum.cost_log`
   constructs its singleton).
2. `sqlite3.connect` is wrapped so that opening anything under
   `~/.local/share/limbic` raises. Describing the rule in a docstring is what
   we had before; this refuses it. A test that legitimately needs the real
   path must say so by un-patching deliberately.

The autouse fixture then gives every test function its own ledger and cache
file, so tests cannot see each other's rows either.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

# --- 1. Redirect before anything imports the singletons. --------------------

PRODUCTION_DIR = (Path.home() / ".local" / "share" / "limbic").resolve()

_SESSION_TMP = Path(tempfile.mkdtemp(prefix="limbic-tests-"))
os.environ["COST_LOG_DB"] = str(_SESSION_TMP / "llm_costs.db")
os.environ["LIMBIC_CALL_CACHE_DB"] = str(_SESSION_TMP / "llm_cache.db")


# --- 2. Refuse a connection to the production directory. --------------------

_real_connect = sqlite3.connect


def _database_path(database: object) -> Path | None:
    """The filesystem path a `sqlite3.connect` argument would open, if any."""
    if not isinstance(database, (str, os.PathLike)):
        return None
    text = str(database)
    if text == ":memory:" or not text:
        return None
    if text.startswith("file:"):
        text = text[len("file:"):].split("?", 1)[0]
        if text.startswith("/") is False and text.startswith(":memory:"):
            return None
    try:
        return Path(text).expanduser().resolve()
    except OSError:  # pragma: no cover - resolve() on an exotic path
        return None


def _guarded_connect(database, *args, **kwargs):
    path = _database_path(database)
    if path is not None and (path == PRODUCTION_DIR or PRODUCTION_DIR in path.parents):
        raise AssertionError(
            f"test opened a production limbic database: {path}\n"
            "Tests must never write to ~/.local/share/limbic (two agents "
            "polluted the real cost ledger this way on 2026-09-20). Pass an "
            "explicit tmp_path, or use the COST_LOG_DB / LIMBIC_CALL_CACHE_DB "
            "environment variables that tests/conftest.py already sets."
        )
    return _real_connect(database, *args, **kwargs)


sqlite3.connect = _guarded_connect  # type: ignore[assignment]


# --- 3. One fresh ledger and cache per test. --------------------------------

@pytest.fixture(autouse=True)
def isolated_limbic_databases(tmp_path, monkeypatch):
    """Point the cost ledger and response cache at this test's tmp_path.

    Re-pointing the already-constructed `cost_log` singleton matters: it
    resolved its path at import time, so setting the env var alone would leave
    it on the session temp file shared by every test.
    """
    ledger = tmp_path / "llm_costs.db"
    cache = tmp_path / "llm_cache.db"
    monkeypatch.setenv("COST_LOG_DB", str(ledger))
    monkeypatch.setenv("LIMBIC_CALL_CACHE_DB", str(cache))

    from limbic.cerebellum.cost_log import cost_log

    cost_log.close()
    monkeypatch.setattr(cost_log, "_db_path", ledger, raising=False)
    monkeypatch.setattr(cost_log, "_conn", None, raising=False)
    yield
    cost_log.close()
