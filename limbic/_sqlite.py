"""SQLite connection helper with no dependency beyond the standard library.

Kept outside the sub-packages so that stdlib-only modules (`hippocampus.resolve`)
can use it without importing `limbic.amygdala`, which loads numpy.
"""

import sqlite3
from pathlib import Path


def connect(db_path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection with best-practice PRAGMAs.

    Applies: WAL journal mode, 30s busy timeout, NORMAL synchronous (with WAL),
    64MB page cache, foreign key enforcement.

    Args:
        db_path: Path to SQLite database file, or ":memory:" for in-memory.
        readonly: If True, open with uri=True and ?mode=ro for read-only access.

    Returns:
        Configured sqlite3.Connection with row_factory=sqlite3.Row.
    """
    path_str = str(db_path)
    if readonly and path_str != ":memory:":
        uri = f"file:{path_str}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
    else:
        conn = sqlite3.connect(path_str, timeout=30)
    conn.row_factory = sqlite3.Row
    if path_str != ":memory:" and not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn
