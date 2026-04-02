"""Centralized LLM cost logger — tracks spend across projects, models, and hosts.

Logs every LLM call to a single SQLite database with cost computed via litellm's
pricing data (2,500+ models).  Works with any SDK: litellm, google-genai, anthropic,
openai, or raw REST.

Usage — standalone (any SDK):

    from limbic.cerebellum.cost_log import cost_log
    cost_log.log(project="petrarca", model="gemini/gemini-2.5-flash",
                 prompt_tokens=1200, completion_tokens=340)

Usage — litellm callback (auto-captures every litellm.completion call):

    import litellm
    from limbic.cerebellum.cost_log import cost_log
    litellm.callbacks = [cost_log.callback("alif")]

DB location (in order of precedence):
    1. COST_LOG_DB environment variable
    2. ~/.local/share/limbic/llm_costs.db

Sync:
    python -m limbic.cerebellum.cost_log sync --host alif
    python -m limbic.cerebellum.cost_log report --days 7
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_costs (
    id          TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    project     TEXT NOT NULL,
    host        TEXT NOT NULL,
    model       TEXT NOT NULL,
    api_key_hint TEXT DEFAULT '',
    prompt_tokens    INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cached_tokens    INTEGER DEFAULT 0,
    cost_usd    REAL DEFAULT 0.0,
    script      TEXT DEFAULT '',
    purpose     TEXT DEFAULT '',
    metadata    TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_costs_ts ON llm_costs(ts);
CREATE INDEX IF NOT EXISTS idx_costs_project ON llm_costs(project);
CREATE INDEX IF NOT EXISTS idx_costs_model ON llm_costs(model);
CREATE INDEX IF NOT EXISTS idx_costs_host ON llm_costs(host);
"""

# ---------------------------------------------------------------------------
# Default DB path
# ---------------------------------------------------------------------------

def _default_db_path() -> Path:
    env = os.environ.get("COST_LOG_DB")
    if env:
        return Path(env)
    return Path.home() / ".local" / "share" / "limbic" / "llm_costs.db"


def _detect_host() -> str:
    return os.environ.get("COST_LOG_HOST", platform.node().split(".")[0])


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

# Fallback pricing (USD per 1M tokens) when litellm is not installed.
# Covers models actively used across projects.  Updated 2026-04.
_FALLBACK_PRICES: dict[str, tuple[float, float]] = {  # (input/M, output/M)
    "gemini-2.0-flash":           (0.10, 0.40),
    "gemini-2.0-flash-lite":      (0.075, 0.30),
    "gemini-2.5-flash":           (0.30, 2.50),
    "gemini-2.5-flash-lite":      (0.15, 0.60),
    "gemini-2.5-pro":             (1.25, 10.00),
    "gemini-3-flash-preview":     (0.30, 2.50),
    "gemini-3.1-flash-lite-preview": (0.15, 0.60),
    "claude-sonnet-4-20250514":   (3.00, 15.00),
    "claude-haiku-4-5-20241022":  (0.80, 4.00),
}


def _fallback_cost(model: str, prompt_tokens: int,
                   completion_tokens: int) -> float | None:
    """Compute cost from built-in price table. Strips gemini/ prefix."""
    key = model.removeprefix("gemini/")
    prices = _FALLBACK_PRICES.get(key)
    if not prices:
        return None
    inp, out = prices
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int,
                 cached_tokens: int = 0) -> float | None:
    """Compute USD cost using litellm's pricing database (2,500+ models).

    Falls back to a built-in price table for common models when litellm
    is not installed.  Returns None if the model is unknown.
    """
    try:
        import litellm
        prompt_cost, compl_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(prompt_cost + compl_cost)
    except ImportError:
        return _fallback_cost(model, prompt_tokens, completion_tokens)
    except Exception:
        return _fallback_cost(model, prompt_tokens, completion_tokens)


# ---------------------------------------------------------------------------
# Core logger
# ---------------------------------------------------------------------------

@dataclass
class CostRecord:
    """A single LLM cost record."""
    id: str
    ts: str
    project: str
    host: str
    model: str
    api_key_hint: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float
    script: str
    purpose: str
    metadata: dict


class CostLog:
    """Central LLM cost logger backed by SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else _default_db_path()
        self._conn: sqlite3.Connection | None = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        if str(self._db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn
        return conn

    def log(self, *, project: str, model: str,
            prompt_tokens: int = 0, completion_tokens: int = 0,
            cached_tokens: int = 0, cost_usd: float | None = None,
            api_key_hint: str = "", host: str | None = None,
            script: str = "", purpose: str = "",
            metadata: dict[str, Any] | None = None) -> CostRecord:
        """Log an LLM call.  If cost_usd is None, computes it via litellm."""

        if cost_usd is None:
            cost_usd = compute_cost(model, prompt_tokens, completion_tokens,
                                    cached_tokens) or 0.0

        record = CostRecord(
            id=uuid.uuid4().hex,
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            project=project,
            host=host or _detect_host(),
            model=model,
            api_key_hint=api_key_hint,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            script=script,
            purpose=purpose,
            metadata=metadata or {},
        )

        conn = self._connect()
        conn.execute(
            """INSERT INTO llm_costs
               (id, ts, project, host, model, api_key_hint,
                prompt_tokens, completion_tokens, cached_tokens,
                cost_usd, script, purpose, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.id, record.ts, record.project, record.host,
             record.model, record.api_key_hint,
             record.prompt_tokens, record.completion_tokens, record.cached_tokens,
             record.cost_usd, record.script, record.purpose,
             json.dumps(record.metadata)),
        )
        conn.commit()
        return record

    # -----------------------------------------------------------------------
    # litellm callback
    # -----------------------------------------------------------------------

    def callback(self, project: str, host: str | None = None):
        """Return a litellm CustomLogger that auto-logs every completion.

        Usage:
            import litellm
            litellm.callbacks = [cost_log.callback("alif")]
        """
        import litellm as _litellm

        outer = self

        class _Callback(_litellm.CustomLogger):
            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                try:
                    slp = kwargs.get("standard_logging_object") or {}
                    model = slp.get("model") or kwargs.get("model", "unknown")
                    cost = slp.get("response_cost") or kwargs.get("response_cost", 0)

                    usage = getattr(response_obj, "usage", None)
                    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
                    ct = getattr(usage, "completion_tokens", 0) if usage else 0

                    api_key = kwargs.get("litellm_params", {}).get("api_key", "")
                    hint = api_key[-4:] if isinstance(api_key, str) and len(api_key) > 4 else ""

                    outer.log(
                        project=project,
                        model=model,
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        cost_usd=float(cost) if cost else None,
                        api_key_hint=hint,
                        host=host,
                        metadata={"litellm_callback": True},
                    )
                except Exception as exc:
                    log.warning("cost_log callback failed: %s", exc)

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                self.log_success_event(kwargs, response_obj, start_time, end_time)

        return _Callback()

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def query(self, *, days: int | None = None, since: str | None = None,
              project: str | None = None, host: str | None = None,
              model: str | None = None) -> list[sqlite3.Row]:
        """Query cost records with optional filters."""
        conn = self._connect()
        clauses, params = [], []
        if days:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            clauses.append("ts >= ?")
            params.append(cutoff)
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        if project:
            clauses.append("project = ?")
            params.append(project)
        if host:
            clauses.append("host = ?")
            params.append(host)
        if model:
            clauses.append("model LIKE ?")
            params.append(f"%{model}%")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return conn.execute(
            f"SELECT * FROM llm_costs{where} ORDER BY ts DESC", params
        ).fetchall()

    def summary(self, *, days: int | None = None, since: str | None = None,
                group_by: str = "project") -> list[dict]:
        """Aggregate costs grouped by project, model, host, or api_key_hint."""
        valid = {"project", "model", "host", "api_key_hint", "script"}
        if group_by not in valid:
            raise ValueError(f"group_by must be one of {valid}")

        conn = self._connect()
        clauses, params = [], []
        if days:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            clauses.append("ts >= ?")
            params.append(cutoff)
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = conn.execute(f"""
            SELECT {group_by} AS grp,
                   COUNT(*) AS calls,
                   SUM(prompt_tokens) AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens,
                   SUM(cached_tokens) AS cached_tokens,
                   SUM(cost_usd) AS cost_usd,
                   MIN(ts) AS first_call,
                   MAX(ts) AS last_call
            FROM llm_costs{where}
            GROUP BY {group_by}
            ORDER BY cost_usd DESC
        """, params).fetchall()
        return [dict(r) for r in rows]

    def total(self, *, days: int | None = None) -> float:
        """Total USD spend."""
        conn = self._connect()
        if days:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs WHERE ts >= ?",
                (cutoff,)).fetchone()
        else:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs").fetchone()
        return float(row[0])

    # -----------------------------------------------------------------------
    # Merge (for sync from remote)
    # -----------------------------------------------------------------------

    def merge_from(self, remote_db_path: str | Path) -> int:
        """Merge rows from a remote cost DB into this one.

        Uses INSERT OR IGNORE on the UUID primary key, so rows that already
        exist are skipped.  Returns number of new rows inserted.
        """
        conn = self._connect()
        remote = str(remote_db_path)
        conn.execute("ATTACH DATABASE ? AS remote", (remote,))
        try:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO llm_costs
                SELECT * FROM remote.llm_costs
            """)
            count = cursor.rowcount
            conn.commit()
        finally:
            conn.execute("DETACH DATABASE remote")
        log.info("Merged %d new rows from %s", count, remote)
        return count

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

cost_log = CostLog()


# ---------------------------------------------------------------------------
# Sync helper
# ---------------------------------------------------------------------------

def sync_from_remote(host: str = "alif",
                     remote_db: str = "/opt/limbic-data/llm_costs.db") -> int:
    """Rsync the remote cost DB and merge into local. Returns new row count."""
    import subprocess
    import tempfile
    tmp = tempfile.mktemp(suffix=".db")
    result = subprocess.run(
        ["rsync", "-az", f"{host}:{remote_db}", tmp],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.warning("rsync failed: %s", result.stderr.strip())
        return -1
    cl = CostLog()
    n = cl.merge_from(tmp)
    os.unlink(tmp)
    return n


# ---------------------------------------------------------------------------
# Dashboard server (stdlib only — no Flask/datasette needed)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Cost Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { --bg: #f7f4ec; --ink: #2a2420; --muted: #6a6458; --accent: #8b2500;
          --rule: #e4dfd4; --card: #fff; --green: #2a7a4a; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'DM Sans', -apple-system, sans-serif; background: var(--bg);
         color: var(--ink); max-width: 1100px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .sync-btn { background: var(--accent); color: #fff; border: none; padding: 6px 16px;
              border-radius: 4px; cursor: pointer; font-size: 13px; float: right; margin-top: -38px; }
  .sync-btn:hover { opacity: 0.85; }
  .totals { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .total-card { background: var(--card); border: 1px solid var(--rule); border-radius: 8px;
                padding: 16px 20px; flex: 1; min-width: 140px; }
  .total-card .label { font-size: 11px; color: var(--muted); text-transform: uppercase;
                       letter-spacing: 0.05em; }
  .total-card .value { font-size: 28px; font-weight: 600; margin-top: 4px; }
  .total-card .value.cost { color: var(--accent); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
  .panel { background: var(--card); border: 1px solid var(--rule); border-radius: 8px; padding: 20px; }
  .panel h2 { font-size: 14px; font-weight: 600; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--muted); font-weight: 500; padding: 6px 8px;
       border-bottom: 1px solid var(--rule); font-size: 11px; text-transform: uppercase; }
  td { padding: 6px 8px; border-bottom: 1px solid var(--rule); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
  .controls select, .controls input { padding: 6px 10px; border: 1px solid var(--rule);
    border-radius: 4px; font-size: 13px; background: var(--card); }
  canvas { max-height: 260px; }
  .chart-panel { grid-column: 1 / -1; }
  .last-sync { font-size: 11px; color: var(--muted); }
</style>
</head>
<body>
<h1>LLM Cost Dashboard</h1>
<div class="subtitle">
  <span id="db-path"></span> &middot; <span class="last-sync" id="last-sync"></span>
</div>
<button class="sync-btn" onclick="doSync()">Sync from Hetzner</button>

<div class="controls">
  <select id="days" onchange="reload()">
    <option value="1">Last 24h</option>
    <option value="7" selected>Last 7 days</option>
    <option value="30">Last 30 days</option>
    <option value="90">Last 90 days</option>
    <option value="0">All time</option>
  </select>
</div>

<div class="totals" id="totals"></div>

<div class="grid">
  <div class="panel chart-panel">
    <h2>Daily Cost</h2>
    <canvas id="daily-chart"></canvas>
  </div>
  <div class="panel">
    <h2>By Project</h2>
    <table id="by-project"></table>
  </div>
  <div class="panel">
    <h2>By Model</h2>
    <table id="by-model"></table>
  </div>
  <div class="panel">
    <h2>By Host</h2>
    <table id="by-host"></table>
  </div>
  <div class="panel">
    <h2>Recent Calls</h2>
    <table id="recent"></table>
  </div>
</div>

<script>
let dailyChart = null;
const $ = id => document.getElementById(id);

async function api(path) {
  const r = await fetch('/api/' + path);
  return r.json();
}

function fmt(n) { return '$' + n.toFixed(4); }
function fmtK(n) { return n >= 1000 ? (n/1000).toFixed(1) + 'k' : n.toString(); }

function renderTable(id, rows, cols) {
  const el = $(id);
  let h = '<thead><tr>' + cols.map(c => `<th>${c[0]}</th>`).join('') + '</tr></thead><tbody>';
  for (const r of rows) {
    h += '<tr>' + cols.map(c => {
      const v = c[1](r);
      return `<td class="${c[2]||''}">${v}</td>`;
    }).join('') + '</tr>';
  }
  el.innerHTML = h + '</tbody>';
}

async function reload() {
  const days = $('days').value;
  const d = await api('summary?days=' + days);

  // Totals
  $('totals').innerHTML = `
    <div class="total-card"><div class="label">Total Cost</div><div class="value cost">${fmt(d.total_usd)}</div></div>
    <div class="total-card"><div class="label">API Calls</div><div class="value">${d.total_calls.toLocaleString()}</div></div>
    <div class="total-card"><div class="label">Prompt Tokens</div><div class="value">${fmtK(d.total_prompt)}</div></div>
    <div class="total-card"><div class="label">Completion Tokens</div><div class="value">${fmtK(d.total_completion)}</div></div>
  `;
  $('db-path').textContent = d.db_path;
  $('last-sync').textContent = 'Last sync: ' + (d.last_sync || 'never');

  const cols = [
    ['Name', r => r.grp, ''],
    ['Calls', r => r.calls.toLocaleString(), 'num'],
    ['Cost', r => fmt(r.cost_usd), 'num'],
  ];
  renderTable('by-project', d.by_project, cols);
  renderTable('by-model', d.by_model, cols);
  renderTable('by-host', d.by_host, cols);

  // Recent calls
  const recent = await api('recent?days=' + days);
  renderTable('recent', recent.slice(0, 20), [
    ['Time', r => r.ts.slice(5, 16).replace('T', ' '), ''],
    ['Project', r => r.project, ''],
    ['Model', r => r.model.replace('gemini/', ''), ''],
    ['Tokens', r => fmtK(r.prompt_tokens + r.completion_tokens), 'num'],
    ['Cost', r => fmt(r.cost_usd), 'num'],
  ]);

  // Daily chart
  const daily = await api('daily?days=' + days);
  if (dailyChart) dailyChart.destroy();
  const projects = [...new Set(daily.flatMap(d => Object.keys(d.projects)))];
  const colors = ['#8b2500','#2a7a4a','#4a6fa5','#d4a056','#7a4a8b','#5a8a6a','#a05040'];
  dailyChart = new Chart($('daily-chart'), {
    type: 'bar',
    data: {
      labels: daily.map(d => d.date.slice(5)),
      datasets: projects.map((p, i) => ({
        label: p,
        data: daily.map(d => d.projects[p] || 0),
        backgroundColor: colors[i % colors.length],
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { stacked: true }, y: { stacked: true, ticks: { callback: v => '$' + v.toFixed(2) } } },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } } },
    },
  });
}

async function doSync() {
  const btn = document.querySelector('.sync-btn');
  btn.textContent = 'Syncing...';
  btn.disabled = true;
  try {
    const r = await fetch('/api/sync', { method: 'POST' });
    const d = await r.json();
    btn.textContent = d.new_rows >= 0 ? `Synced (${d.new_rows} new)` : 'Sync failed';
    if (d.new_rows >= 0) reload();
  } catch { btn.textContent = 'Sync failed'; }
  setTimeout(() => { btn.textContent = 'Sync from Hetzner'; btn.disabled = false; }, 3000);
}

reload();
</script>
</body>
</html>
"""


def _serve_dashboard(port: int = 8042, open_browser: bool = True):
    """Serve the built-in cost dashboard on localhost."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    cl = CostLog()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass  # quiet

        def _json(self, data, status=200):
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            qs = dict(urllib.parse.parse_qsl(parsed.query))
            days = int(qs.get("days", 7)) or None

            if parsed.path == "/":
                body = _DASHBOARD_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)

            elif parsed.path == "/api/summary":
                total_rows = cl.query(days=days)
                self._json({
                    "total_usd": cl.total(days=days),
                    "total_calls": len(total_rows),
                    "total_prompt": sum(r["prompt_tokens"] for r in total_rows),
                    "total_completion": sum(r["completion_tokens"] for r in total_rows),
                    "by_project": cl.summary(days=days, group_by="project"),
                    "by_model": cl.summary(days=days, group_by="model"),
                    "by_host": cl.summary(days=days, group_by="host"),
                    "db_path": str(cl.db_path),
                    "last_sync": _last_sync_time(),
                })

            elif parsed.path == "/api/recent":
                rows = cl.query(days=days)[:50]
                self._json([dict(r) for r in rows])

            elif parsed.path == "/api/daily":
                conn = cl._connect()
                clauses, params = [], []
                if days:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ")
                    clauses.append("ts >= ?")
                    params.append(cutoff)
                where = " WHERE " + " AND ".join(clauses) if clauses else ""
                rows = conn.execute(f"""
                    SELECT substr(ts, 1, 10) AS date, project,
                           SUM(cost_usd) AS cost
                    FROM llm_costs{where}
                    GROUP BY date, project
                    ORDER BY date
                """, params).fetchall()
                daily = {}
                for r in rows:
                    d = r["date"]
                    if d not in daily:
                        daily[d] = {"date": d, "projects": {}}
                    daily[d]["projects"][r["project"]] = round(r["cost"], 6)
                self._json(list(daily.values()))

            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/sync":
                n = sync_from_remote()
                self._json({"new_rows": n, "total": cl.total()})
            else:
                self.send_error(404)

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  DB: {cl.db_path}")
    print(f"  Press Ctrl+C to stop\n")
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def _last_sync_time() -> str | None:
    """Check when the last sync from hetzner happened (most recent hetzner row)."""
    cl = CostLog()
    conn = cl._connect()
    row = conn.execute(
        "SELECT MAX(ts) FROM llm_costs WHERE host = 'hetzner'"
    ).fetchone()
    return row[0] if row and row[0] else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m limbic.cerebellum.cost_log",
        description="LLM cost tracking — report, sync, and dashboard",
    )
    sub = parser.add_subparsers(dest="cmd")

    # -- report --
    rpt = sub.add_parser("report", help="Show cost summary")
    rpt.add_argument("--days", type=int, default=7, help="Look-back window (default 7)")
    rpt.add_argument("--group-by", default="project",
                     choices=["project", "model", "host", "api_key_hint", "script"])
    rpt.add_argument("--json", action="store_true", help="JSON output")

    # -- sync --
    syn = sub.add_parser("sync", help="Sync costs from remote host")
    syn.add_argument("--host", default="alif", help="SSH host alias (default: alif)")
    syn.add_argument("--remote-db", default="/opt/limbic-data/llm_costs.db",
                     help="Remote DB path")

    # -- dashboard --
    dash = sub.add_parser("dashboard", help="Launch web dashboard")
    dash.add_argument("--port", type=int, default=8042)
    dash.add_argument("--no-open", action="store_true", help="Don't open browser")

    # -- datasette --
    ds = sub.add_parser("datasette", help="Launch datasette web UI")
    ds.add_argument("--port", type=int, default=8043)

    args = parser.parse_args()

    if args.cmd == "report":
        cl = CostLog()
        if args.json:
            rows = cl.summary(days=args.days, group_by=args.group_by)
            print(json.dumps({"days": args.days, "group_by": args.group_by,
                              "total_usd": cl.total(days=args.days),
                              "rows": rows}, indent=2, default=str))
        else:
            total = cl.total(days=args.days)
            print(f"\n  LLM costs — last {args.days} days  (total: ${total:.4f})")
            print(f"  DB: {cl.db_path}\n")
            rows = cl.summary(days=args.days, group_by=args.group_by)
            if not rows:
                print("  (no data)")
                return
            print(f"  {'Group':<30} {'Calls':>7} {'Prompt':>10} {'Compl':>10} {'Cost':>10}")
            print(f"  {'-'*30} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
            for r in rows:
                print(f"  {r['grp']:<30} {r['calls']:>7} {r['prompt_tokens']:>10,}"
                      f" {r['completion_tokens']:>10,} ${r['cost_usd']:>9.4f}")
            print()

    elif args.cmd == "sync":
        print(f"Syncing from {args.host}:{args.remote_db} ...")
        n = sync_from_remote(args.host, args.remote_db)
        if n < 0:
            print("Sync failed.", file=sys.stderr)
            sys.exit(1)
        print(f"Merged {n} new rows.  Total: ${CostLog().total():.4f}")

    elif args.cmd == "dashboard":
        _serve_dashboard(port=args.port, open_browser=not args.no_open)

    elif args.cmd == "datasette":
        db = str(CostLog().db_path)
        print(f"Launching datasette on port {args.port} for {db}")
        subprocess.run(["datasette", db, "--port", str(args.port)])

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
