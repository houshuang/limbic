#!/usr/bin/env python3
"""Launch the knowledge probe web UI for a given graph.

Usage:
    python3 experiments/probe_ui.py loro-mirror
    python3 experiments/probe_ui.py hamarquizen
    python3 experiments/probe_ui.py path/to/graph.json
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GRAPHS_DIR = Path(__file__).parent / "graphs"
TEMPLATE = Path("/tmp/knowledge-probe.html")

ALIASES = {
    "loro-mirror": "loro_mirror.json",
    "loro_mirror": "loro_mirror.json",
    "hamarquizen": "hamarquizen.json",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: probe_ui.py <graph-name-or-path>")
        print(f"Available: {', '.join(ALIASES.keys())}")
        for f in GRAPHS_DIR.glob("*.json"):
            print(f"  {f.stem}")
        sys.exit(1)

    arg = sys.argv[1]
    # Resolve graph path
    if arg in ALIASES:
        graph_path = GRAPHS_DIR / ALIASES[arg]
    elif Path(arg).exists():
        graph_path = Path(arg)
    elif (GRAPHS_DIR / arg).exists():
        graph_path = GRAPHS_DIR / arg
    elif (GRAPHS_DIR / f"{arg}.json").exists():
        graph_path = GRAPHS_DIR / f"{arg}.json"
    else:
        print(f"Graph not found: {arg}")
        sys.exit(1)

    with open(graph_path) as f:
        graph_data = json.load(f)

    # Read HTML template
    with open(TEMPLATE) as f:
        html = f.read()

    # Replace the fetch-based loader with embedded data
    loader = f"""
const GRAPH_PATH = 'embedded';
init({json.dumps(graph_data)});
"""
    html = html.replace(
        "const GRAPH_PATH = '__GRAPH_PATH__';\n"
        "if (GRAPH_PATH.startsWith('__')) {\n"
        "  // Embedded graph data (injected by launcher)\n"
        "  document.getElementById('subtitle').textContent = 'No graph loaded';\n"
        "} else {\n"
        "  fetch(GRAPH_PATH).then(r => r.json()).then(init).catch(e => {\n"
        "    document.getElementById('subtitle').textContent = 'Error loading graph: ' + e.message;\n"
        "  });\n"
        "}",
        loader,
    )

    # Write to temp file and open
    out = Path(tempfile.mktemp(suffix=".html", prefix="probe_"))
    out.write_text(html)
    print(f"Opening {out}")
    subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()
