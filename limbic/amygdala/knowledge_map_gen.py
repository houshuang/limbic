"""LLM-powered knowledge graph generation for knowledge_map.

Generates prerequisite DAGs from domain descriptions, document outlines,
or structured content. Uses amygdala.llm for LLM calls.

Usage:
    from amygdala.knowledge_map_gen import graph_from_description, graph_from_outline
    graph = await graph_from_description("Conflict-free replicated data types")
    graph = await graph_from_outline(sections, domain="Loro Mirror")
"""

from __future__ import annotations

import json
import logging
from .knowledge_map import KnowledgeGraph

log = logging.getLogger(__name__)

GRAPH_FROM_DESCRIPTION_PROMPT = """Generate a knowledge graph for adaptive assessment of someone's understanding of:

{domain}

{context}

Output a JSON object with a "nodes" array. Each node has:
- "id": short snake_case identifier
- "title": human-readable name
- "description": 1-2 sentence description of what knowing this means
- "level": 1 (broad area) to 4 (specific detail). Aim for mostly level 2-3.
- "obscurity": 1 (common knowledge) to 5 (specialist). Helps set priors.
- "prerequisites": array of node IDs that should be understood first (can be empty)

RULES:
- Generate 15-50 nodes depending on domain complexity
- Form a DAG — no circular prerequisites
- Balance breadth: don't over-represent any sub-area
- Prerequisites should reflect genuine learning dependencies, not just topic grouping
- Include a mix of foundational and advanced concepts
- Obscurity reflects how likely a general audience would know this, not how hard it is
- CRITICAL: Each node must test exactly ONE concept. If the description mixes a general concept with a domain-specific application, split them into separate nodes with a prerequisite relationship"""

GRAPH_FROM_OUTLINE_PROMPT = """Convert this document outline into a knowledge graph for adaptive assessment.

Domain: {domain}

OUTLINE:
{outline}

{context}

Output a JSON object with a "nodes" array. Each node has:
- "id": short snake_case identifier
- "title": human-readable name (can differ from section title)
- "description": what understanding this concept means
- "level": 1-4 depth level
- "obscurity": 1 (common) to 5 (specialist)
- "prerequisites": array of node IDs (genuine learning dependencies)

RULES:
- Not every section needs to be a node — merge trivial sections, split complex ones
- Add prerequisite concepts that aren't in the outline but are needed (e.g., "React basics" if the doc assumes React knowledge)
- Prerequisites should reflect understanding dependencies, not just reading order
- 15-50 nodes total
- Each node's description should test ONE concept — never mix a general prerequisite with a domain-specific application in the same node"""

GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "level": {"type": "integer"},
                    "obscurity": {"type": "integer"},
                    "prerequisites": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "title", "description", "level", "obscurity", "prerequisites"],
            },
        }
    },
    "required": ["nodes"],
}


async def graph_from_description(
    domain: str,
    context: str = "",
    model: str = "gemini3-flash",
) -> KnowledgeGraph:
    """Generate a knowledge graph from a domain description."""
    from .llm import generate_structured
    prompt = GRAPH_FROM_DESCRIPTION_PROMPT.format(
        domain=domain,
        context=f"ADDITIONAL CONTEXT:\n{context}" if context else "",
    )
    result, meta = await generate_structured(prompt, schema=GRAPH_SCHEMA, model=model)
    nodes = _validate_nodes(result.get("nodes", []))
    log.info(f"Generated {len(nodes)} nodes for '{domain}' ({meta['model']}, ${meta['total_cost_usd']:.4f})")
    return KnowledgeGraph(nodes=nodes)


async def graph_from_outline(
    sections: list[dict],
    domain: str = "",
    context: str = "",
    model: str = "gemini3-flash",
) -> KnowledgeGraph:
    """Generate a knowledge graph from a document outline.

    sections: list of {title, level?, description?} representing headings/sections.
    """
    from .llm import generate_structured
    outline = "\n".join(
        f"{'  ' * (s.get('level', 1) - 1)}- {s['title']}"
        + (f": {s['description']}" if s.get("description") else "")
        for s in sections
    )
    prompt = GRAPH_FROM_OUTLINE_PROMPT.format(
        domain=domain or "this document",
        outline=outline,
        context=f"ADDITIONAL CONTEXT:\n{context}" if context else "",
    )
    result, meta = await generate_structured(prompt, schema=GRAPH_SCHEMA, model=model)
    nodes = _validate_nodes(result.get("nodes", []))
    log.info(f"Generated {len(nodes)} nodes from outline ({meta['model']}, ${meta['total_cost_usd']:.4f})")
    return KnowledgeGraph(nodes=nodes)


def graph_from_dict(data: dict) -> KnowledgeGraph:
    """Load a knowledge graph from a dict (e.g., parsed JSON file)."""
    return KnowledgeGraph(nodes=_validate_nodes(data.get("nodes", data)))


def _validate_nodes(nodes: list[dict]) -> list[dict]:
    """Ensure nodes have required fields and valid prerequisites."""
    ids = {n["id"] for n in nodes}
    validated = []
    for n in nodes:
        if "id" not in n or "title" not in n:
            continue
        n.setdefault("description", "")
        n.setdefault("level", 2)
        n.setdefault("obscurity", 3)
        n["prerequisites"] = [p for p in n.get("prerequisites", []) if p in ids and p != n["id"]]
        validated.append(n)
    return validated


def check_graph_quality(graph) -> list[dict]:
    """Check a knowledge graph for common quality issues.

    Returns a list of {node_id, issue, suggestion} dicts.
    """
    issues = []
    for node in graph.nodes:
        desc = node.get("description", "")
        # Flag composite descriptions (multiple sentences asking about different concepts)
        sentences = [s.strip() for s in desc.replace(". ", ".\n").split("\n") if s.strip()]
        if len(sentences) >= 3:
            issues.append({
                "node_id": node["id"], "issue": "composite_description",
                "suggestion": f"Description has {len(sentences)} sentences — consider splitting into separate nodes.",
            })
        # Flag nodes with no prerequisites and high obscurity (likely missing prereqs)
        if not node.get("prerequisites") and node.get("obscurity", 3) >= 4:
            issues.append({
                "node_id": node["id"], "issue": "obscure_without_prerequisites",
                "suggestion": "Obscure concept with no prerequisites — likely needs a foundational prereq.",
            })
        # Flag empty descriptions
        if len(desc) < 10:
            issues.append({
                "node_id": node["id"], "issue": "empty_description",
                "suggestion": "Description too short — needs enough detail for a meaningful probe question.",
            })
    return issues
