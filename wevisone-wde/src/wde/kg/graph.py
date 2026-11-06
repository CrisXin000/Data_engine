"""Knowledge graph skeleton (§2.1).

Initialized from WordNet (NLTK). Supports:
  add_candidate_term → text link → (vision verify placeholder) → merge aliases.
Preserves graph_version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from wde.schema.models import GraphEdge, GraphNode


# Fallback seed if WordNet download fails — still dozens of common objects.
FALLBACK_CONCEPTS: list[dict[str, Any]] = [
    {"canonical": "dog", "synset": "dog.n.01", "aliases": ["canine", "puppy", "hound"]},
    {"canonical": "cat", "synset": "cat.n.01", "aliases": ["feline", "kitten"]},
    {"canonical": "car", "synset": "car.n.01", "aliases": ["automobile", "auto", "vehicle"]},
    {"canonical": "bicycle", "synset": "bicycle.n.01", "aliases": ["bike", "cycle"]},
    {"canonical": "person", "synset": "person.n.01", "aliases": ["human", "people", "man", "woman"]},
    {"canonical": "chair", "synset": "chair.n.01", "aliases": ["seat"]},
    {"canonical": "table", "synset": "table.n.02", "aliases": ["desk"]},
    {"canonical": "bottle", "synset": "bottle.n.01", "aliases": []},
    {"canonical": "cup", "synset": "cup.n.01", "aliases": ["mug"]},
    {"canonical": "book", "synset": "book.n.01", "aliases": []},
    {"canonical": "bird", "synset": "bird.n.01", "aliases": []},
    {"canonical": "horse", "synset": "horse.n.01", "aliases": []},
    {"canonical": "sheep", "synset": "sheep.n.01", "aliases": []},
    {"canonical": "cow", "synset": "cow.n.01", "aliases": ["cattle"]},
    {"canonical": "airplane", "synset": "airplane.n.01", "aliases": ["plane", "aircraft"]},
    {"canonical": "bus", "synset": "bus.n.01", "aliases": []},
    {"canonical": "train", "synset": "train.n.01", "aliases": []},
    {"canonical": "truck", "synset": "truck.n.01", "aliases": ["lorry"]},
    {"canonical": "boat", "synset": "boat.n.01", "aliases": ["ship"]},
    {"canonical": "traffic light", "synset": "traffic_light.n.01", "aliases": ["stoplight"]},
    {"canonical": "fire hydrant", "synset": "fire_hydrant.n.01", "aliases": ["hydrant"]},
    {"canonical": "stop sign", "synset": "stop_sign.n.01", "aliases": []},
    {"canonical": "bench", "synset": "bench.n.01", "aliases": []},
    {"canonical": "backpack", "synset": "backpack.n.01", "aliases": ["rucksack"]},
    {"canonical": "umbrella", "synset": "umbrella.n.01", "aliases": []},
    {"canonical": "handbag", "synset": "handbag.n.01", "aliases": ["purse"]},
    {"canonical": "suitcase", "synset": "suitcase.n.01", "aliases": ["luggage"]},
    {"canonical": "frisbee", "synset": "frisbee.n.01", "aliases": []},
    {"canonical": "skis", "synset": "skis.n.01", "aliases": ["ski"]},
    {"canonical": "snowboard", "synset": "snowboard.n.01", "aliases": []},
    {"canonical": "sports ball", "synset": "sports_ball.n.01", "aliases": ["ball"]},
    {"canonical": "kite", "synset": "kite.n.01", "aliases": []},
    {"canonical": "baseball bat", "synset": "baseball_bat.n.01", "aliases": []},
    {"canonical": "skateboard", "synset": "skateboard.n.01", "aliases": []},
    {"canonical": "surfboard", "synset": "surfboard.n.01", "aliases": []},
    {"canonical": "tennis racket", "synset": "tennis_racket.n.01", "aliases": ["racquet"]},
    {"canonical": "fork", "synset": "fork.n.01", "aliases": []},
    {"canonical": "knife", "synset": "knife.n.01", "aliases": []},
    {"canonical": "spoon", "synset": "spoon.n.01", "aliases": []},
    {"canonical": "bowl", "synset": "bowl.n.01", "aliases": []},
    {"canonical": "banana", "synset": "banana.n.01", "aliases": []},
    {"canonical": "apple", "synset": "apple.n.01", "aliases": []},
    {"canonical": "sandwich", "synset": "sandwich.n.01", "aliases": []},
    {"canonical": "orange", "synset": "orange.n.01", "aliases": []},
    {"canonical": "broccoli", "synset": "broccoli.n.01", "aliases": []},
    {"canonical": "carrot", "synset": "carrot.n.01", "aliases": []},
    {"canonical": "red rectangle", "synset": None, "aliases": ["red box", "red block"]},
    {"canonical": "blue circle", "synset": None, "aliases": ["blue disk", "blue ball"]},
    {"canonical": "green triangle", "synset": None, "aliases": []},
    {"canonical": "yellow square", "synset": None, "aliases": ["yellow box"]},
]


def _try_wordnet_concept(synset_name: str) -> Optional[dict[str, Any]]:
    try:
        from nltk.corpus import wordnet as wn

        syn = wn.synset(synset_name)
        lemmas = [l.name().replace("_", " ") for l in syn.lemmas()]
        canonical = lemmas[0] if lemmas else synset_name.split(".")[0].replace("_", " ")
        aliases = [x for x in lemmas[1:] if x != canonical]
        return {
            "canonical": canonical,
            "synset": synset_name,
            "aliases": aliases,
            "definition": syn.definition(),
        }
    except Exception:
        return None


def ensure_nltk_wordnet() -> bool:
    """Download WordNet if needed. Returns True if usable."""
    try:
        import nltk
        from nltk.corpus import wordnet as wn

        try:
            wn.synsets("dog")
            return True
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            wn.synsets("dog")
            return True
    except Exception:
        return False


def init_from_wordnet(
    session: Session,
    *,
    synset_names: list[str] | None,
    graph_version: str,
    data_version: str,
    run_id: str | None = None,
) -> int:
    """Seed graph_node from WordNet / fallback. Returns number of nodes added."""
    ensure_nltk_wordnet()
    added = 0
    seen: set[str] = set()

    concepts: list[dict[str, Any]] = []
    if synset_names:
        for name in synset_names:
            c = _try_wordnet_concept(name)
            if c:
                concepts.append(c)
    if len(concepts) < 20:
        concepts = FALLBACK_CONCEPTS

    for c in concepts:
        key = c["canonical"].lower()
        if key in seen:
            continue
        seen.add(key)
        existing = (
            session.query(GraphNode)
            .filter_by(canonical=c["canonical"], graph_version=graph_version)
            .one_or_none()
        )
        if existing:
            continue
        node = GraphNode(
            canonical=c["canonical"],
            synset=c.get("synset"),
            node_type="concept",
            aliases=c.get("aliases") or [],
            meta={"definition": c.get("definition")} if c.get("definition") else None,
            graph_version=graph_version,
            data_version=data_version,
            run_id=run_id,
        )
        session.add(node)
        added += 1
    session.commit()
    return added


def text_link(session: Session, term: str, graph_version: str) -> Optional[GraphNode]:
    """Link a candidate term to an existing node via exact canonical / alias match.

    Soft substring matching is intentionally avoided (e.g. "car" in "scarlet").
    """
    term_l = term.strip().lower()
    nodes = session.query(GraphNode).filter_by(graph_version=graph_version, valid_status="valid").all()
    # Exact canonical first
    for n in nodes:
        if n.canonical.lower() == term_l:
            return n
    # Exact alias
    for n in nodes:
        aliases = n.aliases or []
        if any(a.lower() == term_l for a in aliases):
            return n
    # Token-set equality for multi-word (order-insensitive)
    term_tokens = set(term_l.replace("-", " ").split())
    if len(term_tokens) >= 2:
        for n in nodes:
            canon_tokens = set(n.canonical.lower().replace("-", " ").split())
            if term_tokens == canon_tokens:
                return n
    return None


def vision_verify_placeholder(term: str, node: GraphNode | None) -> dict[str, Any]:
    """[engineering] Visual verification of term↔node is a placeholder in MVP."""
    return {
        "term": term,
        "linked_node": node.canonical if node else None,
        "accepted": node is not None,
        "backend": "placeholder",
        "note": "Vision-language verification not run in MVP; text link only.",
    }


def add_candidate_term(
    session: Session,
    term: str,
    *,
    graph_version: str,
    data_version: str,
    run_id: str | None = None,
    force_new: bool = False,
) -> dict[str, Any]:
    """add_candidate_term → text link → vision placeholder → merge aliases / add node."""
    linked = text_link(session, term, graph_version)
    vision = vision_verify_placeholder(term, linked)

    if linked and not force_new:
        aliases = list(linked.aliases or [])
        if term.lower() not in {a.lower() for a in aliases} and term.lower() != linked.canonical.lower():
            aliases.append(term)
            linked.aliases = aliases
            session.commit()
            action = "merged_alias"
        else:
            action = "already_linked"
        return {
            "action": action,
            "node_id": linked.id,
            "canonical": linked.canonical,
            "vision": vision,
        }

    node = GraphNode(
        canonical=term.strip().lower(),
        synset=None,
        node_type="concept",
        aliases=[],
        meta={"source": "add_candidate_term", "vision": vision},
        graph_version=graph_version,
        data_version=data_version,
        run_id=run_id,
    )
    session.add(node)
    session.commit()
    return {
        "action": "created",
        "node_id": node.id,
        "canonical": node.canonical,
        "vision": vision,
    }


def export_graph(session: Session, graph_version: str, out_path: Path) -> dict[str, Any]:
    nodes = session.query(GraphNode).filter_by(graph_version=graph_version).all()
    edges = session.query(GraphEdge).filter_by(graph_version=graph_version).all()
    payload = {
        "graph_version": graph_version,
        "nodes": [
            {
                "id": n.id,
                "canonical": n.canonical,
                "synset": n.synset,
                "aliases": n.aliases,
                "node_type": n.node_type,
                "valid_status": n.valid_status,
            }
            for n in nodes
        ],
        "edges": [
            {
                "id": e.id,
                "src_id": e.src_id,
                "dst_id": e.dst_id,
                "relation": e.relation,
            }
            for e in edges
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return {"nodes": len(nodes), "edges": len(edges), "path": str(out_path)}
