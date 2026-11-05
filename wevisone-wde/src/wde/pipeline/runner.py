"""End-to-end WDE pipeline for M0+M1 (category box single line)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image as PILImage
from sqlalchemy.orm import Session

from wde.config import load_config
from wde.discover import build_category_tagger, build_referring_generator, discover_queries
from wde.kg.graph import text_link
from wde.matte import matte_instance
from wde.propose import build_experts, run_experts
from wde.schema.models import (
    Candidate,
    Image,
    InstanceLink,
    MaskAlpha,
    Query,
    Verification,
)
from wde.segment import segment_instance
from wde.store.db import get_session, init_db, make_engine
from wde.utils import file_sha256, new_run_id
from wde.verify import build_verifier


def _ingest_image(
    session: Session,
    path: Path,
    *,
    run_id: str,
    data_version: str,
    source: str = "ingest",
) -> Image:
    existing = session.query(Image).filter_by(path=str(path.resolve())).one_or_none()
    if existing:
        return existing
    with PILImage.open(path) as im:
        w, h = im.size
    img = Image(
        path=str(path.resolve()),
        width=w,
        height=h,
        source=source,
        sha256=file_sha256(path),
        run_id=run_id,
        data_version=data_version,
    )
    session.add(img)
    session.flush()
    return img


def run_pipeline(
    image_paths: Iterable[str | Path],
    *,
    run_id: Optional[str] = None,
    mode: str = "mock",
    config_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if mode:
        cfg["models"]["mode"] = mode
        if mode == "mock":
            cfg["models"]["discover"]["category_tagger"] = "mock"
            cfg["models"]["discover"]["referring_generator"] = "mock"
            cfg["models"]["propose"]["expert_a"] = "mock"
            cfg["models"]["propose"]["expert_b"] = "mock"
            cfg["models"]["verify"]["backend"] = "rule"
        elif mode == "real":
            # [engineering] Open-source substitutes for paper SAM3 + in-house DINO OVD.
            # Honor explicit non-mock config values; otherwise default to YOLO-World / OWLv2.
            disc = cfg["models"]["discover"]
            prop = cfg["models"]["propose"]
            if disc.get("category_tagger", "mock") == "mock":
                disc["category_tagger"] = "yolo_world"
            # referring stays mock (Qwen3-VL reserved) unless user set something else
            if prop.get("expert_a", "mock") == "mock":
                prop["expert_a"] = "yolo_world"
            if prop.get("expert_b", "mock") == "mock":
                prop["expert_b"] = "owlv2"

    run_id = run_id or new_run_id()
    data_version = cfg["data_version"]
    graph_version = cfg["graph_version"]
    engine = init_db(cfg["paths"]["db"])
    session = get_session(engine)

    tagger = build_category_tagger(cfg["models"]["discover"]["category_tagger"])
    ref_gen = build_referring_generator(cfg["models"]["discover"]["referring_generator"])
    experts = build_experts(cfg["models"]["propose"])
    verifier = build_verifier(cfg["models"]["verify"]["backend"])
    iou_dedup = float(cfg["pipeline"]["iou_dedup_threshold"])
    accept_min = float(cfg["pipeline"]["accept_min_score"])
    skip_seg = bool(cfg["pipeline"].get("skip_segment", True))
    skip_matte = bool(cfg["pipeline"].get("skip_matte", True))

    stats = {
        "run_id": run_id,
        "data_version": data_version,
        "mode": cfg["models"]["mode"],
        "images": 0,
        "queries": 0,
        "candidates": 0,
        "verifications": 0,
        "instance_links": 0,
        "masks": 0,
        "alphas": 0,
    }

    run_dir = Path(cfg["paths"]["runs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in image_paths]
    for path in paths:
        if not path.exists():
            continue
        img = _ingest_image(session, path, run_id=run_id, data_version=data_version)
        stats["images"] += 1

        discovered = discover_queries(path, category_tagger=tagger, referring_generator=ref_gen)
        for dq in discovered:
            node = text_link(session, dq.text if dq.query_type == "category" else dq.meta.get("base_category", dq.text), graph_version)
            is_neg = bool(dq.meta.get("is_negative_hint"))
            q = Query(
                image_id=img.id,
                text=dq.text,
                query_type=dq.query_type,
                graph_node_id=node.id if node else None,
                source="discover",
                is_negative=is_neg,
                meta=dq.meta,
                run_id=run_id,
                data_version=data_version,
            )
            session.add(q)
            session.flush()
            stats["queries"] += 1

            proposals = run_experts(experts, path, dq.text, dq.query_type)
            cand_rows: list[Candidate] = []
            for prop in proposals:
                c = Candidate(
                    query_id=q.id,
                    expert_name=prop.expert_name,
                    x1=prop.x1,
                    y1=prop.y1,
                    x2=prop.x2,
                    y2=prop.y2,
                    score=prop.score,
                    raw_output=prop.raw,
                    run_id=run_id,
                    data_version=data_version,
                )
                session.add(c)
                session.flush()
                cand_rows.append(c)
                stats["candidates"] += 1

            decisions = verifier.verify(
                dq.text,
                dq.query_type,
                proposals,
                iou_dedup=iou_dedup,
                accept_min_score=accept_min,
            )
            for dec in decisions:
                cand = cand_rows[dec.proposal_index]
                v = Verification(
                    candidate_id=cand.id,
                    decision=dec.decision,
                    reason=dec.reason,
                    score=dec.score,
                    backend=dec.backend,
                    raw_output=dec.raw,
                    run_id=run_id,
                    data_version=data_version,
                )
                session.add(v)
                session.flush()
                stats["verifications"] += 1

                if dec.decision == "accept":
                    uid = f"inst-{uuid.uuid4().hex[:12]}"
                    link = InstanceLink(
                        query_id=q.id,
                        candidate_id=cand.id,
                        verification_id=v.id,
                        instance_uid=uid,
                        x1=cand.x1,
                        y1=cand.y1,
                        x2=cand.x2,
                        y2=cand.y2,
                        run_id=run_id,
                        data_version=data_version,
                    )
                    session.add(link)
                    session.flush()
                    stats["instance_links"] += 1

                    if not skip_seg:
                        mr = segment_instance(
                            image_path=str(path),
                            box_xyxy=(cand.x1, cand.y1, cand.x2, cand.y2),
                            backend=cfg["models"]["segment"]["backend"],
                        )
                        session.add(
                            MaskAlpha(
                                instance_link_id=link.id,
                                kind=mr.kind,
                                path=mr.path,
                                source=mr.source,
                                meta=mr.meta,
                                run_id=run_id,
                                data_version=data_version,
                            )
                        )
                        stats["masks"] += 1
                    if not skip_matte:
                        ar = matte_instance(
                            image_path=str(path),
                            box_xyxy=(cand.x1, cand.y1, cand.x2, cand.y2),
                            backend=cfg["models"]["matte"]["backend"],
                        )
                        session.add(
                            MaskAlpha(
                                instance_link_id=link.id,
                                kind=ar.kind,
                                path=ar.path,
                                source=ar.source,
                                meta=ar.meta,
                                run_id=run_id,
                                data_version=data_version,
                            )
                        )
                        stats["alphas"] += 1

        session.commit()

    # write run manifest
    manifest = {
        "run_id": run_id,
        "stats": stats,
        "config_snapshot": {
            "models": cfg["models"],
            "pipeline": cfg["pipeline"],
            "data_version": data_version,
            "graph_version": graph_version,
        },
        "note": (
            "M1 pipeline — mode="
            + str(cfg["models"]["mode"])
            + "; real experts are YOLO-World / OWLv2 open-source substitutes "
            "(NOT paper SAM3 / in-house DINO)"
        ),
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    session.close()
    return stats


def export_run(run_id: str, out_path: str | Path, config_path: Optional[str | Path] = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    engine = make_engine(cfg["paths"]["db"])
    session = get_session(engine)

    def _row(obj: Any, fields: list[str]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k in fields:
            v = getattr(obj, k)
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            d[k] = v
        return d

    images = session.query(Image).filter_by(run_id=run_id).all()
    # Also include gold images referenced by queries in this run
    q_image_ids = {q.image_id for q in session.query(Query).filter_by(run_id=run_id).all()}
    if q_image_ids:
        extra = session.query(Image).filter(Image.id.in_(q_image_ids)).all()
        seen = {i.id for i in images}
        for e in extra:
            if e.id not in seen:
                images.append(e)

    queries = session.query(Query).filter_by(run_id=run_id).all()
    candidates = session.query(Candidate).filter_by(run_id=run_id).all()
    verifications = session.query(Verification).filter_by(run_id=run_id).all()
    links = session.query(InstanceLink).filter_by(run_id=run_id).all()
    masks = session.query(MaskAlpha).filter_by(run_id=run_id).all()

    payload = {
        "run_id": run_id,
        "data_version": cfg["data_version"],
        "image": [
            _row(i, ["id", "path", "width", "height", "source", "sha256", "created_at", "run_id", "data_version", "valid_status"])
            for i in images
        ],
        "query": [
            _row(q, ["id", "image_id", "text", "query_type", "language", "graph_node_id", "source", "is_negative", "meta", "created_at", "run_id", "data_version", "valid_status"])
            for q in queries
        ],
        "candidate": [
            _row(c, ["id", "query_id", "expert_name", "x1", "y1", "x2", "y2", "score", "raw_output", "created_at", "run_id", "data_version", "valid_status"])
            for c in candidates
        ],
        "verification": [
            _row(v, ["id", "candidate_id", "decision", "reason", "score", "backend", "raw_output", "created_at", "run_id", "data_version", "valid_status"])
            for v in verifications
        ],
        "instance_link": [
            _row(l, ["id", "query_id", "candidate_id", "verification_id", "instance_uid", "x1", "y1", "x2", "y2", "created_at", "run_id", "data_version", "valid_status"])
            for l in links
        ],
        "mask_alpha": [
            _row(m, ["id", "instance_link_id", "kind", "path", "source", "meta", "created_at", "run_id", "data_version", "valid_status"])
            for m in masks
        ],
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    summary = {k: len(payload[k]) for k in ("image", "query", "candidate", "verification", "instance_link", "mask_alpha")}
    summary["path"] = str(out)
    summary["run_id"] = run_id
    session.close()
    return summary
