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
from wde.matte import matte_instance, release_matte_model
from wde.propose import build_experts, run_experts
from wde.schema.models import (
    Candidate,
    Image,
    InstanceLink,
    MaskAlpha,
    Query,
    Verification,
)
from wde.segment import release_segment_model, segment_instance
from wde.store.db import get_session, init_db, make_engine
from wde.utils import file_sha256, new_run_id
from wde.verify import build_verifier, release_expert_models


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
    verify_backend: Optional[str] = None,
    referring_generator: Optional[str] = None,
    segment_backend: Optional[str] = None,
    matte_backend: Optional[str] = None,
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
            # M2: default referring to grounded structured stand-in (NOT Qwen3-VL)
            if disc.get("referring_generator", "mock") == "mock":
                disc["referring_generator"] = "structured"
            if prop.get("expert_a", "mock") == "mock":
                prop["expert_a"] = "yolo_world"
            if prop.get("expert_b", "mock") == "mock":
                prop["expert_b"] = "owlv2"
            # verify backend: keep yaml / CLI; do NOT force rule in real mode

    if verify_backend:
        cfg["models"]["verify"]["backend"] = verify_backend
    if referring_generator:
        cfg["models"]["discover"]["referring_generator"] = referring_generator
    if segment_backend:
        if segment_backend.strip().lower() == "skip":
            cfg["pipeline"]["skip_segment"] = True
        else:
            cfg["pipeline"]["skip_segment"] = False
            cfg["models"]["segment"]["backend"] = segment_backend
    if matte_backend:
        if matte_backend.strip().lower() == "skip":
            cfg["pipeline"]["skip_matte"] = True
        else:
            cfg["pipeline"]["skip_matte"] = False
            cfg["models"]["matte"]["backend"] = matte_backend
    # mock mode always forces rule + mock referring (ignore CLI/yaml for fast regression)
    # Segment stays off unless --segment / yaml skip_segment=false is set (including mock).
    if cfg["models"].get("mode") == "mock":
        cfg["models"]["verify"]["backend"] = "rule"
        cfg["models"]["discover"]["referring_generator"] = "mock"

    run_id = run_id or new_run_id()
    data_version = cfg["data_version"]
    graph_version = cfg["graph_version"]
    engine = init_db(cfg["paths"]["db"])
    session = get_session(engine)

    tagger = build_category_tagger(cfg["models"]["discover"]["category_tagger"])
    ref_kwargs = {}
    disc_cfg = cfg["models"]["discover"]
    if "referring_enable_caption" in disc_cfg:
        ref_kwargs["enable_caption"] = bool(disc_cfg["referring_enable_caption"])
    ref_gen = build_referring_generator(disc_cfg["referring_generator"], **ref_kwargs)
    experts = build_experts(cfg["models"]["propose"])
    verify_cfg = dict(cfg["models"].get("verify") or {})
    verify_name = str(verify_cfg.get("backend", "rule"))
    verifier = build_verifier(verify_name, verify_cfg)
    iou_dedup = float(cfg["pipeline"]["iou_dedup_threshold"])
    accept_min = float(cfg["pipeline"]["accept_min_score"])
    skip_seg = bool(cfg["pipeline"].get("skip_segment", True))
    skip_matte = bool(cfg["pipeline"].get("skip_matte", True))
    # Semantic backends need RAM; unload detectors before loading VLM/CLIP.
    needs_image_verify = verify_name.lower() not in {"rule"}

    stats = {
        "run_id": run_id,
        "data_version": data_version,
        "mode": cfg["models"]["mode"],
        "verify_backend": verify_name,
        "images": 0,
        "queries": 0,
        "candidates": 0,
        "verifications": 0,
        "instance_links": 0,
        "masks": 0,
        "alphas": 0,
        "segment_backend": cfg["models"].get("segment", {}).get("backend", "stub"),
        "skip_segment": bool(cfg["pipeline"].get("skip_segment", True)),
        "matte_backend": cfg["models"].get("matte", {}).get("backend", "stub"),
        "skip_matte": bool(cfg["pipeline"].get("skip_matte", True)),
    }

    run_dir = Path(cfg["paths"]["runs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pending_segments: list[dict[str, Any]] = []

    paths = [Path(p) for p in image_paths]
    for path in paths:
        if not path.exists():
            continue
        img = _ingest_image(session, path, run_id=run_id, data_version=data_version)
        stats["images"] += 1

        discovered = discover_queries(path, category_tagger=tagger, referring_generator=ref_gen)
        # Phase 1: propose for all queries while detectors are resident.
        pending: list[dict[str, Any]] = []
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

            pending.append(
                {
                    "query": q,
                    "dq": dq,
                    "proposals": proposals,
                    "cand_rows": cand_rows,
                }
            )

        # Phase 2: free detector RAM before loading CLIP / Florence.
        if needs_image_verify:
            release_expert_models(experts)
            # Category tagger may hold another YOLO-World copy
            if hasattr(tagger, "_model"):
                try:
                    tagger._model = None
                except Exception:
                    pass
            # Structured referring unloads YOLO per call; clear any leftover attr
            if hasattr(ref_gen, "_model"):
                try:
                    ref_gen._model = None
                except Exception:
                    pass

        for item in pending:
            q = item["query"]
            dq = item["dq"]
            proposals = item["proposals"]
            cand_rows = item["cand_rows"]
            decisions = verifier.verify(
                dq.text,
                dq.query_type,
                proposals,
                iou_dedup=iou_dedup,
                accept_min_score=accept_min,
                image=path,
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

                    if not skip_seg or not skip_matte:
                        pending_segments.append(
                            {
                                "link_id": link.id,
                                "image_path": str(path),
                                "box": (cand.x1, cand.y1, cand.x2, cand.y2),
                                "instance_uid": uid,
                                "query_id": q.id,
                                "query_text": q.text,
                            }
                        )

        session.commit()

    # Phase 3: unload detectors + VLM/CLIP before loading a segmenter (CPU RAM).
    try:
        verifier.release()
    except Exception:
        pass
    release_expert_models(experts)
    if hasattr(tagger, "_model"):
        try:
            tagger._model = None
        except Exception:
            pass
    if hasattr(ref_gen, "_model"):
        try:
            ref_gen._model = None
        except Exception:
            pass

    if pending_segments and not skip_seg:
        seg_cfg = dict(cfg["models"].get("segment") or {})
        mask_dir = run_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for item in pending_segments:
            out_path = mask_dir / f"{item['instance_uid']}.png"
            mr = segment_instance(
                image_path=item["image_path"],
                box_xyxy=item["box"],
                backend=str(seg_cfg.get("backend", "stub")),
                output_path=str(out_path),
                instance_uid=item["instance_uid"],
                query_text=item["query_text"],
                query_id=item["query_id"],
                extra_meta={"instance_link_id": item["link_id"]},
                weights=seg_cfg.get("weights"),
                device=str(seg_cfg.get("device") or "cpu"),
                imgsz=int(seg_cfg.get("imgsz") or 512),
            )
            session.add(
                MaskAlpha(
                    instance_link_id=item["link_id"],
                    kind=mr.kind,
                    path=mr.path,
                    source=mr.source,
                    meta=mr.meta,
                    run_id=run_id,
                    data_version=data_version,
                )
            )
            item["mask_path"] = mr.path
            stats["masks"] += 1
        try:
            release_segment_model()
        except Exception:
            pass
        session.commit()

    if pending_segments and not skip_matte:
        # Unload the segmenter before loading a matting session (CPU RAM).
        try:
            release_segment_model()
        except Exception:
            pass
        matte_cfg = dict(cfg["models"].get("matte") or {})
        alpha_dir = run_dir / "alphas"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        for item in pending_segments:
            out_path = alpha_dir / f"{item['instance_uid']}.png"
            ar = matte_instance(
                image_path=item["image_path"],
                box_xyxy=item["box"],
                backend=str(matte_cfg.get("backend", "stub")),
                mask_path=item.get("mask_path"),
                output_path=str(out_path),
                instance_uid=item["instance_uid"],
                query_text=item["query_text"],
                query_id=item["query_id"],
                extra_meta={"instance_link_id": item["link_id"]},
                weights=matte_cfg.get("weights"),
                device=str(matte_cfg.get("device") or "cpu"),
                imgsz=int(matte_cfg.get("imgsz") or 1024),
                crop_pad_ratio=float(matte_cfg.get("crop_pad_ratio") or 0.15),
                bin_threshold=float(matte_cfg.get("bin_threshold") or 0.5),
                iou_min=float(matte_cfg.get("iou_min") or 0.5),
                coverage_min=float(matte_cfg.get("coverage_min") or 0.6),
                area_ratio_min=float(matte_cfg.get("area_ratio_min") or 0.5),
                area_ratio_max=float(matte_cfg.get("area_ratio_max") or 2.0),
                fallback=str(matte_cfg.get("fallback") or "feather"),
                feather_radius=float(matte_cfg.get("feather_radius") or 3.0),
                feather_band=float(matte_cfg.get("feather_band") or 4.0),
            )
            session.add(
                MaskAlpha(
                    instance_link_id=item["link_id"],
                    kind=ar.kind,
                    path=ar.path,
                    source=ar.source,
                    meta=ar.meta,
                    run_id=run_id,
                    data_version=data_version,
                )
            )
            stats["alphas"] += 1
        try:
            release_matte_model()
        except Exception:
            pass
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
            "M1+M2 pipeline — mode="
            + str(cfg["models"]["mode"])
            + "; verify="
            + str(verify_name)
            + "; referring="
            + str(cfg["models"]["discover"].get("referring_generator"))
            + "; real experts are YOLO-World / OWLv2 open-source substitutes "
            "(NOT paper SAM3 / in-house DINO); "
            "referring structured/grounded/qwen3_vl-alias and verify clip/vlm_json "
            "are open-source stand-ins for Qwen3-VL (NOT Qwen3-VL); "
            "segment="
            + str(cfg["models"].get("segment", {}).get("backend"))
            + " (ultralytics SAM2-Tiny / MobileSAM / box_fill stand-in for paper SAM2; "
            "NOT the official Meta sam2 package); "
            "matte="
            + str(cfg["models"].get("matte", {}).get("backend"))
            + " (ONNX BiRefNet-General-Lite stand-in for paper BiRefNet/MatAnyone2; "
            "NOT MatAnyone2, NOT a trained paper matting annotator)"
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
