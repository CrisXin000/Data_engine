#!/usr/bin/env python3
"""Write COMPARISON.md for rule vs vlm_json smoke runs."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def by_image_query(exp: dict) -> dict:
    images = {i["id"]: Path(i["path"]).name for i in exp["image"]}
    queries = {q["id"]: q for q in exp["query"]}
    cands = {c["id"]: c for c in exp["candidate"]}
    out: dict = defaultdict(Counter)
    for v in exp["verification"]:
        c = cands[v["candidate_id"]]
        q = queries[c["query_id"]]
        key = (images[q["image_id"]], q["text"], q["query_type"])
        out[key][v["decision"]] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-id", required=True)
    ap.add_argument("--vlm-id", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    rule = json.loads(Path(f"data/runs/{args.rule_id}/export.json").read_text(encoding="utf-8"))
    vlm = json.loads(Path(f"data/runs/{args.vlm_id}/export.json").read_text(encoding="utf-8"))

    rd = Counter(v["decision"] for v in rule["verification"])
    vd = Counter(v["decision"] for v in vlm["verification"])

    for v in vlm["verification"]:
        raw = v.get("raw_output") or {}
        assert raw.get("mock") is False, "expected mock:false"
        assert raw.get("stands_in_for") == "Qwen3-VL", raw
        model = str(raw.get("model") or "")
        assert "Qwen3" not in model, f"must not claim Qwen3 model: {model}"

    print(f"[rule] decisions={dict(rd)}")
    print(f"[vlm_json] decisions={dict(vd)}")

    br = by_image_query(rule)
    bv = by_image_query(vlm)
    keys = sorted(set(br) | set(bv))

    samples = []
    for v in vlm["verification"][:3]:
        raw = v.get("raw_output") or {}
        samples.append(
            {
                "decision": v["decision"],
                "reason": (v.get("reason") or "")[:140],
                "model": raw.get("model"),
                "vlm_json_impl": raw.get("vlm_json_impl"),
                "fallback_reason": (raw.get("fallback_reason") or "")[:180],
                "sim_crop": raw.get("sim_crop"),
                "stands_in_for": raw.get("stands_in_for"),
                "mock": raw.get("mock"),
                "prompt_version": raw.get("prompt_version"),
            }
        )

    lines = [
        "# Rule vs VLM/CLIP semantic verify",
        "",
        f"- rule run_id: `{args.rule_id}`",
        f"- vlm_json run_id: `{args.vlm_id}`",
        "- images: `data/gold/images_real_smoke_vlm/` (bus + dog preferred)",
        "- detectors: YOLO-World (Expert A) + OWLv2 (Expert B); verify after detector unload",
        "",
        "| | rule | vlm_json (CLIP fallback) |",
        "|--|------|--------------------------|",
        f"| accept | {rd.get('accept', 0)} | {vd.get('accept', 0)} |",
        f"| reject | {rd.get('reject', 0)} | {vd.get('reject', 0)} |",
        f"| duplicate | {rd.get('duplicate', 0)} | {vd.get('duplicate', 0)} |",
        f"| uncertain | {rd.get('uncertain', 0)} | {vd.get('uncertain', 0)} |",
        "",
        "## Notes",
        "",
        "- Paper verifier is **Qwen3-VL**. This smoke uses **CLIP ViT-B/32** as a weak open-source stand-in",
        "  (`vlm_json` tries Florence-2-base first; current transformers 5.x breaks Florence remote code,",
        "  so we auto-fall back to CLIP and record `fallback_reason` in `raw_output`).",
        "- Rule accepts by detector score + IoU only (no semantics). CLIP rejects crops whose",
        "  text similarity to the query is low, and uses `uncertain` for the middle band",
        "  (never treats unknown as accept).",
        f"- Viz: `data/runs/{args.rule_id}/viz/` vs `data/runs/{args.vlm_id}/viz/`",
        "",
        "## Per image/query",
        "",
        "| image | query | type | rule | vlm_json |",
        "|-------|-------|------|------|----------|",
    ]
    for k in keys:
        img, text, qt = k
        lines.append(
            f"| {img} | {text[:40]} | {qt} | {dict(br.get(k, {}))} | {dict(bv.get(k, {}))} |"
        )
    lines += [
        "",
        "## Sample vlm raw_output",
        "",
        "```json",
        json.dumps(samples, indent=2, ensure_ascii=False),
        "```",
        "",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_dir / "meta.json").write_text(
        json.dumps(
            {
                "rule_run_id": args.rule_id,
                "vlm_run_id": args.vlm_id,
                "rule_decisions": dict(rd),
                "vlm_decisions": dict(vd),
                "model": "openai/CLIP-ViT-B/32",
                "stands_in_for": "Qwen3-VL",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", args.out_dir / "COMPARISON.md")


if __name__ == "__main__":
    main()
