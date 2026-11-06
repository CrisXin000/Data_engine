"""CLI entry: python -m wde.cli ..."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from wde.config import load_config, project_root
from wde.kg.graph import add_candidate_term, export_graph, init_from_wordnet
from wde.pipeline.runner import export_run, run_pipeline
from wde.store.db import get_session, init_db, make_engine, table_counts
from wde.utils import new_run_id

console = Console()


@click.group()
@click.option("--config", "config_path", default=None, help="Path to YAML config")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """WeVisOne-WDE MVP CLI (M0–M3 + optional M4 matte)."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["cfg"] = load_config(config_path)


@main.command("init-db")
@click.pass_context
def init_db_cmd(ctx: click.Context) -> None:
    """Create SQLite tables."""
    cfg = ctx.obj["cfg"]
    engine = init_db(cfg["paths"]["db"])
    counts = table_counts(engine)
    console.print(f"[green]DB ready:[/green] {cfg['paths']['db']}")
    console.print(counts)


@main.command("ingest-gold")
@click.pass_context
def ingest_gold_cmd(ctx: click.Context) -> None:
    """Ingest synthetic gold images + annotations into DB."""
    from wde.gold import ingest_gold

    cfg = ctx.obj["cfg"]
    init_db(cfg["paths"]["db"])
    result = ingest_gold(cfg)
    console.print(f"[green]Gold ingested:[/green] {result}")


@main.command("run")
@click.option("--images", "images_dir", required=True, type=click.Path(exists=True))
@click.option("--mode", default="mock", type=click.Choice(["mock", "real"]))
@click.option(
    "--verify",
    "verify_backend",
    default=None,
    type=click.Choice(["rule", "vlm_json", "vlm", "florence2", "clip_semantic", "clip"]),
    help="Override models.verify.backend (real mode). mock mode still forces rule.",
)
@click.option(
    "--referring",
    "referring_generator",
    default=None,
    type=click.Choice(["mock", "structured", "grounded", "qwen3_vl"]),
    help=(
        "Override models.discover.referring_generator. "
        "structured/grounded = YOLO-World+crop stats (stand-in for Qwen3-VL); "
        "qwen3_vl = alias→structured (NOT real Qwen3-VL). mock mode still forces mock."
    ),
)
@click.option(
    "--segment",
    "segment_backend",
    default=None,
    type=click.Choice(["stub", "sam2", "mobile_sam", "box_fill", "skip"]),
    help=(
        "Override models.segment.backend and set skip_segment=false. "
        "sam2 = ultralytics SAM2-Tiny stand-in (NOT official Meta sam2). "
        "mobile_sam / box_fill = weaker fallbacks. skip = leave skip_segment=true."
    ),
)
@click.option(
    "--matte",
    "matte_backend",
    default=None,
    type=click.Choice(
        ["stub", "birefnet_lite", "birefnet", "birefnet_matanyone", "feather", "copy_s", "skip"]
    ),
    help=(
        "Override models.matte.backend and set skip_matte=false. "
        "birefnet_lite/birefnet = ONNX BiRefNet-General-Lite stand-in for paper "
        "BiRefNet/MatAnyone2 (NOT MatAnyone2, NOT a trained paper annotator). "
        "feather/copy_s = soften or copy the binary mask S. skip = leave skip_matte=true."
    ),
)
@click.option("--run-id", default=None)
@click.pass_context
def run_cmd(
    ctx: click.Context,
    images_dir: str,
    mode: str,
    verify_backend: str | None,
    referring_generator: str | None,
    segment_backend: str | None,
    matte_backend: str | None,
    run_id: str | None,
) -> None:
    """Run discover (incl. referring) → propose → verify → (optional segment/matte) → write DB."""
    cfg = ctx.obj["cfg"]
    init_db(cfg["paths"]["db"])
    paths = sorted(Path(images_dir).glob("*"))
    paths = [p for p in paths if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
    if not paths:
        raise click.ClickException(f"No images under {images_dir}")
    rid = run_id or new_run_id()
    stats = run_pipeline(
        paths,
        run_id=rid,
        mode=mode,
        config_path=ctx.obj["config_path"],
        verify_backend=verify_backend,
        referring_generator=referring_generator,
        segment_backend=segment_backend,
        matte_backend=matte_backend,
    )
    console.print("[green]Pipeline done[/green]")
    console.print(stats)
    console.print(f"Run dir: {Path(cfg['paths']['runs']) / rid}")


@main.command("export")
@click.option("--run-id", required=True)
@click.option("--out", required=True, type=click.Path())
@click.pass_context
def export_cmd(ctx: click.Context, run_id: str, out: str) -> None:
    """Export run tables to JSON."""
    summary = export_run(run_id, out, config_path=ctx.obj["config_path"])
    console.print("[green]Exported[/green]", summary)


@main.command("status")
@click.pass_context
def status_cmd(ctx: click.Context) -> None:
    """Show DB table row counts and recent runs."""
    cfg = ctx.obj["cfg"]
    db = Path(cfg["paths"]["db"])
    if not db.exists():
        console.print("[yellow]DB not initialized.[/yellow] Run: python -m wde.cli init-db")
        return
    engine = make_engine(db)
    counts = table_counts(engine)
    table = Table(title="WDE SQLite tables")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for k, v in counts.items():
        table.add_row(k, str(v))
    console.print(table)

    runs_dir = Path(cfg["paths"]["runs"])
    if runs_dir.exists():
        runs = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True)[:10]
        console.print("Recent runs:", runs or "(none)")


@main.group()
def kg() -> None:
    """Knowledge graph commands."""


@kg.command("init")
@click.pass_context
def kg_init(ctx: click.Context) -> None:
    """Initialize KG from WordNet / fallback concepts."""
    cfg = ctx.obj["cfg"]
    engine = init_db(cfg["paths"]["db"])
    session = get_session(engine)
    n = init_from_wordnet(
        session,
        synset_names=cfg.get("kg", {}).get("wordnet_synsets"),
        graph_version=cfg["graph_version"],
        data_version=cfg["data_version"],
    )
    console.print(f"[green]KG init[/green]: added {n} nodes (version={cfg['graph_version']})")
    session.close()


@kg.command("add")
@click.argument("term")
@click.option("--force-new", is_flag=True)
@click.pass_context
def kg_add(ctx: click.Context, term: str, force_new: bool) -> None:
    """Add candidate term → text link → merge aliases / create node."""
    cfg = ctx.obj["cfg"]
    engine = init_db(cfg["paths"]["db"])
    session = get_session(engine)
    result = add_candidate_term(
        session,
        term,
        graph_version=cfg["graph_version"],
        data_version=cfg["data_version"],
        force_new=force_new,
    )
    console.print(result)
    session.close()


@kg.command("export")
@click.option("--out", default=None)
@click.pass_context
def kg_export(ctx: click.Context, out: str | None) -> None:
    """Export graph nodes/edges to JSON."""
    cfg = ctx.obj["cfg"]
    engine = make_engine(cfg["paths"]["db"])
    session = get_session(engine)
    out_path = Path(out) if out else Path(cfg["paths"]["runs"]) / "kg_export.json"
    result = export_graph(session, cfg["graph_version"], out_path)
    console.print("[green]KG exported[/green]", result)
    session.close()


if __name__ == "__main__":
    main()
