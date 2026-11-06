"""SQLAlchemy ORM models for WDE tables.

All tables carry created_at, run_id, data_version, valid_status.
Raw expert / VLM outputs are append-only (never in-place overwrite).
Coordinates are xyxy in original image pixels.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "image"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="ingest")  # gold | ingest | acquire
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")  # valid|invalid|pending

    queries: Mapped[list["Query"]] = relationship(back_populates="image")


class Query(Base):
    __tablename__ = "query"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(32), nullable=False)  # category | referring
    language: Mapped[str] = mapped_column(String(16), default="en")
    graph_node_id: Mapped[Optional[int]] = mapped_column(ForeignKey("graph_node.id"), nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="discover")  # discover|gold|manual
    is_negative: Mapped[bool] = mapped_column(Boolean, default=False)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")

    image: Mapped["Image"] = relationship(back_populates="queries")
    candidates: Mapped[list["Candidate"]] = relationship(back_populates="query")


class Candidate(Base):
    """Localization proposal from one expert. Append-only raw output."""

    __tablename__ = "candidate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("query.id"), nullable=False, index=True)
    expert_name: Mapped[str] = mapped_column(String(64), nullable=False)  # ExpertA|ExpertB|...
    # xyxy original pixels
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    raw_output: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")

    query: Mapped["Query"] = relationship(back_populates="candidates")
    verifications: Mapped[list["Verification"]] = relationship(back_populates="candidate")


class Verification(Base):
    """VLM / rule verification decision for a candidate. Append-only."""

    __tablename__ = "verification"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidate.id"), nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)  # accept|reject|uncertain|duplicate
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    backend: Mapped[str] = mapped_column(String(64), default="rule")  # rule | vlm_json
    raw_output: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")

    candidate: Mapped["Candidate"] = relationship(back_populates="verifications")


class InstanceLink(Base):
    """Accepted query↔instance binding after verification (cross-task identity)."""

    __tablename__ = "instance_link"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("query.id"), nullable=False, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidate.id"), nullable=False, index=True)
    verification_id: Mapped[int] = mapped_column(ForeignKey("verification.id"), nullable=False)
    instance_uid: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # retained box (xyxy)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")


class MaskAlpha(Base):
    """SAM2 mask / matting alpha — stub in M1."""

    __tablename__ = "mask_alpha"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instance_link_id: Mapped[int] = mapped_column(ForeignKey("instance_link.id"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # mask | alpha
    path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="stub")  # stub|sam2|birefnet|matanyone
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")


class ErrorRecord(Base):
    """Model / annotation error for feedback loop (§2.7)."""

    __tablename__ = "error_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[Optional[int]] = mapped_column(ForeignKey("image.id"), nullable=True)
    query_id: Mapped[Optional[int]] = mapped_column(ForeignKey("query.id"), nullable=True)
    failure_type: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")


class Revision(Base):
    """Annotation correction history — never overwrite originals."""

    __tablename__ = "revision"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_table: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)  # correct|invalidate|regenerate
    before: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    after: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    reviewer: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")


class GraphNode(Base):
    __tablename__ = "graph_node"
    __table_args__ = (UniqueConstraint("canonical", "graph_version", name="uq_node_canonical_ver"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical: Mapped[str] = mapped_column(String(256), nullable=False)
    synset: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    node_type: Mapped[str] = mapped_column(String(32), default="concept")  # concept|part|alias_group
    aliases: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    graph_version: Mapped[str] = mapped_column(String(64), default="kg-v0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")


class GraphEdge(Base):
    __tablename__ = "graph_edge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    src_id: Mapped[int] = mapped_column(ForeignKey("graph_node.id"), nullable=False, index=True)
    dst_id: Mapped[int] = mapped_column(ForeignKey("graph_node.id"), nullable=False, index=True)
    relation: Mapped[str] = mapped_column(String(64), nullable=False)  # hypernym|hyponym|part_of|alias_of
    graph_version: Mapped[str] = mapped_column(String(64), default="kg-v0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")


class ReleaseManifest(Base):
    __tablename__ = "release_manifest"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    release_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    counts: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    provenance: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    data_version: Mapped[str] = mapped_column(String(64), default="m1-mvp-0.1")
    valid_status: Mapped[str] = mapped_column(String(32), default="valid")
