"""Training feedback / revision helpers (§2.7) — skeleton for later rounds."""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session

from wde.schema.models import ErrorRecord, Revision


def record_error(
    session: Session,
    *,
    failure_type: str,
    detail: str | None = None,
    image_id: int | None = None,
    query_id: int | None = None,
    model_version: str | None = None,
    run_id: str | None = None,
    data_version: str = "m1-mvp-0.1",
) -> ErrorRecord:
    rec = ErrorRecord(
        failure_type=failure_type,
        detail=detail,
        image_id=image_id,
        query_id=query_id,
        model_version=model_version,
        run_id=run_id,
        data_version=data_version,
    )
    session.add(rec)
    session.flush()
    return rec


def add_revision(
    session: Session,
    *,
    target_table: str,
    target_id: int,
    action: str,
    before: Optional[dict[str, Any]] = None,
    after: Optional[dict[str, Any]] = None,
    reviewer: str | None = None,
    note: str | None = None,
    run_id: str | None = None,
    data_version: str = "m1-mvp-0.1",
) -> Revision:
    rev = Revision(
        target_table=target_table,
        target_id=target_id,
        action=action,
        before=before,
        after=after,
        reviewer=reviewer,
        note=note,
        run_id=run_id,
        data_version=data_version,
    )
    session.add(rev)
    session.flush()
    return rev
