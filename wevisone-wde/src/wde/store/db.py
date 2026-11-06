"""SQLite store helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from wde.schema.models import Base


def make_engine(db_path: str | Path) -> Engine:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{path.resolve()}"
    engine = create_engine(url, echo=False, future=True)

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def init_db(db_path: str | Path) -> Engine:
    """Create all tables (migrate/init). Idempotent."""
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine: Engine) -> Session:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)()


def table_counts(engine: Engine) -> dict[str, int]:
    counts: dict[str, int] = {}
    with Session(engine) as session:
        for table in Base.metadata.sorted_tables:
            n = session.execute(text(f"SELECT COUNT(*) FROM {table.name}")).scalar_one()
            counts[table.name] = int(n)
    return counts
