"""Best-effort event logging to Postgres — see CLAUDE.md's analytics convention.

Fire-and-forget: a scan (or any other feature) must succeed even if Postgres
is unreachable or the driver isn't installed, e.g. when running the backend
natively without `docker compose up`. Every failure here is caught and
logged rather than raised.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_schema_ready = False


def log_event(event_type: str, metadata: dict[str, Any] | None = None) -> None:
    settings = get_settings()
    if not settings.database_url:
        return

    global _schema_ready
    try:
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
            if not _schema_ready:
                conn.execute(_CREATE_TABLE_SQL)
                _schema_ready = True
            conn.execute(
                "INSERT INTO events (event_type, metadata) VALUES (%s, %s)",
                (event_type, Jsonb(metadata) if metadata is not None else None),
            )
    except Exception:
        logger.warning("Failed to log analytics event %r", event_type, exc_info=True)
