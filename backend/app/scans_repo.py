"""Scan persistence — reliable, not best-effort.

Contrast with app/analytics.py's log_event: that one is deliberately
fire-and-forget because a scan must succeed even if analytics logging
fails. This module is the opposite on purpose — a scan that can't be
persisted can never legitimately be sold, so every failure here (a missing
DATABASE_URL, an unreachable Postgres, a bad query) propagates to the
caller instead of being swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from app.config import get_settings
from app.vision.season_classifier import Season

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scans (
    id UUID PRIMARY KEY,
    season TEXT NOT NULL,
    swatches JSONB NOT NULL,
    paragraph TEXT,
    paid BOOLEAN NOT NULL DEFAULT FALSE,
    stripe_checkout_session_id TEXT,
    stripe_payment_intent_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    paid_at TIMESTAMPTZ
)
"""

_schema_ready = False


@dataclass(frozen=True)
class ScanRow:
    id: str
    season: Season
    swatches: list[dict[str, Any]]
    paragraph: str | None
    paid: bool
    stripe_checkout_session_id: str | None
    stripe_payment_intent_id: str | None
    full_report_paragraph: str | None


def _connect() -> psycopg.Connection:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured; scan persistence requires a database.")
    # prepare_threshold=None disables psycopg3's automatic server-side
    # prepared statements. The configured DATABASE_URL may point at
    # Supabase's PgBouncer pooler in transaction mode, which multiplexes
    # each transaction across different physical backend connections —
    # a prepared statement created on one backend won't exist on the next,
    # so re-preparing eventually raises "prepared statement ... does not
    # exist". Each call here is a short-lived, mostly single-statement
    # connection anyway, so there's no real benefit to preparing, only risk.
    return psycopg.connect(settings.database_url, connect_timeout=3, prepare_threshold=None)


def _ensure_schema(conn: psycopg.Connection) -> None:
    global _schema_ready
    if not _schema_ready:
        conn.execute(_CREATE_TABLE_SQL)
        # No migration tool (e.g. Alembic) exists in this repo — schema
        # changes to an already-deployed table go here as idempotent ALTERs,
        # each its own conn.execute() call since psycopg3 doesn't reliably
        # run multiple statements passed to a single execute().
        conn.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS full_report_paragraph TEXT")
        _schema_ready = True


def create_scan(scan_id: str, season: Season, swatches: list[dict[str, Any]], paragraph: str | None) -> None:
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute(
            "INSERT INTO scans (id, season, swatches, paragraph) VALUES (%s, %s, %s, %s)",
            (scan_id, season, Jsonb(swatches), paragraph),
        )


def get_scan(scan_id: str) -> ScanRow | None:
    with _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(
            """
            SELECT id, season, swatches, paragraph, paid,
                   stripe_checkout_session_id, stripe_payment_intent_id,
                   full_report_paragraph
            FROM scans WHERE id = %s
            """,
            (scan_id,),
        ).fetchone()

    if row is None:
        return None
    return ScanRow(
        id=str(row[0]),
        season=row[1],
        swatches=row[2],
        paragraph=row[3],
        paid=row[4],
        stripe_checkout_session_id=row[5],
        stripe_payment_intent_id=row[6],
        full_report_paragraph=row[7],
    )


def set_checkout_session(scan_id: str, session_id: str) -> None:
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute(
            "UPDATE scans SET stripe_checkout_session_id = %s WHERE id = %s",
            (session_id, scan_id),
        )


def set_full_report_paragraph(scan_id: str, paragraph: str) -> None:
    """Cache a lazily-generated paid-tier AI paragraph so subsequent reads
    of the same scan don't pay for another Anthropic call."""
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute(
            "UPDATE scans SET full_report_paragraph = %s WHERE id = %s",
            (paragraph, scan_id),
        )


def mark_scan_paid(scan_id: str, payment_intent_id: str | None) -> None:
    """Idempotent: safe to call more than once for the same scan (a
    redelivered webhook, or the webhook and the GET /api/scans/{id}
    fallback-verify both confirming the same payment)."""
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            UPDATE scans
            SET paid = TRUE,
                stripe_payment_intent_id = COALESCE(%s, stripe_payment_intent_id),
                paid_at = COALESCE(paid_at, now())
            WHERE id = %s
            """,
            (payment_intent_id, scan_id),
        )
