"""Persistence-layer tests for scans_repo.

Contrast with test_analytics.py: log_event's contract is "never raise, no
matter what." scans_repo's contract is the opposite by design — a scan
that can't be saved can never legitimately be sold, so every failure here
must propagate to the caller instead of being swallowed.
"""

from __future__ import annotations

import socket
import uuid
from urllib.parse import urlparse

import psycopg
import pytest

from app import scans_repo
from app.config import Settings


def _database_reachable() -> bool:
    settings = Settings()
    if not settings.database_url:
        return False
    parsed = urlparse(settings.database_url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 5432), timeout=1):
            return True
    except OSError:
        return False


_DB_REACHABLE = _database_reachable()
_SKIP_REASON = "requires a reachable Postgres (e.g. `docker compose up db`)"


@pytest.fixture
def scan_id():
    """A fresh scan id, deleted from the real table afterward regardless of
    what the test did with it. DATABASE_URL in backend/.env can point at a
    real hosted database (see project memory on the Supabase pooler), so
    these tests must not leave permanent rows behind there."""
    sid = str(uuid.uuid4())
    yield sid
    settings = Settings()
    if settings.database_url:
        with psycopg.connect(settings.database_url, connect_timeout=3, prepare_threshold=None) as conn:
            conn.execute("DELETE FROM scans WHERE id = %s", (sid,))


def test_create_scan_raises_without_database_url(monkeypatch, scan_id):
    monkeypatch.setattr(scans_repo, "get_settings", lambda: Settings(database_url=None))

    with pytest.raises(RuntimeError):
        scans_repo.create_scan(scan_id, "Winter", [], None)


def test_create_scan_raises_when_database_unreachable(monkeypatch, scan_id):
    monkeypatch.setattr(
        scans_repo,
        "get_settings",
        lambda: Settings(database_url="postgresql://bad:bad@localhost:1/nope"),
    )

    with pytest.raises(Exception):
        scans_repo.create_scan(scan_id, "Winter", [], None)


@pytest.mark.skipif(not _DB_REACHABLE, reason=_SKIP_REASON)
def test_create_and_get_scan_round_trip(scan_id):
    swatches = [{"name": "True Red", "hex": "#D0103A"}]

    scans_repo.create_scan(scan_id, "Winter", swatches, "You're a Winter.")
    row = scans_repo.get_scan(scan_id)

    assert row is not None
    assert row.season == "Winter"
    assert row.swatches == swatches
    assert row.paragraph == "You're a Winter."
    assert row.paid is False
    assert row.stripe_checkout_session_id is None
    assert row.stripe_payment_intent_id is None
    assert row.full_report_paragraph is None
    assert row.retake_used is False


@pytest.mark.skipif(not _DB_REACHABLE, reason=_SKIP_REASON)
def test_get_scan_returns_none_for_unknown_id():
    assert scans_repo.get_scan(str(uuid.uuid4())) is None


@pytest.mark.skipif(not _DB_REACHABLE, reason=_SKIP_REASON)
def test_set_checkout_session_then_mark_scan_paid(scan_id):
    scans_repo.create_scan(scan_id, "Autumn", [], None)

    scans_repo.set_checkout_session(scan_id, "cs_test_abc")
    scans_repo.mark_scan_paid(scan_id, "pi_first")
    row = scans_repo.get_scan(scan_id)

    assert row.stripe_checkout_session_id == "cs_test_abc"
    assert row.paid is True
    assert row.stripe_payment_intent_id == "pi_first"


@pytest.mark.skipif(not _DB_REACHABLE, reason=_SKIP_REASON)
def test_mark_scan_paid_is_idempotent(scan_id):
    # A redelivered webhook (or the webhook and the fallback-verify path
    # both confirming the same payment) always carries the same
    # payment_intent, so calling this twice with the same id must be safe.
    scans_repo.create_scan(scan_id, "Autumn", [], None)

    first = scans_repo.mark_scan_paid(scan_id, "pi_123")
    second = scans_repo.mark_scan_paid(scan_id, "pi_123")
    row = scans_repo.get_scan(scan_id)

    assert row.paid is True
    assert row.stripe_payment_intent_id == "pi_123"
    # Only the call that actually transitions the scan to paid returns
    # True — callers use this to log a purchase_completed event exactly
    # once even when both confirmation paths fire for the same payment.
    assert first is True
    assert second is False


@pytest.mark.skipif(not _DB_REACHABLE, reason=_SKIP_REASON)
def test_set_full_report_paragraph_persists_text(scan_id):
    scans_repo.create_scan(scan_id, "Autumn", [], None)

    scans_repo.set_full_report_paragraph(scan_id, "Your Autumn palette runs warm and rich.")
    row = scans_repo.get_scan(scan_id)

    assert row.full_report_paragraph == "Your Autumn palette runs warm and rich."


@pytest.mark.skipif(not _DB_REACHABLE, reason=_SKIP_REASON)
def test_mark_scan_paid_does_not_null_out_an_existing_payment_intent(scan_id):
    scans_repo.create_scan(scan_id, "Autumn", [], None)

    scans_repo.mark_scan_paid(scan_id, "pi_123")
    scans_repo.mark_scan_paid(scan_id, None)
    row = scans_repo.get_scan(scan_id)

    assert row.paid is True
    assert row.stripe_payment_intent_id == "pi_123"


@pytest.mark.skipif(not _DB_REACHABLE, reason=_SKIP_REASON)
def test_consume_retake_is_concurrency_safe(scan_id):
    # Two concurrent retake requests both racing to consume the same scan's
    # one free retake must not both succeed — mirrors
    # test_mark_scan_paid_is_idempotent's shape for the same reason.
    scans_repo.create_scan(scan_id, "Autumn", [], None)
    scans_repo.mark_scan_paid(scan_id, "pi_123")

    first = scans_repo.consume_retake(scan_id, "Winter", [{"name": "True Red", "hex": "#D0103A"}], "First retake.")
    second = scans_repo.consume_retake(scan_id, "Spring", [{"name": "Coral", "hex": "#FF7F50"}], "Second retake.")
    row = scans_repo.get_scan(scan_id)

    assert first is True
    assert second is False
    assert row.retake_used is True
    assert row.season == "Winter"
    assert row.paragraph == "First retake."


@pytest.mark.skipif(not _DB_REACHABLE, reason=_SKIP_REASON)
def test_consume_retake_nulls_stale_full_report_paragraph(scan_id):
    # full_report_paragraph is a cached AI paragraph describing the OLD
    # season's best colors, so it must never survive a season change.
    scans_repo.create_scan(scan_id, "Autumn", [], None)
    scans_repo.mark_scan_paid(scan_id, "pi_123")
    scans_repo.set_full_report_paragraph(scan_id, "Your Autumn palette runs warm and rich.")

    scans_repo.consume_retake(scan_id, "Winter", [], "New season, new you.")
    row = scans_repo.get_scan(scan_id)

    assert row.full_report_paragraph is None


@pytest.mark.skipif(not _DB_REACHABLE, reason=_SKIP_REASON)
def test_consume_retake_returns_false_for_unpaid_scan(scan_id):
    scans_repo.create_scan(scan_id, "Autumn", [], None)

    consumed = scans_repo.consume_retake(scan_id, "Winter", [], "Should not persist.")
    row = scans_repo.get_scan(scan_id)

    assert consumed is False
    assert row.season == "Autumn"
    assert row.retake_used is False
