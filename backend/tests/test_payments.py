"""Router tests for Stripe Checkout creation, scan-state lookup (including
its fallback payment verification), and the webhook handler.

Follows test_scan.py's style: TestClient against the real app, with
monkeypatch used to stub out the scans_repo persistence layer and the
Stripe SDK itself — no real database, no real Stripe API calls.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import stripe
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.routers import payments as payments_module
from app.scans_repo import ScanRow

client = TestClient(app)

_SCAN_ID = "11111111-1111-1111-1111-111111111111"
_OTHER_SCAN_ID = "22222222-2222-2222-2222-222222222222"


def _fake_session(**attrs) -> SimpleNamespace:
    """A stand-in for stripe's real Session object. Real Stripe SDK objects
    (as of stripe-python 15.x) deliberately raise on dict-style .get() —
    "... is not a dict, use .to_dict()" — and only support attribute
    access; SimpleNamespace matches that (and, unlike a plain dict, would
    itself fail loudly if production code regressed to calling .get() on
    it, the same bug this file's fixtures used to mask)."""
    return SimpleNamespace(**attrs)


def _fake_event(event_type: str, session: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, data=SimpleNamespace(object=session))


def _unpaid_row(scan_id: str = _SCAN_ID) -> ScanRow:
    return ScanRow(
        id=scan_id,
        season="Winter",
        swatches=[{"name": "True Red", "hex": "#D0103A"}],
        paragraph="You're a Winter.",
        paid=False,
        stripe_checkout_session_id=None,
        stripe_payment_intent_id=None,
    )


def _paid_row(scan_id: str = _SCAN_ID) -> ScanRow:
    return dataclasses.replace(_unpaid_row(scan_id), paid=True, stripe_payment_intent_id="pi_existing")


def _configured_settings(**overrides) -> Settings:
    return Settings(stripe_secret_key="sk_test_fake", stripe_webhook_secret="whsec_fake", **overrides)


# --- POST /api/scans/{scan_id}/checkout -------------------------------------


def test_checkout_builds_fixed_price_and_scan_metadata(monkeypatch):
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: _unpaid_row())
    monkeypatch.setattr(payments_module, "set_checkout_session", lambda scan_id, session_id: None)
    monkeypatch.setattr(payments_module, "get_settings", _configured_settings)

    captured = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_session(id="cs_test_123", url="https://checkout.stripe.com/pay/cs_test_123")

    monkeypatch.setattr(payments_module.stripe.checkout.Session, "create", lambda **kw: _fake_create(**kw))

    response = client.post(f"/api/scans/{_SCAN_ID}/checkout")

    assert response.status_code == 200
    assert response.json() == {"checkout_url": "https://checkout.stripe.com/pay/cs_test_123"}
    assert captured["metadata"] == {"scan_id": _SCAN_ID}
    line_item = captured["line_items"][0]
    assert line_item["price_data"]["unit_amount"] == 299
    assert line_item["price_data"]["currency"] == "usd"


def test_checkout_404s_for_unknown_scan(monkeypatch):
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: None)

    response = client.post(f"/api/scans/{_SCAN_ID}/checkout")

    assert response.status_code == 404


def test_checkout_409s_when_already_paid(monkeypatch):
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: _paid_row())

    response = client.post(f"/api/scans/{_SCAN_ID}/checkout")

    assert response.status_code == 409


def test_checkout_500s_when_stripe_not_configured(monkeypatch):
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: _unpaid_row())
    monkeypatch.setattr(payments_module, "get_settings", lambda: Settings(stripe_secret_key=None))

    response = client.post(f"/api/scans/{_SCAN_ID}/checkout")

    assert response.status_code == 500


def test_checkout_502s_on_stripe_error(monkeypatch):
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: _unpaid_row())
    monkeypatch.setattr(payments_module, "get_settings", _configured_settings)

    def _raise(**kwargs):
        raise stripe.error.StripeError("boom")

    monkeypatch.setattr(payments_module.stripe.checkout.Session, "create", lambda **kw: _raise(**kw))

    response = client.post(f"/api/scans/{_SCAN_ID}/checkout")

    assert response.status_code == 502


# --- GET /api/scans/{scan_id} ------------------------------------------------


def test_get_scan_state_404s_for_unknown_scan(monkeypatch):
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: None)

    response = client.get(f"/api/scans/{_SCAN_ID}")

    assert response.status_code == 404


def test_get_scan_state_unpaid_has_no_full_report(monkeypatch):
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: _unpaid_row())
    logged = []
    monkeypatch.setattr(payments_module, "log_event", lambda *a, **k: logged.append((a, k)))

    response = client.get(f"/api/scans/{_SCAN_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["paid"] is False
    assert body["full_report"] is None
    assert logged and logged[0][0][0] == "paywall_viewed"


def test_get_scan_state_paid_returns_full_report_stub(monkeypatch):
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: _paid_row())
    monkeypatch.setattr(payments_module, "log_event", lambda *a, **k: pytest.fail("should not log paywall_viewed"))

    response = client.get(f"/api/scans/{_SCAN_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["paid"] is True
    assert body["full_report"]["swatches"] == body["swatches"]
    assert body["full_report"]["note"]


def test_get_scan_state_self_heals_on_matching_session(monkeypatch):
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: _unpaid_row())
    monkeypatch.setattr(payments_module, "get_settings", _configured_settings)
    monkeypatch.setattr(payments_module, "log_event", lambda *a, **k: None)

    marked = []
    monkeypatch.setattr(payments_module, "mark_scan_paid", lambda scan_id, pi: marked.append((scan_id, pi)))
    monkeypatch.setattr(
        payments_module.stripe.checkout.Session,
        "retrieve",
        lambda session_id, **kw: _fake_session(
            payment_status="paid",
            metadata=SimpleNamespace(scan_id=_SCAN_ID),
            payment_intent="pi_new",
        ),
    )

    response = client.get(f"/api/scans/{_SCAN_ID}", params={"session_id": "cs_test_456"})

    assert response.status_code == 200
    assert response.json()["paid"] is True
    assert marked == [(_SCAN_ID, "pi_new")]


def test_get_scan_state_ignores_session_for_a_different_scan(monkeypatch):
    """Security case: a session_id that paid for a *different* scan must
    not unlock this one."""
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: _unpaid_row())
    monkeypatch.setattr(payments_module, "get_settings", _configured_settings)
    monkeypatch.setattr(payments_module, "log_event", lambda *a, **k: None)

    monkeypatch.setattr(
        payments_module, "mark_scan_paid", lambda *a, **k: pytest.fail("must not mark paid")
    )
    monkeypatch.setattr(
        payments_module.stripe.checkout.Session,
        "retrieve",
        lambda session_id, **kw: _fake_session(
            payment_status="paid",
            metadata=SimpleNamespace(scan_id=_OTHER_SCAN_ID),
            payment_intent="pi_new",
        ),
    )

    response = client.get(f"/api/scans/{_SCAN_ID}", params={"session_id": "cs_test_789"})

    assert response.status_code == 200
    assert response.json()["paid"] is False


# --- POST /api/stripe/webhook ------------------------------------------------


def test_webhook_rejects_bad_signature(monkeypatch):
    monkeypatch.setattr(payments_module, "get_settings", _configured_settings)

    def _raise(*a, **k):
        raise stripe.error.SignatureVerificationError("bad sig", "sig_header")

    monkeypatch.setattr(payments_module.stripe.Webhook, "construct_event", _raise)

    response = client.post(
        "/api/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "bogus"}
    )

    assert response.status_code == 400


def test_webhook_marks_paid_and_logs_purchase_completed(monkeypatch):
    monkeypatch.setattr(payments_module, "get_settings", _configured_settings)

    fake_event = _fake_event(
        "checkout.session.completed",
        _fake_session(metadata=SimpleNamespace(scan_id=_SCAN_ID), payment_intent="pi_webhook", id="cs_test_evt"),
    )
    monkeypatch.setattr(payments_module.stripe.Webhook, "construct_event", lambda *a, **k: fake_event)

    marked = []
    logged = []
    monkeypatch.setattr(payments_module, "mark_scan_paid", lambda scan_id, pi: marked.append((scan_id, pi)))
    monkeypatch.setattr(payments_module, "log_event", lambda *a, **k: logged.append(a))

    response = client.post(
        "/api/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "valid"}
    )

    assert response.status_code == 200
    assert marked == [(_SCAN_ID, "pi_webhook")]
    assert logged and logged[0][0] == "purchase_completed"


def test_webhook_ignores_unrecognized_event_types(monkeypatch):
    monkeypatch.setattr(payments_module, "get_settings", _configured_settings)
    fake_event = _fake_event("payment_intent.created", _fake_session())
    monkeypatch.setattr(payments_module.stripe.Webhook, "construct_event", lambda *a, **k: fake_event)
    monkeypatch.setattr(
        payments_module, "mark_scan_paid", lambda *a, **k: pytest.fail("must not be called")
    )

    response = client.post(
        "/api/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "valid"}
    )

    assert response.status_code == 200


def test_webhook_surfaces_failure_when_mark_scan_paid_raises(monkeypatch):
    """Lets Stripe retry the delivery instead of silently losing a payment
    confirmation if the database is unreachable.

    Uses a client with raise_server_exceptions=False: the default
    TestClient re-raises an unhandled exception into the test process
    (useful for catching bugs elsewhere), but here the whole point is to
    check what a real ASGI server would hand back to Stripe — a 500, not a
    crash — so this test needs the same behavior production gets.
    """
    monkeypatch.setattr(payments_module, "get_settings", _configured_settings)
    fake_event = _fake_event(
        "checkout.session.completed",
        _fake_session(metadata=SimpleNamespace(scan_id=_SCAN_ID), payment_intent="pi_webhook", id="cs_test_evt"),
    )
    monkeypatch.setattr(payments_module.stripe.Webhook, "construct_event", lambda *a, **k: fake_event)

    def _raise(scan_id, pi):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(payments_module, "mark_scan_paid", _raise)

    lenient_client = TestClient(app, raise_server_exceptions=False)
    response = lenient_client.post(
        "/api/stripe/webhook", content=b"{}", headers={"Stripe-Signature": "valid"}
    )

    assert response.status_code == 500
