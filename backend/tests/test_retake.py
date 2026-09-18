"""Tests for the one-time free retake on a paid scan.

Follows test_payments.py's style: TestClient against the real app, with
monkeypatch stubbing out scans_repo functions and the vision pipeline
directly on the router module — no real database, no real face detection.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import payments as payments_module
from app.routers import scan as scan_module
from app.scans_repo import ScanRow
from app.vision.season_classifier import SeasonClassification, SeasonClassificationResult
from app.vision.skin_sampling import AnchorPoints, SkinSampleResult

client = TestClient(app)

_SCAN_ID = "11111111-1111-1111-1111-111111111111"

_ANCHORS = AnchorPoints(
    forehead=np.array([100.0, 80.0]),
    left_cheek=np.array([60.0, 200.0]),
    right_cheek=np.array([180.0, 200.0]),
    patch_half_size=20.0,
)


def _blank_jpeg_bytes() -> bytes:
    blank_image = np.full((480, 640, 3), 200, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank_image)
    assert ok
    return encoded.tobytes()


def _paid_row(retake_used: bool = False) -> ScanRow:
    return ScanRow(
        id=_SCAN_ID,
        season="Autumn",
        swatches=[{"name": "Terracotta", "hex": "#C1652E"}],
        paragraph="You're an Autumn.",
        paid=True,
        stripe_checkout_session_id="cs_test_existing",
        stripe_payment_intent_id="pi_existing",
        full_report_paragraph=None,
        retake_used=retake_used,
    )


def _unpaid_row() -> ScanRow:
    return dataclasses.replace(_paid_row(), paid=False, stripe_payment_intent_id=None)


def _stub_successful_pipeline(monkeypatch, season: str = "Winter") -> None:
    monkeypatch.setattr(
        scan_module,
        "sample_skin_regions",
        lambda image_bgr: SkinSampleResult(
            success=True,
            forehead_rgb=(210, 180, 170),
            left_cheek_rgb=(210, 180, 170),
            right_cheek_rgb=(210, 180, 170),
            anchors=_ANCHORS,
        ),
    )
    classification = SeasonClassification(
        season=season,
        undertone="cool",
        depth="deep",
        clarity="clear",
        avg_lab=(50.0, 10.0, 5.0),
        hue_deg=20.0,
        chroma=15.0,
        depth_lightness=50.0,
    )
    monkeypatch.setattr(
        scan_module,
        "classify_season",
        lambda forehead_rgb, left_cheek_rgb, right_cheek_rgb, sclera_rgb=None: SeasonClassificationResult(
            success=True, classification=classification
        ),
    )


def _post_retake(photo_bytes: bytes = b""):
    return client.post(
        f"/api/scans/{_SCAN_ID}/retake",
        files={"photo": ("selfie.jpg", photo_bytes or _blank_jpeg_bytes(), "image/jpeg")},
    )


def test_retake_succeeds_and_replaces_scan_result(monkeypatch):
    monkeypatch.setattr(scan_module, "get_scan", lambda scan_id: _paid_row())
    _stub_successful_pipeline(monkeypatch, season="Winter")

    captured = {}

    def _fake_consume_retake(scan_id, season, swatches, paragraph):
        captured.update(scan_id=scan_id, season=season, swatches=swatches, paragraph=paragraph)
        return True

    monkeypatch.setattr(scan_module, "consume_retake", _fake_consume_retake)
    logged = []
    monkeypatch.setattr(scan_module, "log_event", lambda *a, **k: logged.append(a))

    response = _post_retake()

    assert response.status_code == 200
    body = response.json()
    assert body["scan_id"] == _SCAN_ID
    assert body["season"] == "Winter"
    assert 4 <= len(body["swatches"]) <= 5
    assert captured["scan_id"] == _SCAN_ID
    assert captured["season"] == "Winter"
    assert logged[0][0] == "retake_started"


def test_retake_rejects_second_attempt_on_same_scan(monkeypatch):
    monkeypatch.setattr(scan_module, "get_scan", lambda scan_id: _paid_row(retake_used=True))
    monkeypatch.setattr(
        scan_module, "sample_skin_regions", lambda image_bgr: pytest.fail("must not run the pipeline")
    )
    monkeypatch.setattr(
        scan_module, "consume_retake", lambda *a, **k: pytest.fail("must not consume the retake")
    )

    response = _post_retake()

    assert response.status_code == 409


def test_retake_rejects_unpaid_scan(monkeypatch):
    monkeypatch.setattr(scan_module, "get_scan", lambda scan_id: _unpaid_row())
    monkeypatch.setattr(
        scan_module, "sample_skin_regions", lambda image_bgr: pytest.fail("must not run the pipeline")
    )
    monkeypatch.setattr(
        scan_module, "consume_retake", lambda *a, **k: pytest.fail("must not consume the retake")
    )

    response = _post_retake()

    assert response.status_code == 403


def test_retake_404s_for_unknown_scan(monkeypatch):
    monkeypatch.setattr(scan_module, "get_scan", lambda scan_id: None)

    response = _post_retake()

    assert response.status_code == 404


def test_retake_race_returns_409_when_consume_retake_loses(monkeypatch):
    """Two requests can both pass the upfront paid/retake_used checks and
    both run the (slow) pipeline — the real safety guarantee is
    consume_retake's atomic conditional UPDATE, re-checked fresh afterward.
    Simulates the loser of that race: consume_retake returns False even
    though the initial read said the retake was still available."""
    monkeypatch.setattr(scan_module, "get_scan", lambda scan_id: _paid_row())
    _stub_successful_pipeline(monkeypatch)
    monkeypatch.setattr(scan_module, "consume_retake", lambda *a, **k: False)

    response = _post_retake()

    assert response.status_code == 409


def test_retake_low_confidence_does_not_consume_the_retake(monkeypatch):
    monkeypatch.setattr(scan_module, "get_scan", lambda scan_id: _paid_row())
    monkeypatch.setattr(
        scan_module,
        "sample_skin_regions",
        lambda image_bgr: SkinSampleResult(success=False, error="no_face_detected"),
    )
    monkeypatch.setattr(
        scan_module, "consume_retake", lambda *a, **k: pytest.fail("must not consume the retake")
    )

    response = _post_retake()

    assert response.status_code == 422


def test_retake_503s_when_persisting_fails(monkeypatch):
    monkeypatch.setattr(scan_module, "get_scan", lambda scan_id: _paid_row())
    _stub_successful_pipeline(monkeypatch)

    def _raise(*args, **kwargs):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(scan_module, "consume_retake", _raise)

    response = _post_retake()

    assert response.status_code == 503


def test_retake_failure_leaves_existing_result_visible_afterward(monkeypatch):
    """A failed retake attempt must never disturb the customer's existing,
    already-working result. consume_retake is the only function anywhere
    that writes season/swatches/paragraph to a scan row, and it's never
    called on this path — this test proves that end to end by fetching the
    scan again (via the separate payments router) after the failed retake
    and confirming nothing changed."""
    row = _paid_row()
    monkeypatch.setattr(scan_module, "get_scan", lambda scan_id: row)
    monkeypatch.setattr(payments_module, "get_scan", lambda scan_id: row)
    monkeypatch.setattr(payments_module, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(
        scan_module,
        "sample_skin_regions",
        lambda image_bgr: SkinSampleResult(success=False, error="no_face_detected"),
    )
    monkeypatch.setattr(
        scan_module, "consume_retake", lambda *a, **k: pytest.fail("must not consume the retake")
    )

    retake_response = _post_retake()
    assert retake_response.status_code == 422

    get_response = client.get(f"/api/scans/{_SCAN_ID}")
    assert get_response.status_code == 200
    body = get_response.json()
    assert body["season"] == row.season
    assert body["swatches"] == row.swatches
    assert body["paragraph"] == row.paragraph
    assert body["paid"] is True
    assert body["retake_used"] is False
