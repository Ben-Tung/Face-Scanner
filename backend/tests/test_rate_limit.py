"""Tests for the per-IP rate limiter on POST /api/scan and /api/scan/manual.

Follows test_retake.py's style: TestClient against the real app, with
monkeypatch stubbing out the vision pipeline directly on the router module
— no real face detection, no real Anthropic/DB calls.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import client_ip as client_ip_module
from app import rate_limit as rate_limit_module
from app.client_ip import get_client_ip
from app.config import Settings
from app.main import app
from app.routers import scan as scan_module
from app.vision.season_classifier import SeasonClassification, SeasonClassificationResult
from app.vision.skin_sampling import AnchorPoints, SkinSampleResult

client = TestClient(app)

_SCAN_ID = "22222222-2222-2222-2222-222222222222"

_ANCHORS = AnchorPoints(
    forehead=np.array([100.0, 80.0]),
    left_cheek=np.array([60.0, 200.0]),
    right_cheek=np.array([180.0, 200.0]),
    patch_half_size=20.0,
)


@pytest.fixture(autouse=True)
def _reset_ipware_cache():
    client_ip_module._ipware.cache_clear()
    yield
    client_ip_module._ipware.cache_clear()


def _blank_jpeg_bytes() -> bytes:
    blank_image = np.full((480, 640, 3), 200, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank_image)
    assert ok
    return encoded.tobytes()


def _stub_successful_pipeline(monkeypatch) -> None:
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
        season="Winter",
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


def _tiny_limit_settings(per_minute: int = 2, per_day: int = 100) -> Settings:
    return Settings(scan_rate_limit_per_minute=per_minute, scan_rate_limit_per_day=per_day)


def _post_scan(headers: dict[str, str] | None = None):
    return client.post(
        "/api/scan",
        files={"photo": ("selfie.jpg", _blank_jpeg_bytes(), "image/jpeg")},
        headers=headers,
    )


def _post_scan_manual(headers: dict[str, str] | None = None):
    return client.post(
        "/api/scan/manual",
        files={"photo": ("selfie.jpg", _blank_jpeg_bytes(), "image/jpeg")},
        data={
            "forehead_x": 100.0,
            "forehead_y": 80.0,
            "left_cheek_x": 60.0,
            "left_cheek_y": 200.0,
            "right_cheek_x": 180.0,
            "right_cheek_y": 200.0,
            "patch_half_size": 20.0,
        },
        headers=headers,
    )


def test_scan_returns_429_after_exceeding_per_minute_limit(monkeypatch):
    monkeypatch.setattr(rate_limit_module, "get_settings", lambda: _tiny_limit_settings())
    _stub_successful_pipeline(monkeypatch)

    first = _post_scan()
    second = _post_scan()
    third = _post_scan()

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429


def test_scan_429_response_includes_retry_after_header(monkeypatch):
    monkeypatch.setattr(rate_limit_module, "get_settings", lambda: _tiny_limit_settings())
    _stub_successful_pipeline(monkeypatch)

    _post_scan()
    _post_scan()
    response = _post_scan()

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0


def test_scan_and_scan_manual_have_independent_rate_limit_counters(monkeypatch):
    """Design choice: /scan and /scan/manual are limited per-route (slowapi's
    default key_style="url"), not via one shared bucket — exhausting one
    endpoint's counter must not affect the other's."""
    monkeypatch.setattr(rate_limit_module, "get_settings", lambda: _tiny_limit_settings())
    _stub_successful_pipeline(monkeypatch)

    assert _post_scan().status_code == 200
    assert _post_scan().status_code == 200
    assert _post_scan().status_code == 429

    assert _post_scan_manual().status_code == 200
    assert _post_scan_manual().status_code == 200
    assert _post_scan_manual().status_code == 429


def test_retake_is_not_rate_limited(monkeypatch):
    from app.scans_repo import ScanRow

    row = ScanRow(
        id=_SCAN_ID,
        season="Autumn",
        swatches=[{"name": "Terracotta", "hex": "#C1652E"}],
        paragraph="You're an Autumn.",
        paid=True,
        stripe_checkout_session_id="cs_test_existing",
        stripe_payment_intent_id="pi_existing",
        full_report_paragraph=None,
        retake_used=False,
    )
    monkeypatch.setattr(scan_module, "get_scan", lambda scan_id: row)
    monkeypatch.setattr(scan_module, "consume_retake", lambda *a, **k: True)
    _stub_successful_pipeline(monkeypatch)

    for _ in range(8):
        response = client.post(
            f"/api/scans/{_SCAN_ID}/retake",
            files={"photo": ("selfie.jpg", _blank_jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200


def test_rate_limit_key_is_per_ip_not_global(monkeypatch):
    monkeypatch.setattr(rate_limit_module, "get_settings", lambda: _tiny_limit_settings(per_minute=2))
    _stub_successful_pipeline(monkeypatch)

    for ip in ("203.0.113.10", "203.0.113.20"):
        headers = {"X-Forwarded-For": ip}
        assert _post_scan(headers=headers).status_code == 200
        assert _post_scan(headers=headers).status_code == 200


def _make_request(xff: str | None, client_host: str = "10.0.0.5") -> Request:
    headers = []
    if xff is not None:
        headers.append((b"x-forwarded-for", xff.encode()))
    scope = {"type": "http", "headers": headers, "client": (client_host, 12345)}
    return Request(scope)


def test_get_client_ip_trusts_rightmost_xff_entry_by_default():
    request = _make_request("9.9.9.9, 8.8.8.8, 203.0.113.5")

    assert get_client_ip(request) == "203.0.113.5"


def test_get_client_ip_falls_back_to_request_client_host_when_no_xff():
    request = _make_request(None, client_host="203.0.113.9")

    assert get_client_ip(request) == "203.0.113.9"


def test_get_client_ip_respects_trust_leftmost_setting(monkeypatch):
    monkeypatch.setattr(client_ip_module, "get_settings", lambda: Settings(client_ip_trust_leftmost=True))
    client_ip_module._ipware.cache_clear()

    request = _make_request("9.9.9.9, 8.8.8.8, 203.0.113.5")

    assert get_client_ip(request) == "9.9.9.9"
