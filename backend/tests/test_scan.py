from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import scan as scan_module
from app.vision.season_classifier import SeasonClassificationResult
from app.vision.skin_sampling import AnchorPoints, ScleraSampleResult, SkinSampleResult

client = TestClient(app)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "photos"
PHOTO_PATHS = sorted(FIXTURES_DIR.glob("*.jp*g")) if FIXTURES_DIR.exists() else []


def _blank_jpeg_bytes() -> bytes:
    blank_image = np.full((480, 640, 3), 200, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank_image)
    assert ok
    return encoded.tobytes()


def test_scan_rejects_non_image_upload():
    response = client.post(
        "/api/scan",
        files={"photo": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 422


def test_scan_reports_no_face_detected_for_blank_photo():
    response = client.post(
        "/api/scan",
        files={"photo": ("blank.jpg", _blank_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422
    assert "face" in response.json()["detail"].lower()


@pytest.mark.skipif(not PHOTO_PATHS, reason="requires local photo fixtures at tests/fixtures/photos/")
@pytest.mark.parametrize("photo_path", PHOTO_PATHS, ids=lambda p: p.name)
def test_scan_returns_season_and_swatches_for_real_photos(photo_path: Path):
    with photo_path.open("rb") as f:
        response = client.post(
            "/api/scan",
            files={"photo": (photo_path.name, f, "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["scan_id"]
    assert body["season"] in {"Spring", "Summer", "Autumn", "Winter"}
    assert 4 <= len(body["swatches"]) <= 5
    for swatch in body["swatches"]:
        assert swatch["name"]
        assert swatch["hex"].startswith("#")


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def _solid_jpeg_bytes(rgb: tuple[int, int, int], size: int = 60) -> bytes:
    image_bgr = np.full((size, size, 3), _bgr(rgb), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    assert ok
    return encoded.tobytes()


def _striped_jpeg_bytes(
    forehead_rgb: tuple[int, int, int],
    left_cheek_rgb: tuple[int, int, int],
    right_cheek_rgb: tuple[int, int, int],
    stripe_width: int = 100,
    height: int = 100,
) -> bytes:
    """Three solid-color vertical stripes in one image, wide enough that a
    patch centered in each stripe never touches the neighboring stripe."""
    image_bgr = np.zeros((height, stripe_width * 3, 3), dtype=np.uint8)
    image_bgr[:, 0:stripe_width] = _bgr(forehead_rgb)
    image_bgr[:, stripe_width : 2 * stripe_width] = _bgr(left_cheek_rgb)
    image_bgr[:, 2 * stripe_width : 3 * stripe_width] = _bgr(right_cheek_rgb)
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    assert ok
    return encoded.tobytes()


def _manual_form_fields(
    forehead: tuple[float, float],
    left_cheek: tuple[float, float],
    right_cheek: tuple[float, float],
    patch_half_size: float,
) -> dict[str, str]:
    return {
        "forehead_x": str(forehead[0]),
        "forehead_y": str(forehead[1]),
        "left_cheek_x": str(left_cheek[0]),
        "left_cheek_y": str(left_cheek[1]),
        "right_cheek_x": str(right_cheek[0]),
        "right_cheek_y": str(right_cheek[1]),
        "patch_half_size": str(patch_half_size),
    }


def test_scan_manual_returns_season_and_swatches_for_well_lit_patches():
    photo_bytes = _solid_jpeg_bytes((216, 165, 152), size=60)
    data = _manual_form_fields((30, 15), (15, 40), (45, 40), 8)

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("patch.jpg", photo_bytes, "image/jpeg")},
        data=data,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scan_id"]
    assert body["season"] in {"Spring", "Summer", "Autumn", "Winter"}
    assert 4 <= len(body["swatches"]) <= 5
    for swatch in body["swatches"]:
        assert swatch["name"]
        assert swatch["hex"].startswith("#")


def test_scan_manual_samples_at_dragged_anchors_and_normalizes_against_sclera(monkeypatch):
    # The manual path must get the same sclera lighting correction /scan
    # does - otherwise the same boxes on the same photo can land on a
    # different season depending on which endpoint produced them.
    captured = {}
    sclera_rgb = (230, 225, 220)

    def _fake_sample_skin_regions(image_bgr, anchors=None):
        captured["anchors"] = anchors
        return SkinSampleResult(
            success=True,
            forehead_rgb=(216, 165, 152),
            left_cheek_rgb=(208, 158, 145),
            right_cheek_rgb=(222, 170, 158),
            anchors=anchors,
            sclera=ScleraSampleResult(success=True, sclera_rgb=sclera_rgb),
        )

    real_classify_season = scan_module.classify_season

    def _spy_classify_season(*args, **kwargs):
        captured["sclera_rgb"] = kwargs.get("sclera_rgb")
        return real_classify_season(*args, **kwargs)

    monkeypatch.setattr(scan_module, "sample_skin_regions", _fake_sample_skin_regions)
    monkeypatch.setattr(scan_module, "classify_season", _spy_classify_season)

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("patch.jpg", _solid_jpeg_bytes((216, 165, 152), size=60), "image/jpeg")},
        data=_manual_form_fields((30, 15), (15, 40), (45, 40), 8),
    )

    assert response.status_code == 200
    assert captured["sclera_rgb"] == sclera_rgb
    anchors = captured["anchors"]
    assert anchors.forehead.tolist() == [30.0, 15.0]
    assert anchors.left_cheek.tolist() == [15.0, 40.0]
    assert anchors.right_cheek.tolist() == [45.0, 40.0]
    assert anchors.patch_half_size == 8.0


def test_scan_manual_reports_patch_clipped():
    image_bgr = np.full((60, 60, 3), _bgr((200, 150, 130)), dtype=np.uint8)
    image_bgr[7:24, 22:39] = 255  # blow out the forehead patch region
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    assert ok

    data = _manual_form_fields((30, 15), (15, 40), (45, 40), 8)
    response = client.post(
        "/api/scan/manual",
        files={"photo": ("clipped.jpg", encoded.tobytes(), "image/jpeg")},
        data=data,
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason"] == "patch_clipped"
    assert detail["patches"]["forehead"] == {"x": 30.0, "y": 15.0}
    assert detail["image"] == {"width": 60, "height": 60}


def test_scan_manual_reports_inconsistent_patches():
    # Same three RGBs as test_season_classifier's inconsistent-patches case
    # (max pairwise ΔH* ~16.0), laid out as three stripes in one image.
    photo_bytes = _striped_jpeg_bytes(
        forehead_rgb=(123, 102, 99),
        left_cheek_rgb=(193, 185, 144),
        right_cheek_rgb=(255, 247, 227),
    )
    data = _manual_form_fields((50, 50), (150, 50), (250, 50), 20)

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("stripes.jpg", photo_bytes, "image/jpeg")},
        data=data,
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason"] == "inconsistent_patches"
    assert detail["image"] == {"width": 300, "height": 100}


def test_scan_manual_rejects_out_of_bounds_coordinates():
    photo_bytes = _solid_jpeg_bytes((216, 165, 152), size=60)
    data = _manual_form_fields((30, 15), (15, 40), (999, 40), 8)

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("patch.jpg", photo_bytes, "image/jpeg")},
        data=data,
    )

    assert response.status_code == 400


def test_scan_manual_rejects_non_positive_patch_half_size():
    photo_bytes = _solid_jpeg_bytes((216, 165, 152), size=60)
    data = _manual_form_fields((30, 15), (15, 40), (45, 40), 0)

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("patch.jpg", photo_bytes, "image/jpeg")},
        data=data,
    )

    assert response.status_code == 400


@pytest.mark.parametrize("patch_half_size", ["nan", "inf", "-inf"])
def test_scan_manual_rejects_non_finite_patch_half_size(patch_half_size: str):
    # NaN slips past a plain `<= 0` check (every comparison with NaN is
    # False), and both NaN and inf used to crash patch sampling with a 500.
    photo_bytes = _solid_jpeg_bytes((216, 165, 152), size=60)
    data = {**_manual_form_fields((30, 15), (15, 40), (45, 40), 8), "patch_half_size": patch_half_size}

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("patch.jpg", photo_bytes, "image/jpeg")},
        data=data,
    )

    assert response.status_code == 400


def test_scan_manual_rejects_non_finite_coordinate():
    photo_bytes = _solid_jpeg_bytes((216, 165, 152), size=60)
    data = {**_manual_form_fields((30, 15), (15, 40), (45, 40), 8), "forehead_x": "nan"}

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("patch.jpg", photo_bytes, "image/jpeg")},
        data=data,
    )

    assert response.status_code == 400


def test_scan_manual_rejects_oversized_patch_half_size():
    # A huge patch would median the whole photo, background included, into
    # a "skin" reading.
    photo_bytes = _solid_jpeg_bytes((216, 165, 152), size=60)
    data = _manual_form_fields((30, 15), (15, 40), (45, 40), 1e7)

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("patch.jpg", photo_bytes, "image/jpeg")},
        data=data,
    )

    assert response.status_code == 400


def test_scan_manual_rejects_non_image_upload():
    data = _manual_form_fields((30, 15), (15, 40), (45, 40), 8)

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("notes.txt", b"hello", "text/plain")},
        data=data,
    )

    assert response.status_code == 422


def test_scan_reports_structured_detail_for_patch_clipped(monkeypatch):
    anchors = AnchorPoints(
        forehead=np.array([100.0, 80.0]),
        left_cheek=np.array([60.0, 200.0]),
        right_cheek=np.array([180.0, 200.0]),
        patch_half_size=20.0,
    )
    monkeypatch.setattr(
        scan_module,
        "sample_skin_regions",
        lambda image_bgr: SkinSampleResult(success=False, error="patch_clipped", anchors=anchors),
    )

    response = client.post(
        "/api/scan",
        files={"photo": ("blank.jpg", _blank_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason"] == "patch_clipped"
    assert detail["patches"]["forehead"] == {"x": 100.0, "y": 80.0}
    assert detail["patches"]["patch_half_size"] == 20.0
    assert detail["image"] == {"width": 640, "height": 480}


def test_scan_reports_structured_detail_for_face_out_of_frame(monkeypatch):
    anchors = AnchorPoints(
        forehead=np.array([-5.0, 80.0]),
        left_cheek=np.array([60.0, 200.0]),
        right_cheek=np.array([180.0, 200.0]),
        patch_half_size=20.0,
    )
    monkeypatch.setattr(
        scan_module,
        "sample_skin_regions",
        lambda image_bgr: SkinSampleResult(success=False, error="face_out_of_frame", anchors=anchors),
    )

    response = client.post(
        "/api/scan",
        files={"photo": ("blank.jpg", _blank_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason"] == "face_out_of_frame"
    assert detail["patches"]["forehead"] == {"x": -5.0, "y": 80.0}
    assert detail["patches"]["patch_half_size"] == 20.0
    assert detail["image"] == {"width": 640, "height": 480}


def test_scan_does_not_block_other_requests_while_waiting_on_paragraph(monkeypatch):
    """A scan waiting on its (blocking, multi-second) Anthropic call must
    not stall every other request on the server. Used as a context manager,
    TestClient runs all requests on one shared event loop - the same shape
    as the single uvicorn process in production - so a handler that blocks
    the loop holds up the health check below until the paragraph returns."""
    monkeypatch.setattr(
        scan_module,
        "sample_skin_regions",
        lambda image_bgr, anchors=None: SkinSampleResult(
            success=True,
            forehead_rgb=(216, 165, 152),
            left_cheek_rgb=(208, 158, 145),
            right_cheek_rgb=(222, 170, 158),
            anchors=AnchorPoints(
                forehead=np.array([30.0, 15.0]),
                left_cheek=np.array([15.0, 40.0]),
                right_cheek=np.array([45.0, 40.0]),
                patch_half_size=8.0,
            ),
        ),
    )
    paragraph_entered = threading.Event()
    release_paragraph = threading.Event()

    def _slow_paragraph(*args, **kwargs):
        paragraph_entered.set()
        release_paragraph.wait(timeout=5)
        return None

    monkeypatch.setattr(scan_module, "generate_paragraph", _slow_paragraph)

    with TestClient(app) as shared_client:
        scan_response = {}

        def _scan():
            scan_response["value"] = shared_client.post(
                "/api/scan", files={"photo": ("p.jpg", _solid_jpeg_bytes((216, 165, 152)), "image/jpeg")}
            )

        scan_thread = threading.Thread(target=_scan)
        scan_thread.start()
        try:
            assert paragraph_entered.wait(timeout=5), "scan never reached the paragraph call"
            started = time.monotonic()
            health = shared_client.get("/api/health")
            health_seconds = time.monotonic() - started
        finally:
            release_paragraph.set()
            scan_thread.join(timeout=10)

    assert health.status_code == 200
    assert health_seconds < 1.0, f"health check waited {health_seconds:.2f}s behind an in-flight scan"
    assert scan_response["value"].status_code == 200


def test_scan_returns_503_when_persistence_fails(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("DATABASE_URL is not configured; scan persistence requires a database.")

    monkeypatch.setattr(scan_module, "create_scan", _raise)

    response = client.post(
        "/api/scan/manual",
        files={"photo": ("patch.jpg", _solid_jpeg_bytes((216, 165, 152), size=60), "image/jpeg")},
        data=_manual_form_fields((30, 15), (15, 40), (45, 40), 8),
    )

    assert response.status_code == 503


def test_scan_reports_structured_detail_for_inconsistent_patches(monkeypatch):
    anchors = AnchorPoints(
        forehead=np.array([100.0, 80.0]),
        left_cheek=np.array([60.0, 200.0]),
        right_cheek=np.array([180.0, 200.0]),
        patch_half_size=20.0,
    )
    monkeypatch.setattr(
        scan_module,
        "sample_skin_regions",
        lambda image_bgr: SkinSampleResult(
            success=True,
            forehead_rgb=(123, 102, 99),
            left_cheek_rgb=(193, 185, 144),
            right_cheek_rgb=(255, 247, 227),
            anchors=anchors,
        ),
    )
    monkeypatch.setattr(
        scan_module,
        "classify_season",
        lambda forehead_rgb, left_cheek_rgb, right_cheek_rgb, sclera_rgb=None: SeasonClassificationResult(
            success=False, error="inconsistent_patches"
        ),
    )

    response = client.post(
        "/api/scan",
        files={"photo": ("blank.jpg", _blank_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason"] == "inconsistent_patches"
    assert detail["image"] == {"width": 640, "height": 480}
