"""Parity check: resubmitting a photo through the drag-the-boxes recovery
flow (POST /api/scan/manual) with the boxes left exactly where the automatic
scan placed them must produce the same season as the automatic scan
(POST /api/scan).

Both endpoints sample the same patches, so any disagreement comes from
classification, not sampling. Before the manual path normalized against the
sclera (whites of the eyes) the way the automatic path does (see
season_classifier.normalize_depth_lightness / normalize_undertone_ab), 4 of
these fixtures - Sample1, Sample4, Sample8, SampleLightWhite - got a
different season through the manual path, purely because it classified raw
skin color while the automatic path corrected for the photo's own lighting.

Goes through the real HTTP endpoints (TestClient), not the functions behind
them, so the router's own wiring - form anchors -> sampling -> sclera ->
classify_season - is what's being checked. Persistence and the AI paragraph
are stubbed out by conftest.py's autouse fixtures, and the rate limiter is
reset per test, so each parametrized case's two requests can't trip it.

Requires local, non-committed photo fixtures at backend/tests/fixtures/photos/
(and its lighting/ subdirectory - see .gitignore) and the MediaPipe model file
from backend/scripts/download_models.sh. Skips cleanly when either is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.vision.image_decode import decode_image
from app.vision.skin_sampling import _MODEL_PATH, sample_skin_regions

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "photos"
PHOTO_PATHS = (
    sorted(FIXTURES_DIR.glob("*.jp*g")) + sorted((FIXTURES_DIR / "lighting").glob("*.jp*g"))
    if FIXTURES_DIR.exists()
    else []
)

pytestmark = pytest.mark.skipif(
    not PHOTO_PATHS or not _MODEL_PATH.exists(),
    reason=(
        "Local photo fixtures or the MediaPipe model aren't available - see "
        "backend/tests/fixtures/photos/ and backend/scripts/download_models.sh"
    ),
)

client = TestClient(app)


@pytest.mark.parametrize("photo_path", PHOTO_PATHS, ids=lambda p: p.stem)
def test_manual_scan_with_untouched_boxes_matches_automatic_scan(photo_path: Path):
    photo_bytes = photo_path.read_bytes()

    automatic = client.post("/api/scan", files={"photo": (photo_path.name, photo_bytes, "image/jpeg")})
    if automatic.status_code != 200:
        pytest.skip(f"automatic scan rejected this photo ({automatic.status_code}), nothing to compare against")

    anchors = sample_skin_regions(decode_image(photo_bytes)).anchors
    manual = client.post(
        "/api/scan/manual",
        files={"photo": (photo_path.name, photo_bytes, "image/jpeg")},
        data={
            "forehead_x": str(anchors.forehead[0]),
            "forehead_y": str(anchors.forehead[1]),
            "left_cheek_x": str(anchors.left_cheek[0]),
            "left_cheek_y": str(anchors.left_cheek[1]),
            "right_cheek_x": str(anchors.right_cheek[0]),
            "right_cheek_y": str(anchors.right_cheek[1]),
            "patch_half_size": str(anchors.patch_half_size),
        },
    )

    assert manual.status_code == 200, manual.json()
    assert manual.json()["season"] == automatic.json()["season"]
