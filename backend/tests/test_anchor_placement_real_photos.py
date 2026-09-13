"""Regression test: forehead/cheek anchors must land on real skin, measured
numerically against real photos rather than eyeballed screenshots.

This is what replaces the last three rounds of "looks right in this crop" -
each anchor's distance to the TRUE face boundary (the face oval, interpolated
at that exact anchor's height, not a single global bounding-box figure) is
computed and asserted against a fixed numeric range.

Requires local, non-committed photo fixtures at
backend/tests/fixtures/photos/ (see .gitignore - these are real people's
photos, some possibly copyrighted, and never get committed) and the
MediaPipe model file from backend/scripts/download_models.sh. Skips cleanly
when either is missing, so a fresh clone or CI run stays green; run this
locally with the fixtures present to actually verify anchor geometry.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from mediapipe import Image, ImageFormat

from app.vision.skin_sampling import (
    _FACE_OVAL,
    _LEFT_EYEBROW,
    _MODEL_PATH,
    _RIGHT_EYEBROW,
    _centroid,
    _connection_indices,
    _face_bbox_area,
    _landmarker,
    compute_anchor_points,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "photos"
PHOTO_PATHS = sorted(FIXTURES_DIR.glob("*.jp*g")) if FIXTURES_DIR.exists() else []

# Fraction of the hairline-to-brow gap the forehead anchor sits at (0 = at
# the hairline, 1 = at the brow line). Measured 0.08-0.35 across the fixture
# corpus; the pre-fix ratio measured -0.75 (above the hairline, into hair).
_FOREHEAD_FRACTION_BOUNDS = (0.05, 0.55)
# Fraction of the local half-width each cheek anchor sits at, measured from
# the true face-oval boundary interpolated AT THAT ANCHOR'S HEIGHT (0 =
# center, 1.0 = exactly at the visible edge of the face). Measured 0.37-0.78
# across the fixture corpus; the ratio this replaces measured 0.74-1.28
# (regularly past the edge of the face, i.e. at/past the jaw or ear).
_CHEEK_FRACTION_BOUNDS = (0.25, 0.90)

pytestmark = pytest.mark.skipif(
    not PHOTO_PATHS or not _MODEL_PATH.exists(),
    reason=(
        "Local photo fixtures or the MediaPipe model aren't available - see "
        "backend/tests/fixtures/photos/ and backend/scripts/download_models.sh"
    ),
)


def _detect_largest_face(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
    result = _landmarker().detect(mp_image)
    height, width = image_rgb.shape[:2]
    faces = [
        np.array([(landmark.x * width, landmark.y * height) for landmark in landmarks])
        for landmarks in result.face_landmarks
    ]
    return max(faces, key=_face_bbox_area)


def _oval_boundary_x_at_y(points: np.ndarray, y: float) -> tuple[float, float] | None:
    """Interpolate the face-oval's left/right boundary x at a given height.

    Walks the oval's own landmark connections (not just their bounding box)
    so the boundary reflects the face's actual width at that specific
    height, which can differ a lot from its widest point (e.g. cheekbone
    height vs. jaw height). Returns None if no segment straddles that
    height at all.
    """
    oval_indices = _connection_indices(_FACE_OVAL)
    center_x = points[oval_indices][:, 0].mean()
    left_xs: list[float] = []
    right_xs: list[float] = []
    for connection in _FACE_OVAL:
        p1, p2 = points[connection.start], points[connection.end]
        y1, y2 = p1[1], p2[1]
        if (y1 - y) * (y2 - y) > 0 or y1 == y2:
            continue  # segment doesn't straddle this height
        t = (y - y1) / (y2 - y1)
        x = p1[0] + t * (p2[0] - p1[0])
        (left_xs if x < center_x else right_xs).append(x)
    if not left_xs or not right_xs:
        return None
    return min(left_xs), max(right_xs)


def _forehead_fraction_from_hairline(points: np.ndarray, forehead: np.ndarray) -> float:
    oval_indices = _connection_indices(_FACE_OVAL)
    hairline_y = points[oval_indices][:, 1].min()
    brow_center = _centroid(
        points, _connection_indices(_LEFT_EYEBROW) + _connection_indices(_RIGHT_EYEBROW)
    )
    gap = brow_center[1] - hairline_y
    return (forehead[1] - hairline_y) / gap


def _cheek_fraction_from_boundary(points: np.ndarray, cheek: np.ndarray) -> float:
    boundary = _oval_boundary_x_at_y(points, cheek[1])
    assert boundary is not None, "cheek anchor's height falls outside the traced face oval"
    left_x, right_x = boundary
    half_width = (right_x - left_x) / 2
    center_x = (right_x + left_x) / 2
    return abs(cheek[0] - center_x) / half_width


@pytest.mark.parametrize("photo_path", PHOTO_PATHS, ids=lambda p: p.name)
def test_forehead_anchor_sits_between_hairline_and_brows(photo_path):
    points = _detect_largest_face(photo_path)
    anchors = compute_anchor_points(points)
    fraction = _forehead_fraction_from_hairline(points, anchors.forehead)
    low, high = _FOREHEAD_FRACTION_BOUNDS
    assert low <= fraction <= high, (
        f"forehead anchor at {fraction:.2f} of the hairline-to-brow gap, expected within [{low}, {high}]"
    )


@pytest.mark.parametrize("photo_path", PHOTO_PATHS, ids=lambda p: p.name)
def test_cheek_anchors_land_inside_face_boundary(photo_path):
    points = _detect_largest_face(photo_path)
    anchors = compute_anchor_points(points)
    low, high = _CHEEK_FRACTION_BOUNDS
    for label, cheek in [("left_cheek", anchors.left_cheek), ("right_cheek", anchors.right_cheek)]:
        fraction = _cheek_fraction_from_boundary(points, cheek)
        assert low <= fraction <= high, (
            f"{label} anchor at {fraction:.2f} of the local half-width from center "
            f"(1.0 = face edge), expected within [{low}, {high}]"
        )
