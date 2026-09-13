import numpy as np
import pytest

from app.vision.skin_sampling import (
    compute_anchor_points,
    patch_clipped_fraction,
    sample_patch_rgb,
    sample_skin_regions,
)


def _synthetic_landmarks() -> np.ndarray:
    """468 landmark points shaped roughly like an upright, centered face.

    Only the indices used by compute_anchor_points (eyebrows, eyes, face
    oval) need to be anatomically sane; the rest are filler so array
    indexing by real mediapipe index values doesn't go out of bounds.
    """
    points = np.full((468, 2), 300.0)

    # Face oval: a ring of points, chin (lowest) at y=500.
    from app.vision.skin_sampling import _FACE_OVAL, _LEFT_EYE, _LEFT_EYEBROW, _RIGHT_EYE, _RIGHT_EYEBROW, _connection_indices

    oval_indices = _connection_indices(_FACE_OVAL)
    for i, idx in enumerate(oval_indices):
        angle = 2 * np.pi * i / len(oval_indices)
        points[idx] = (300 + 100 * np.sin(angle), 400 + 100 * np.cos(angle))

    left_eyebrow_indices = _connection_indices(_LEFT_EYEBROW)
    right_eyebrow_indices = _connection_indices(_RIGHT_EYEBROW)
    for idx in left_eyebrow_indices:
        points[idx] = (260, 280)
    for idx in right_eyebrow_indices:
        points[idx] = (340, 280)

    left_eye_indices = _connection_indices(_LEFT_EYE)
    right_eye_indices = _connection_indices(_RIGHT_EYE)
    for idx in left_eye_indices:
        points[idx] = (260, 300)
    for idx in right_eye_indices:
        points[idx] = (340, 300)

    return points


def test_compute_anchor_points_places_forehead_above_brows_between_eyes():
    points = _synthetic_landmarks()

    anchors = compute_anchor_points(points)

    # Forehead should sit above (lower y than) the brow line, roughly
    # centered between the eyes horizontally.
    assert anchors.forehead[1] < 280
    assert 250 < anchors.forehead[0] < 350

    # Left/right cheeks should sit below the eyes and lateral to them.
    assert anchors.left_cheek[1] > 300
    assert anchors.left_cheek[0] < 260
    assert anchors.right_cheek[1] > 300
    assert anchors.right_cheek[0] > 340

    assert anchors.patch_half_size > 0


def test_sample_patch_rgb_rejects_shadow_and_highlight_outliers():
    patch_size = 20
    image_rgb = np.full((patch_size, patch_size, 3), 180, dtype=np.uint8)
    image_lab = np.full((patch_size, patch_size, 3), 150, dtype=np.uint8)

    # Inject a dark shadow corner and a blown-out highlight corner.
    image_rgb[:4, :4] = 20
    image_lab[:4, :4, 0] = 10
    image_rgb[-4:, -4:] = 255
    image_lab[-4:, -4:, 0] = 250

    center = np.array([patch_size / 2, patch_size / 2])
    result = sample_patch_rgb(image_rgb, image_lab, center, half_size=patch_size / 2)

    assert result == (180, 180, 180)


def test_sample_patch_rgb_returns_none_when_patch_is_out_of_frame():
    image_rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    image_lab = np.zeros((50, 50, 3), dtype=np.uint8)

    center = np.array([-100, -100])
    result = sample_patch_rgb(image_rgb, image_lab, center, half_size=10)

    assert result is None


def test_patch_clipped_fraction_flags_majority_clipped_patch():
    patch_size = 20
    image_rgb = np.full((patch_size, patch_size, 3), 180, dtype=np.uint8)
    # Blow out half the patch to a clipped highlight.
    image_rgb[:, : patch_size // 2] = 255

    center = np.array([patch_size / 2, patch_size / 2])
    fraction = patch_clipped_fraction(image_rgb, center, half_size=patch_size / 2)

    assert fraction == pytest.approx(0.5)


def test_patch_clipped_fraction_ignores_single_channel_near_ceiling():
    # A warm/bright but genuinely valid skin tone: red channel near the
    # ceiling, green/blue clearly lower - real color, not a blown highlight
    # (which would desaturate toward white across all three channels).
    patch_size = 20
    image_rgb = np.empty((patch_size, patch_size, 3), dtype=np.uint8)
    image_rgb[..., 0] = 253
    image_rgb[..., 1] = 204
    image_rgb[..., 2] = 172

    center = np.array([patch_size / 2, patch_size / 2])
    fraction = patch_clipped_fraction(image_rgb, center, half_size=patch_size / 2)

    assert fraction == 0.0


def test_patch_clipped_fraction_is_zero_for_normal_patch():
    patch_size = 20
    image_rgb = np.full((patch_size, patch_size, 3), 180, dtype=np.uint8)

    center = np.array([patch_size / 2, patch_size / 2])
    fraction = patch_clipped_fraction(image_rgb, center, half_size=patch_size / 2)

    assert fraction == 0.0


def test_patch_clipped_fraction_returns_none_when_patch_is_out_of_frame():
    image_rgb = np.zeros((50, 50, 3), dtype=np.uint8)

    center = np.array([-100, -100])
    fraction = patch_clipped_fraction(image_rgb, center, half_size=10)

    assert fraction is None


def test_sample_skin_regions_reports_no_face_detected_for_blank_image():
    blank_image = np.full((480, 640, 3), 200, dtype=np.uint8)

    result = sample_skin_regions(blank_image)

    assert result.success is False
    assert result.error == "no_face_detected"
