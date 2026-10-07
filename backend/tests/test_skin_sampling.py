import numpy as np
import pytest

from app.vision import skin_sampling
from app.vision.skin_sampling import (
    AnchorPoints,
    ScleraSampleResult,
    compute_anchor_points,
    patch_clipped_fraction,
    sample_at_anchors,
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

    # Face oval: an ellipse taller than it is wide, chin (lowest) at y=500.
    # Deliberately not a circle - a circle's extent is the same in every
    # direction, so it can't catch geometry that's only right for an
    # upright head (see test_compute_anchor_points_is_equivariant_under_head_roll).
    from app.vision.skin_sampling import _FACE_OVAL, _LEFT_EYE, _LEFT_EYEBROW, _RIGHT_EYE, _RIGHT_EYEBROW, _connection_indices

    oval_indices = _connection_indices(_FACE_OVAL)
    for i, idx in enumerate(oval_indices):
        angle = 2 * np.pi * i / len(oval_indices)
        points[idx] = (300 + 80 * np.sin(angle), 400 + 100 * np.cos(angle))

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


@pytest.mark.parametrize("roll_deg", [-30, 15, 30])
def test_compute_anchor_points_is_equivariant_under_head_roll(roll_deg: float):
    # A tilted head should move the sample points WITH the face, not
    # relative to it: anchors computed from rotated landmarks must equal the
    # upright anchors rotated the same way. Measuring the face against the
    # image axes instead (e.g. its x-extent for width, its lowest point for
    # the chin) breaks this, and on real faces pushed the cheek samples from
    # 0.55 of the half-width out to 0.74 at 35 degrees of roll.
    points = _synthetic_landmarks()
    eye_center = np.array([300.0, 300.0])
    theta = np.radians(roll_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def rotate(p: np.ndarray) -> np.ndarray:
        return (p - eye_center) @ rotation.T + eye_center

    upright = compute_anchor_points(points)
    rolled = compute_anchor_points(rotate(points))

    for name in ("forehead", "left_cheek", "right_cheek"):
        np.testing.assert_allclose(getattr(rolled, name), rotate(getattr(upright, name)), atol=0.5, err_msg=name)
    assert rolled.patch_half_size == pytest.approx(upright.patch_half_size)


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


def test_sample_skin_regions_with_anchors_samples_even_without_a_face():
    # Manual-adjustment flow: the user's dragged boxes are still sampled when
    # no face is found - there's just no sclera to normalize against, so
    # classification falls back to uncorrected color.
    blank_image = np.full((480, 640, 3), 200, dtype=np.uint8)
    anchors = AnchorPoints(
        forehead=np.array([320.0, 100.0]),
        left_cheek=np.array([250.0, 300.0]),
        right_cheek=np.array([390.0, 300.0]),
        patch_half_size=20.0,
    )

    result = sample_skin_regions(blank_image, anchors=anchors)

    assert result.success is True
    assert result.forehead_rgb == (200, 200, 200)
    assert result.anchors is anchors
    assert result.sclera is None


def test_sample_skin_regions_with_anchors_samples_there_and_reads_sclera_from_detected_face(monkeypatch):
    # Manual-adjustment flow with a detectable face: patches come from the
    # caller's anchors, not the detected face's own computed ones, while the
    # sclera is still read from that face - the same lighting reference the
    # automatic path gets.
    image_bgr = np.full((600, 600, 3), 150, dtype=np.uint8)
    face = _synthetic_landmarks()
    anchors = AnchorPoints(
        forehead=np.array([100.0, 100.0]),
        left_cheek=np.array([80.0, 200.0]),
        right_cheek=np.array([120.0, 200.0]),
        patch_half_size=10.0,
    )
    sclera = ScleraSampleResult(success=True, sclera_rgb=(230, 225, 220))
    sclera_calls = []

    monkeypatch.setattr(skin_sampling, "_detect_largest_face", lambda image_rgb: face)

    def _fake_sample_sclera(image_rgb, image_lab, points):
        sclera_calls.append(points)
        return sclera

    monkeypatch.setattr(skin_sampling, "sample_sclera", _fake_sample_sclera)

    result = sample_skin_regions(image_bgr, anchors=anchors)

    assert result.success is True
    assert result.anchors is anchors
    assert result.sclera is sclera
    assert len(sclera_calls) == 1 and sclera_calls[0] is face


def test_sample_at_anchors_returns_success_for_well_lit_patches():
    image_bgr = np.full((60, 60, 3), 150, dtype=np.uint8)
    anchors = AnchorPoints(
        forehead=np.array([30.0, 15.0]),
        left_cheek=np.array([15.0, 40.0]),
        right_cheek=np.array([45.0, 40.0]),
        patch_half_size=8.0,
    )

    result = sample_at_anchors(image_bgr, anchors)

    assert result.success is True
    assert result.forehead_rgb == (150, 150, 150)
    assert result.left_cheek_rgb == (150, 150, 150)
    assert result.right_cheek_rgb == (150, 150, 150)
    assert result.anchors is anchors


def test_sample_at_anchors_reports_patch_clipped():
    image_bgr = np.full((60, 60, 3), 150, dtype=np.uint8)
    # Blow out the entire forehead patch to a clipped highlight.
    image_bgr[7:24, 22:39] = 255
    anchors = AnchorPoints(
        forehead=np.array([30.0, 15.0]),
        left_cheek=np.array([15.0, 40.0]),
        right_cheek=np.array([45.0, 40.0]),
        patch_half_size=8.0,
    )

    result = sample_at_anchors(image_bgr, anchors)

    assert result.success is False
    assert result.error == "patch_clipped"
    assert result.anchors is anchors


_SEPARATE_PATCH_ANCHORS = AnchorPoints(
    forehead=np.array([30.0, 15.0]),
    left_cheek=np.array([15.0, 40.0]),
    right_cheek=np.array([45.0, 40.0]),
    patch_half_size=8.0,
)
# Each anchor's patch region (y, x slices) in a 60x60 image - no overlaps.
_PATCH_REGIONS = (
    (slice(7, 23), slice(22, 38)),  # forehead
    (slice(32, 48), slice(7, 23)),  # left cheek
    (slice(32, 48), slice(37, 53)),  # right cheek
)


def _skin_image_with_red_at(red: int, patch_count: int) -> np.ndarray:
    """Light skin (RGB 200, 160, 140), with the red channel of the first
    `patch_count` patches raised to `red` and green/blue left alone."""
    image_bgr = np.full((60, 60, 3), (140, 160, 200), dtype=np.uint8)
    for rows, cols in _PATCH_REGIONS[:patch_count]:
        image_bgr[rows, cols, 2] = red
    return image_bgr


@pytest.mark.parametrize(("red", "expected_success"), [(252, False), (249, True)])
def test_sample_at_anchors_rejects_a_sampled_channel_at_the_ceiling(red, expected_success):
    # Bright, light skin with only red near 255 on one patch: no
    # all-channel-clipped pixels for patch_clipped_fraction to notice, but a
    # red channel pinned at the ceiling compresses R:G and reads the hue
    # warmer than it is.
    image_bgr = _skin_image_with_red_at(red, patch_count=1)
    anchors = _SEPARATE_PATCH_ANCHORS

    result = sample_at_anchors(image_bgr, anchors)

    assert patch_clipped_fraction(image_bgr[..., ::-1], anchors.forehead, anchors.patch_half_size) == 0.0
    assert result.success is expected_success
    if not expected_success:
        assert result.error == "patch_clipped"


@pytest.mark.parametrize(
    ("saturated_patches", "expected_error"),
    [(1, "patch_clipped"), (2, "overexposed"), (3, "overexposed")],
)
def test_sample_at_anchors_reports_overexposed_only_when_two_patches_are_blown(saturated_patches, expected_error):
    # One fully saturated patch is a highlight the user can drag a box off
    # of; two or more means the face itself is blown out, which only a
    # retake fixes.
    image_bgr = _skin_image_with_red_at(253, patch_count=saturated_patches)

    result = sample_at_anchors(image_bgr, _SEPARATE_PATCH_ANCHORS)

    assert result.success is False
    assert result.error == expected_error
    assert result.anchors is _SEPARATE_PATCH_ANCHORS


def test_patch_saturated_fraction_counts_pixels_with_any_channel_at_the_ceiling():
    image_rgb = np.full((60, 60, 3), (200, 160, 140), dtype=np.uint8)
    rows, cols = _PATCH_REGIONS[0]
    image_rgb[rows, 22:30, 0] = 252  # left half of the forehead patch: red only

    fraction = skin_sampling.patch_saturated_fraction(
        image_rgb, _SEPARATE_PATCH_ANCHORS.forehead, _SEPARATE_PATCH_ANCHORS.patch_half_size
    )

    assert fraction == pytest.approx(0.5)
    assert patch_clipped_fraction(
        image_rgb, _SEPARATE_PATCH_ANCHORS.forehead, _SEPARATE_PATCH_ANCHORS.patch_half_size
    ) == 0.0


def test_sample_at_anchors_reports_face_out_of_frame_for_anchor_outside_image():
    image_bgr = np.full((60, 60, 3), 150, dtype=np.uint8)
    anchors = AnchorPoints(
        forehead=np.array([-500.0, -500.0]),
        left_cheek=np.array([15.0, 40.0]),
        right_cheek=np.array([45.0, 40.0]),
        patch_half_size=8.0,
    )

    result = sample_at_anchors(image_bgr, anchors)

    assert result.success is False
    assert result.error == "face_out_of_frame"
    assert result.anchors is anchors
