from collections import namedtuple

import cv2
import numpy as np
import pytest

from app.vision.skin_sampling import (
    _LEFT_EYE,
    _LEFT_IRIS,
    _RIGHT_EYE,
    _RIGHT_IRIS,
    _connection_indices,
    _ordered_loop,
    compute_eye_geometry,
    sample_sclera,
    sclera_clipped_fraction,
)

_Edge = namedtuple("_Edge", ["start", "end"])


def test_ordered_loop_walks_full_simple_cycle():
    # A synthetic 4-node square loop, deliberately listed out of walk order -
    # _ordered_loop must reconstruct the actual boundary order regardless.
    edges = [_Edge(0, 1), _Edge(2, 3), _Edge(1, 2), _Edge(3, 0)]

    loop = _ordered_loop(edges)

    assert sorted(loop) == [0, 1, 2, 3]
    wrapped = loop + [loop[0]]
    edge_set = {frozenset((e.start, e.end)) for e in edges}
    for a, b in zip(wrapped, wrapped[1:]):
        assert frozenset((a, b)) in edge_set


def test_ordered_loop_walks_real_eye_connections():
    # The real FACE_LANDMARKS_LEFT_EYE connection set - confirms the helper
    # works against actual MediaPipe data, not just a hand-built toy graph.
    loop = _ordered_loop(_LEFT_EYE)

    assert sorted(loop) == _connection_indices(_LEFT_EYE)
    assert len(loop) == 16


def _place_eye(points, loop, iris_indices, center, half_width, half_height, iris_radius):
    cx, cy = center
    n = len(loop)
    for i, idx in enumerate(loop):
        angle = 2 * np.pi * i / n
        points[idx] = (cx + half_width * np.cos(angle), cy + half_height * np.sin(angle))
    ordered_iris = sorted(iris_indices)
    n_iris = len(ordered_iris)
    for i, idx in enumerate(ordered_iris):
        angle = 2 * np.pi * i / n_iris
        points[idx] = (cx + iris_radius * np.cos(angle), cy + iris_radius * np.sin(angle))


def _synthetic_eye_landmarks(
    left_geometry=(60.0, 25.0, 15.0),
    right_geometry=(60.0, 25.0, 15.0),
    left_center=(150.0, 150.0),
    right_center=(350.0, 150.0),
) -> np.ndarray:
    """478-point landmark array with only the eye/iris indices placed.

    Each eye is an ellipse (half_width, half_height) with an iris circle
    (iris_radius) at its center, built from the real connection sets so
    tests exercise the actual landmark indices/ordering, not stand-ins.
    Everything else stays at (0, 0) - fine, since compute_eye_geometry only
    reads eye/iris indices.
    """
    points = np.full((478, 2), 0.0)
    left_loop = _ordered_loop(_LEFT_EYE)
    right_loop = _ordered_loop(_RIGHT_EYE)
    _place_eye(points, left_loop, _connection_indices(_LEFT_IRIS), left_center, *left_geometry)
    _place_eye(points, right_loop, _connection_indices(_RIGHT_IRIS), right_center, *right_geometry)
    return points


def _sclera_background_image(shape=(300, 500), sclera_rgb=(235, 230, 225)) -> np.ndarray:
    """A synthetic image with a plausible sclera-colored box around each
    default-positioned eye (see _synthetic_eye_landmarks' default centers),
    generous enough to fully cover the masked sclera region after shrink."""
    image_rgb = np.full((*shape, 3), 150, dtype=np.uint8)
    image_rgb[100:200, 60:240] = sclera_rgb
    image_rgb[100:200, 260:440] = sclera_rgb
    return image_rgb


def test_compute_eye_geometry_places_iris_center_and_radius():
    points = _synthetic_eye_landmarks(left_geometry=(60.0, 25.0, 12.0))

    left, _right = compute_eye_geometry(points)

    assert left.iris_center == pytest.approx((150.0, 150.0), abs=0.5)
    assert left.iris_radius == pytest.approx(12.0, abs=0.5)


def test_compute_eye_geometry_openness_ratio_reflects_eye_shape():
    flat_points = _synthetic_eye_landmarks(left_geometry=(60.0, 3.0, 2.0))
    round_points = _synthetic_eye_landmarks(left_geometry=(60.0, 40.0, 15.0))

    flat_left, _ = compute_eye_geometry(flat_points)
    round_left, _ = compute_eye_geometry(round_points)

    assert flat_left.openness_ratio < round_left.openness_ratio
    assert flat_left.openness_ratio < 0.15  # below _EYE_OPENNESS_MIN_RATIO: reads as closed


def test_sclera_clipped_fraction_flags_blown_out_eye_and_not_a_normal_one():
    points = _synthetic_eye_landmarks()
    left, right = compute_eye_geometry(points)

    normal_image = _sclera_background_image()
    glare_image = _sclera_background_image()
    glare_image[100:200, 60:240] = 255  # blow out the whole left eye region

    assert sclera_clipped_fraction(normal_image, left) == pytest.approx(0.0)
    assert sclera_clipped_fraction(glare_image, left) >= 0.3
    assert sclera_clipped_fraction(glare_image, right) == pytest.approx(0.0)


def test_sample_sclera_pools_both_eyes_when_both_reliable():
    points = _synthetic_eye_landmarks()
    image_rgb = _sclera_background_image(sclera_rgb=(235, 230, 225))
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    result = sample_sclera(image_rgb, image_lab, points)

    assert result.success is True
    assert result.sclera_rgb == (235, 230, 225)


def test_sample_sclera_falls_back_to_single_eye_when_other_is_closed():
    # Left eye squeezed nearly flat (closed/squinting); right eye normal.
    points = _synthetic_eye_landmarks(left_geometry=(60.0, 3.0, 2.0))
    image_rgb = _sclera_background_image()
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    result = sample_sclera(image_rgb, image_lab, points)

    assert result.success is True
    assert result.sclera_rgb == (235, 230, 225)


def test_sample_sclera_fails_when_both_eyes_closed():
    points = _synthetic_eye_landmarks(left_geometry=(60.0, 3.0, 2.0), right_geometry=(60.0, 3.0, 2.0))
    image_rgb = _sclera_background_image()
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    result = sample_sclera(image_rgb, image_lab, points)

    assert result.success is False
    assert result.error == "eyes_closed"
    assert result.sclera_rgb is None


def test_sample_sclera_trims_dark_eyelid_margin_contamination():
    points = _synthetic_eye_landmarks()
    image_rgb = _sclera_background_image()
    # A dark band along the bottom of the left eye's masked region -
    # eyelash/lid-crease shadow, a minority of the masked pixels.
    image_rgb[164:172, 60:240] = (20, 20, 20)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    result = sample_sclera(image_rgb, image_lab, points)

    assert result.success is True
    assert result.sclera_rgb == (235, 230, 225)  # unaffected by the shadow band


def test_sample_sclera_trims_bright_specular_catchlight():
    points = _synthetic_eye_landmarks()
    image_rgb = _sclera_background_image()
    # A small blown-out catchlight cluster, well under 5% of the masked
    # region - should be trimmed by the percentile band's top cut, not pull
    # the median toward white.
    image_rgb[145:155, 105:115] = 255
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    result = sample_sclera(image_rgb, image_lab, points)

    assert result.success is True
    assert result.sclera_rgb == (235, 230, 225)


def test_sample_sclera_fails_with_insufficient_pixels_for_tiny_eyes():
    points = _synthetic_eye_landmarks(left_geometry=(1.0, 1.0, 0.3), right_geometry=(1.0, 1.0, 0.3))
    image_rgb = _sclera_background_image()
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    result = sample_sclera(image_rgb, image_lab, points)

    assert result.success is False
    assert result.error == "insufficient_pixels"


def test_sample_sclera_excludes_extreme_angle_eye_but_succeeds_via_other():
    # Right eye much narrower than the left - a foreshortening pattern
    # consistent with an extreme head angle, so only the left eye should
    # be trusted, but that's enough for a successful reading.
    points = _synthetic_eye_landmarks(right_geometry=(15.0, 25.0, 10.0))
    image_rgb = _sclera_background_image()
    image_rgb[100:200, 320:380] = (235, 230, 225)  # cover the narrower right eye's box too
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    result = sample_sclera(image_rgb, image_lab, points)

    assert result.success is True
    assert result.sclera_rgb == (235, 230, 225)


def test_sample_sclera_fails_when_both_eyes_glared():
    points = _synthetic_eye_landmarks()
    image_rgb = _sclera_background_image()
    image_rgb[100:200, 60:240] = 255
    image_rgb[100:200, 260:440] = 255
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    result = sample_sclera(image_rgb, image_lab, points)

    assert result.success is False
    assert result.error == "sclera_clipped"
