"""Deterministic face-landmark based skin color sampling.

No LLM involved anywhere in this module — see CLAUDE.md's classifier rule.
This is the input stage for the (separate) rule-based season classifier: it
locates a face, samples forehead/cheek patches, and hands back robust median
RGB per region. The caller decodes the upload and discards it right after
calling this; nothing here persists the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

_LEFT_EYE = vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
_RIGHT_EYE = vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
_LEFT_EYEBROW = vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
_RIGHT_EYEBROW = vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
_FACE_OVAL = vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL

# Anchor offsets and patch size are ratios of the face's own eyebrow/eye/chin
# geometry rather than fixed landmark indices or pixel counts, so they scale
# with head size, distance from camera, and mild tilt instead of assuming a
# fixed hairline or cheek position.
_FOREHEAD_OFFSET_RATIO = 0.20  # of chin-to-brow distance, above the brows
_CHEEK_DOWN_RATIO = 0.45  # of interocular distance, below the eyes
# Of half the face oval's own width, measured out from eye-line center - not
# a multiple of interocular distance. Interocular-to-face-width isn't
# guaranteed to scale consistently across face shapes, and tuning this as a
# multiple of interocular distance (twice, across two sessions) kept landing
# right at the jaw/ear boundary instead of mid-cheek once measured against
# real photos instead of eyeballed. Measuring directly against the face's
# own width is the quantity we actually care about: how far in from the
# visible edge of the face the sample point sits. 0.55 was picked by sweeping
# candidates against real photos and measuring each cheek anchor's distance
# to the true oval boundary interpolated AT THAT ANCHOR'S HEIGHT (see
# test_anchor_placement_real_photos.py) - it keeps every sample in the
# fixture corpus between ~37% and ~78% of the local half-width in from the
# edge (0.75+ already lands one sample's anchor outside the face entirely).
_CHEEK_OUT_RATIO = 0.55
_PATCH_HALF_SIZE_RATIO = 0.09  # of interocular distance
_MIN_PATCH_HALF_SIZE = 4  # pixels, for close-up/high-res photos
_OUTLIER_PERCENTILE_BAND = (10, 90)  # trims shadow/highlight pixels by Lab L
_CLIPPING_NEAR_THRESHOLD = 5  # channel value within this of 0 or 255 counts as clipped
_CLIPPED_PIXEL_FRACTION_THRESHOLD = 0.3  # this much of a patch clipped -> unreliable

RGB = tuple[int, int, int]
SampleFailureReason = Literal["no_face_detected", "face_out_of_frame", "patch_clipped"]


@dataclass(frozen=True)
class SkinSampleResult:
    success: bool
    forehead_rgb: RGB | None = None
    left_cheek_rgb: RGB | None = None
    right_cheek_rgb: RGB | None = None
    error: SampleFailureReason | None = None


@dataclass(frozen=True)
class AnchorPoints:
    forehead: np.ndarray
    left_cheek: np.ndarray
    right_cheek: np.ndarray
    patch_half_size: float


def _connection_indices(connections) -> list[int]:
    indices = set()
    for connection in connections:
        indices.add(connection.start)
        indices.add(connection.end)
    return sorted(indices)


def _centroid(points: np.ndarray, indices: list[int]) -> np.ndarray:
    return points[indices].mean(axis=0)


def _lowest_point(points: np.ndarray, indices: list[int]) -> np.ndarray:
    subset = points[indices]
    return subset[np.argmax(subset[:, 1])]


def compute_anchor_points(points: np.ndarray) -> AnchorPoints:
    """Derive forehead/cheek sample centers from face landmark pixel coordinates.

    `points` is an (N, 2) array of (x, y) pixel coordinates, N covering at
    least the eyebrow, eye, and face-oval landmark indices. Pure geometry —
    no model inference — so it's unit-testable with synthetic landmarks.
    """
    brow_center = _centroid(
        points, _connection_indices(_LEFT_EYEBROW) + _connection_indices(_RIGHT_EYEBROW)
    )
    oval_indices = _connection_indices(_FACE_OVAL)
    chin = _lowest_point(points, oval_indices)
    oval_x = points[oval_indices][:, 0]
    half_face_width = (oval_x.max() - oval_x.min()) / 2
    left_eye = _centroid(points, _connection_indices(_LEFT_EYE))
    right_eye = _centroid(points, _connection_indices(_RIGHT_EYE))

    face_axis = brow_center - chin  # "up" the face, from chin toward brow
    face_height = np.linalg.norm(face_axis)
    face_up = face_axis / face_height
    face_down = -face_up

    interocular = np.linalg.norm(right_eye - left_eye)
    eye_center = (left_eye + right_eye) / 2
    left_out = (left_eye - eye_center) / (np.linalg.norm(left_eye - eye_center) + 1e-6)
    right_out = (right_eye - eye_center) / (np.linalg.norm(right_eye - eye_center) + 1e-6)

    forehead = brow_center + face_up * face_height * _FOREHEAD_OFFSET_RATIO
    cheek_down_offset = face_down * interocular * _CHEEK_DOWN_RATIO
    left_cheek = eye_center + cheek_down_offset + left_out * half_face_width * _CHEEK_OUT_RATIO
    right_cheek = eye_center + cheek_down_offset + right_out * half_face_width * _CHEEK_OUT_RATIO

    patch_half_size = max(interocular * _PATCH_HALF_SIZE_RATIO, _MIN_PATCH_HALF_SIZE)

    return AnchorPoints(forehead, left_cheek, right_cheek, patch_half_size)


def _patch_bounds(
    image_shape: tuple[int, int], center: np.ndarray, half_size: float
) -> tuple[int, int, int, int] | None:
    """Clamp a square patch to the image, or None if it falls entirely outside."""
    height, width = image_shape
    center_x, center_y = int(round(center[0])), int(round(center[1]))
    radius = max(int(round(half_size)), 1)

    x0, x1 = max(center_x - radius, 0), min(center_x + radius, width)
    y0, y1 = max(center_y - radius, 0), min(center_y + radius, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, x1, y0, y1


def sample_patch_rgb(image_rgb: np.ndarray, image_lab: np.ndarray, center: np.ndarray, half_size: float) -> RGB | None:
    """Median RGB of a square patch, after dropping shadow/highlight outliers.

    Outliers are identified by Lab lightness (L channel) percentile rather
    than raw RGB brightness, since Lab separates luminance from hue/chroma
    cleanly. Returns None if the patch falls entirely outside the image.
    """
    bounds = _patch_bounds(image_rgb.shape[:2], center, half_size)
    if bounds is None:
        return None
    x0, x1, y0, y1 = bounds

    rgb_pixels = image_rgb[y0:y1, x0:x1].reshape(-1, 3)
    lightness = image_lab[y0:y1, x0:x1, 0].reshape(-1)

    low, high = np.percentile(lightness, _OUTLIER_PERCENTILE_BAND)
    keep = (lightness >= low) & (lightness <= high)
    if not keep.any():
        keep = np.ones_like(keep)

    median_rgb = np.median(rgb_pixels[keep], axis=0)
    return int(round(median_rgb[0])), int(round(median_rgb[1])), int(round(median_rgb[2]))


def patch_clipped_fraction(image_rgb: np.ndarray, center: np.ndarray, half_size: float) -> float | None:
    """Fraction of a patch's raw pixels that are true optical blowout/blackout.

    A pixel only counts as clipped when ALL THREE channels sit near 0 or
    near 255 together — a genuine loss of sensor detail desaturates toward
    black or white across every channel. A single channel near its ceiling
    (e.g. a warm/bright but otherwise normal skin tone with a high red
    channel and clearly lower green/blue) is a real, valid color, not a
    clipped one, so it must not trip this check on its own.

    Operates on the raw (pre-trim) pixels rather than the trimmed median,
    since a majority-clipped patch's trimmed median can still land on a
    plausible-looking value — trimming only rejects a minority of outliers,
    so it can't catch clipping that affects most of the patch. Returns None
    if the patch falls entirely outside the image, mirroring sample_patch_rgb.
    """
    bounds = _patch_bounds(image_rgb.shape[:2], center, half_size)
    if bounds is None:
        return None
    x0, x1, y0, y1 = bounds

    pixels = image_rgb[y0:y1, x0:x1].reshape(-1, 3)
    near_white = np.all(pixels >= 255 - _CLIPPING_NEAR_THRESHOLD, axis=1)
    near_black = np.all(pixels <= _CLIPPING_NEAR_THRESHOLD, axis=1)
    return float((near_white | near_black).mean())


@lru_cache(maxsize=1)
def _landmarker() -> vision.FaceLandmarker:
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _face_bbox_area(points: np.ndarray) -> float:
    width = points[:, 0].max() - points[:, 0].min()
    height = points[:, 1].max() - points[:, 1].min()
    return float(width * height)


def sample_skin_regions(image_bgr: np.ndarray) -> SkinSampleResult:
    """Detect a face and median-sample forehead/left-cheek/right-cheek color.

    `image_bgr` is a decoded image array (e.g. from `cv2.imdecode`), in
    OpenCV's default BGR channel order. If multiple faces are found, the
    largest (by bounding-box area) is used, on the assumption this is a
    single-subject selfie. Never raises for "expected" bad input — a
    missing or unusable face comes back as `success=False` with a reason,
    so the API layer can turn it into a "please retake your photo" response.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)

    result = _landmarker().detect(mp_image)
    if not result.face_landmarks:
        return SkinSampleResult(success=False, error="no_face_detected")

    height, width = image_rgb.shape[:2]
    faces = [
        np.array([(landmark.x * width, landmark.y * height) for landmark in landmarks])
        for landmarks in result.face_landmarks
    ]
    largest_face = max(faces, key=_face_bbox_area)

    anchors = compute_anchor_points(largest_face)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    forehead_rgb = sample_patch_rgb(image_rgb, image_lab, anchors.forehead, anchors.patch_half_size)
    left_cheek_rgb = sample_patch_rgb(image_rgb, image_lab, anchors.left_cheek, anchors.patch_half_size)
    right_cheek_rgb = sample_patch_rgb(image_rgb, image_lab, anchors.right_cheek, anchors.patch_half_size)

    if forehead_rgb is None or left_cheek_rgb is None or right_cheek_rgb is None:
        return SkinSampleResult(success=False, error="face_out_of_frame")

    clipped_fractions = (
        patch_clipped_fraction(image_rgb, anchors.forehead, anchors.patch_half_size),
        patch_clipped_fraction(image_rgb, anchors.left_cheek, anchors.patch_half_size),
        patch_clipped_fraction(image_rgb, anchors.right_cheek, anchors.patch_half_size),
    )
    if any(f is not None and f >= _CLIPPED_PIXEL_FRACTION_THRESHOLD for f in clipped_fractions):
        return SkinSampleResult(success=False, error="patch_clipped")

    return SkinSampleResult(
        success=True,
        forehead_rgb=forehead_rgb,
        left_cheek_rgb=left_cheek_rgb,
        right_cheek_rgb=right_cheek_rgb,
    )
