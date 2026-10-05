"""Tests for decode_image: color-managed decoding of uploaded photos."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

from app.vision.image_decode import decode_image
from app.vision.season_classifier import hue_and_chroma, rgb_to_lab
from app.vision.skin_sampling import _MODEL_PATH, sample_skin_regions

_P3_FIXTURE = Path(__file__).parent / "fixtures" / "photos" / "lighting" / "SampleLightDaylight.jpeg"


def _jpeg_bytes(icc_profile: bytes | None = None) -> bytes:
    # A smooth gradient rather than a flat color, so a transform that's
    # silently skipped or misapplied can't hide behind one lucky pixel value.
    gradient = np.zeros((32, 48, 3), dtype=np.uint8)
    gradient[..., 0] = np.linspace(40, 240, 48, dtype=np.uint8)[None, :]
    gradient[..., 1] = np.linspace(30, 200, 32, dtype=np.uint8)[:, None]
    gradient[..., 2] = 120
    buffer = io.BytesIO()
    kwargs = {"icc_profile": icc_profile} if icc_profile is not None else {}
    Image.fromarray(gradient).save(buffer, "JPEG", quality=95, **kwargs)
    return buffer.getvalue()


def _naive_decode(contents: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_untagged_image_decodes_unchanged():
    contents = _jpeg_bytes()

    np.testing.assert_array_equal(decode_image(contents), _naive_decode(contents))


def test_srgb_tagged_image_is_effectively_unchanged():
    srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    contents = _jpeg_bytes(icc_profile=srgb_profile)

    difference = np.abs(decode_image(contents).astype(int) - _naive_decode(contents).astype(int))
    assert difference.max() <= 1


def test_corrupt_icc_profile_falls_back_to_unconverted_decode():
    contents = _jpeg_bytes(icc_profile=b"definitely not an ICC profile")

    np.testing.assert_array_equal(decode_image(contents), _naive_decode(contents))


def test_undecodable_bytes_return_none():
    assert decode_image(b"not an image at all") is None


@pytest.mark.skipif(
    not _P3_FIXTURE.exists() or not _MODEL_PATH.exists(),
    reason="requires the local Display P3 fixture photo and the MediaPipe model",
)
def test_display_p3_photo_is_converted_to_srgb():
    # An iPhone photo's P3 pixel values describe more saturated colors than
    # the same values read as sRGB, so converting should raise the sampled
    # skin chroma - by ~18-20% on this photo.
    contents = _P3_FIXTURE.read_bytes()
    naive_sample = sample_skin_regions(_naive_decode(contents))
    converted_sample = sample_skin_regions(decode_image(contents))
    assert naive_sample.success and converted_sample.success

    _, naive_chroma = hue_and_chroma(rgb_to_lab(naive_sample.left_cheek_rgb))
    _, converted_chroma = hue_and_chroma(rgb_to_lab(converted_sample.left_cheek_rgb))
    assert converted_chroma > naive_chroma * 1.1
