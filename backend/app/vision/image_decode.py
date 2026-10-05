"""Decode uploaded photo bytes into an sRGB pixel array.

Every color threshold and reference constant in season_classifier.py is in
sRGB terms, but phones don't all shoot sRGB: iPhones embed a Display P3 ICC
profile, and iOS Safari keeps it when it transcodes HEIC to JPEG for upload.
cv2.imdecode ignores embedded profiles, so a P3 photo's pixel values used to
be read as if they were sRGB - under-reading chroma by ~18-20% and shifting
hue - while an Android or desktop-camera photo of the same face was read
correctly. Converting through the embedded profile puts every upload on the
same scale.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageCms

logger = logging.getLogger(__name__)

_SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def decode_image(contents: bytes) -> np.ndarray | None:
    """Decode upload bytes into a BGR array in sRGB, or None if undecodable.

    Pixels and EXIF orientation come from cv2.imdecode exactly as before
    (see test_exif_orientation.py), so patch coordinates still line up with
    the browser's rendering of the same photo. An embedded ICC profile, when
    there is one, is then used to convert to sRGB. Untagged images are
    treated as sRGB already - the web's convention, and what the desktop
    camera capture's canvas.toBlob produces.
    """
    image_bgr = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    icc_profile = _embedded_icc_profile(contents)
    transform = _to_srgb_transform(icc_profile) if icc_profile else None
    if transform is None:
        return image_bgr

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    converted = ImageCms.applyTransform(Image.fromarray(image_rgb), transform)
    return cv2.cvtColor(np.asarray(converted), cv2.COLOR_RGB2BGR)


def _embedded_icc_profile(contents: bytes) -> bytes | None:
    """The upload's embedded ICC profile, read from its header only (no
    pixel decode), or None if it has none or Pillow can't parse the file."""
    try:
        with Image.open(io.BytesIO(contents)) as image:
            return image.info.get("icc_profile") or None
    except Exception:
        return None


@lru_cache(maxsize=8)
def _to_srgb_transform(icc_profile: bytes) -> ImageCms.ImageCmsTransform | None:
    """A transform from `icc_profile` to sRGB, or None if the profile can't
    be used (corrupt, or not an RGB profile - e.g. CMYK or grayscale).

    Cached by profile bytes, since real uploads carry only a handful of
    distinct profiles. NOCACHE turns off lcms's one-pixel cache, which is
    what makes one shared transform safe to apply from several threads.
    Relative colorimetric is the only intent matrix-shaper profiles like
    Apple's Display P3 implement.
    """
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
        return ImageCms.buildTransform(
            source,
            _SRGB_PROFILE,
            "RGB",
            "RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            flags=ImageCms.Flags.NOCACHE,
        )
    except (ImageCms.PyCMSError, OSError, TypeError, ValueError):
        logger.warning("Ignoring an unusable embedded ICC profile; decoding as sRGB", exc_info=True)
        return None
