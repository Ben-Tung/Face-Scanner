"""Pins an assumption the manual patch-adjustment feature depends on: that
`cv2.imdecode` applies EXIF orientation the same way a browser <img> does.

The backend hands the frontend patch coordinates in the *decoded* image's
pixel space, and the frontend overlays them on a browser-rendered <img> of
the same photo. If the two disagreed on which way is "up" for a rotated
phone photo, the overlay would silently misplace the boxes. OpenCV's
IMREAD_COLOR has applied EXIF rotation by default for a long time (see
loadsave.cpp: only IMREAD_IGNORE_ORIENTATION or IMREAD_UNCHANGED skip it),
matching how browsers render <img> by default — this test is a regression
guard for that default surviving a future opencv-python-headless bump.
"""

from __future__ import annotations

import cv2
import numpy as np


def _jpeg_bytes_with_exif_orientation(image_bgr: np.ndarray, orientation: int) -> bytes:
    """Encode as JPEG and splice in a minimal EXIF APP1 segment declaring
    the given Orientation tag."""
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    assert ok
    jpeg_bytes = encoded.tobytes()

    tiff_header = b"II" + (42).to_bytes(2, "little") + (8).to_bytes(4, "little")
    ifd_entry = (
        (0x0112).to_bytes(2, "little")
        + (3).to_bytes(2, "little")
        + (1).to_bytes(4, "little")
        + orientation.to_bytes(2, "little")
        + b"\x00\x00"
    )
    ifd = (1).to_bytes(2, "little") + ifd_entry + (0).to_bytes(4, "little")
    exif_payload = b"Exif\x00\x00" + tiff_header + ifd
    app1 = b"\xff\xe1" + (len(exif_payload) + 2).to_bytes(2, "big") + exif_payload

    return jpeg_bytes[:2] + app1 + jpeg_bytes[2:]  # insert right after the JPEG SOI marker


def test_decode_honors_exif_orientation_like_browsers():
    # 40 wide x 20 tall, landscape.
    image_bgr = np.full((20, 40, 3), 100, dtype=np.uint8)
    # Orientation 6 = "rotate 90 CW to display" - what a phone held in one
    # portrait grip commonly tags a landscape sensor capture with.
    jpeg_bytes = _jpeg_bytes_with_exif_orientation(image_bgr, orientation=6)

    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    # A decoder that ignores EXIF would keep shape (20, 40) - the raw sensor
    # layout. One that honors it, matching browser <img> rendering, swaps to
    # a tall portrait image.
    assert decoded.shape[:2] == (40, 20)
