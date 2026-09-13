#!/usr/bin/env python3
"""Manual dev tool: draw the sampled forehead/cheek patches onto a copy of
the photo so you can see exactly where sample_skin_regions() is reading
from. Diagnostic only — not production code, no tests.

Usage:
    python scripts/visualize_patches.py path/to/photo.jpg
Writes path/to/photo_annotated.jpg next to the original.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision.skin_sampling import _face_bbox_area, _landmarker, compute_anchor_points

_BOX_COLOR = (0, 255, 0)  # BGR
_BOX_THICKNESS = 2
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.6


def _draw_patch(image_bgr: np.ndarray, label: str, center: np.ndarray, half_size: float) -> None:
    x, y = int(round(center[0])), int(round(center[1]))
    radius = max(int(round(half_size)), 1)
    top_left = (x - radius, y - radius)
    bottom_right = (x + radius, y + radius)
    cv2.rectangle(image_bgr, top_left, bottom_right, _BOX_COLOR, _BOX_THICKNESS)
    cv2.putText(
        image_bgr, label, (top_left[0], top_left[1] - 8), _LABEL_FONT, _LABEL_SCALE, _BOX_COLOR, 2
    )


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/photo.jpg", file=sys.stderr)
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        raise SystemExit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)

    result = _landmarker().detect(mp_image)
    if not result.face_landmarks:
        print("No face detected")
        raise SystemExit(1)

    height, width = image_rgb.shape[:2]
    faces = [
        np.array([(landmark.x * width, landmark.y * height) for landmark in landmarks])
        for landmarks in result.face_landmarks
    ]
    largest_face = max(faces, key=_face_bbox_area)
    anchors = compute_anchor_points(largest_face)

    annotated = image_bgr.copy()
    _draw_patch(annotated, "forehead", anchors.forehead, anchors.patch_half_size)
    _draw_patch(annotated, "left_cheek", anchors.left_cheek, anchors.patch_half_size)
    _draw_patch(annotated, "right_cheek", anchors.right_cheek, anchors.patch_half_size)

    output_path = image_path.with_name(f"{image_path.stem}_annotated{image_path.suffix}")
    cv2.imwrite(str(output_path), annotated)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
