#!/usr/bin/env python3
"""Manual dev tool: run the full detect -> sample -> classify pipeline on a
real photo file and print the result. Not a unit test — skin_sampling.py's
and season_classifier.py's pytest suites cover those with deterministic
synthetic inputs. This script is for eyeballing behavior on an actual photo.

Usage:
    python scripts/classify_photo.py path/to/photo.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision.season_classifier import classify_season, hue_and_chroma, rgb_to_lab
from app.vision.skin_sampling import sample_skin_regions


def _print_patch(name: str, rgb) -> None:
    lab = rgb_to_lab(rgb)
    hue_deg, chroma = hue_and_chroma(lab)
    print(
        f"{name:12s} rgb={rgb!s:16s} L={lab[0]:6.2f} a={lab[1]:6.2f} b={lab[2]:6.2f} "
        f"hue={hue_deg:6.2f} chroma={chroma:6.2f}"
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

    sample = sample_skin_regions(image_bgr)
    if not sample.success:
        print(f"Sampling failed: {sample.error}")
        raise SystemExit(1)

    _print_patch("forehead", sample.forehead_rgb)
    _print_patch("left_cheek", sample.left_cheek_rgb)
    _print_patch("right_cheek", sample.right_cheek_rgb)

    result = classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb)
    if not result.success:
        print()
        print(f"Classification failed: {result.error}")
        raise SystemExit(1)

    classification = result.classification
    print()
    print(f"season    = {classification.season}")
    print(f"undertone = {classification.undertone}")
    print(f"depth     = {classification.depth}")
    print(f"clarity   = {classification.clarity}")
    print(
        f"avg_lab   = ({classification.avg_lab[0]:.2f}, {classification.avg_lab[1]:.2f}, "
        f"{classification.avg_lab[2]:.2f})"
    )
    print(f"hue_deg   = {classification.hue_deg:.2f}")
    print(f"chroma    = {classification.chroma:.2f}")


if __name__ == "__main__":
    main()
