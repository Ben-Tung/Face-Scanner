#!/usr/bin/env python3
"""Manual dev tool: run the full detect -> sample -> classify pipeline across
every photo in the local fixture batch and print one comparison row each,
raw vs. sclera-corrected depth. Not a unit test - the permanent, scripted
version of the by-hand, one-photo-at-a-time comparison that originally
surfaced the raw-L*/lighting confound this correction addresses. Use this to
eyeball whether depth ordering across a batch matches visual judgment, and to
tune the sclera-normalization constants in season_classifier.py and
skin_sampling.py against a larger photo set over time.

Usage:
    python scripts/classify_photo_batch.py [path/to/fixtures/dir]

Defaults to backend/tests/fixtures/photos (the same gitignored local fixture
directory the real-photo pytest suites use).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision.season_classifier import classify_season, rgb_to_lab
from app.vision.skin_sampling import sample_skin_regions

_DEFAULT_FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "photos"


def main() -> None:
    fixtures_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_FIXTURES_DIR
    photo_paths = sorted(fixtures_dir.glob("*.jp*g"))
    if not photo_paths:
        print(f"No photos found in {fixtures_dir}", file=sys.stderr)
        raise SystemExit(1)

    header = (
        f"{'photo':14s} {'raw_L':>7s} {'sclera_L':>9s} {'depth_L':>9s} "
        f"{'raw_depth':>10s} {'new_depth':>10s} {'season':>8s} sclera_status"
    )
    print(header)
    print("-" * len(header))

    fallback_count = 0
    for photo_path in photo_paths:
        image_bgr = cv2.imread(str(photo_path))
        if image_bgr is None:
            print(f"{photo_path.name:14s} (could not read image)")
            continue

        sample = sample_skin_regions(image_bgr)
        if not sample.success:
            print(f"{photo_path.name:14s} sampling failed: {sample.error}")
            continue

        raw_result = classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb)
        if not raw_result.success:
            print(f"{photo_path.name:14s} classification failed: {raw_result.error}")
            continue
        raw_l = raw_result.classification.avg_lab[0]
        raw_depth = raw_result.classification.depth

        sclera_rgb = None
        sclera_l_str = "-"
        sclera_status = "n/a"
        if sample.sclera is not None:
            if sample.sclera.success:
                sclera_rgb = sample.sclera.sclera_rgb
                sclera_l_str = f"{rgb_to_lab(sclera_rgb)[0]:.1f}"
                sclera_status = "ok"
            else:
                sclera_status = sample.sclera.error or "unreliable"

        if sclera_rgb is None:
            fallback_count += 1

        result = classify_season(
            sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb, sclera_rgb=sclera_rgb
        )
        if not result.success:
            print(f"{photo_path.name:14s} classification (corrected) failed: {result.error}")
            continue

        classification = result.classification
        print(
            f"{photo_path.name:14s} {raw_l:7.2f} {sclera_l_str:>9s} {classification.depth_lightness:9.2f} "
            f"{raw_depth:>10s} {classification.depth:>10s} {classification.season:>8s} {sclera_status}"
        )

    print("-" * len(header))
    print(f"{fallback_count}/{len(photo_paths)} photos fell back to uncorrected L* (sclera unreliable or unreadable)")


if __name__ == "__main__":
    main()
