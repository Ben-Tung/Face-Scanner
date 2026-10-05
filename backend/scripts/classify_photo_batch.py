#!/usr/bin/env python3
"""Manual dev tool: run the full detect -> sample -> classify pipeline across
every photo in the local fixture batch and print one comparison row each,
raw vs. sclera-corrected depth AND undertone (hue). Not a unit test - the
permanent, scripted version of the by-hand, one-photo-at-a-time comparison
that originally surfaced the raw-L*/lighting confound the depth correction
addresses, extended to the analogous raw-hue/color-cast confound the
undertone correction (normalize_undertone_ab) addresses. Use this to eyeball
whether depth ordering and undertone agreement across a batch match visual
judgment, and to (re)tune the sclera-normalization constants in
season_classifier.py against a larger photo set over time - the printed
batch-mean sclera Lab at the end is exactly what those constants are
calibrated from.

Usage:
    python scripts/classify_photo_batch.py [path/to/fixtures/dir]

Defaults to backend/tests/fixtures/photos (the same gitignored local fixture
directory the real-photo pytest suites use).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision.image_decode import decode_image
from app.vision.season_classifier import classify_season, rgb_to_lab
from app.vision.skin_sampling import sample_skin_regions

_DEFAULT_FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "photos"


def _format_optional(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def main() -> None:
    fixtures_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_FIXTURES_DIR
    photo_paths = sorted(fixtures_dir.glob("*.jp*g"))
    if not photo_paths:
        print(f"No photos found in {fixtures_dir}", file=sys.stderr)
        raise SystemExit(1)

    header = (
        f"{'photo':14s} {'raw_L':>6s} {'scl_L':>6s} {'scl_a':>6s} {'scl_b':>6s} {'cast':>6s} "
        f"{'depth_L':>8s} {'raw_dep':>8s} {'new_dep':>8s} "
        f"{'raw_hue':>8s} {'cor_hue':>8s} {'raw_und':>8s} {'cor_und':>8s} "
        f"{'season':>7s} {'fh_drop':>7s} sclera_status"
    )
    print(header)
    print("-" * len(header))

    fallback_count = 0
    sclera_l_values: list[float] = []
    sclera_a_values: list[float] = []
    sclera_b_values: list[float] = []

    for photo_path in photo_paths:
        image_bgr = decode_image(photo_path.read_bytes())
        if image_bgr is None:
            print(f"{photo_path.name:14s} (could not read image)")
            continue

        sample = sample_skin_regions(image_bgr)
        if not sample.success:
            print(f"{photo_path.name:14s} sampling failed: {sample.error}")
            continue

        raw_result = classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb)
        if not raw_result.success:
            print(
                f"{photo_path.name:14s} classification failed: {raw_result.error} "
                f"(max ΔH* {raw_result.max_hue_difference:.2f})"
            )
            continue
        raw_l = raw_result.classification.avg_lab[0]
        raw_depth = raw_result.classification.depth
        raw_hue = raw_result.classification.hue_deg
        raw_undertone = raw_result.classification.undertone

        sclera_rgb = None
        sclera_l_str = sclera_a_str = sclera_b_str = "-"
        sclera_status = "n/a"
        if sample.sclera is not None:
            if sample.sclera.success:
                sclera_rgb = sample.sclera.sclera_rgb
                sclera_l, sclera_a, sclera_b = rgb_to_lab(sclera_rgb)
                sclera_l_str = f"{sclera_l:.1f}"
                sclera_a_str = f"{sclera_a:.1f}"
                sclera_b_str = f"{sclera_b:.1f}"
                sclera_l_values.append(sclera_l)
                sclera_a_values.append(sclera_a)
                sclera_b_values.append(sclera_b)
                sclera_status = "ok"
            else:
                sclera_status = sample.sclera.error or "unreliable"

        if sclera_rgb is None:
            fallback_count += 1

        result = classify_season(
            sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb, sclera_rgb=sclera_rgb
        )
        if not result.success:
            print(
                f"{photo_path.name:14s} classification (corrected) failed: {result.error} "
                f"(max ΔH* {result.max_hue_difference:.2f}, color cast {_format_optional(result.color_cast)})"
            )
            continue

        classification = result.classification
        print(
            f"{photo_path.name:14s} {raw_l:6.1f} {sclera_l_str:>6s} {sclera_a_str:>6s} {sclera_b_str:>6s} "
            f"{_format_optional(result.color_cast):>6s} "
            f"{classification.depth_lightness:8.2f} {raw_depth:>8s} {classification.depth:>8s} "
            f"{raw_hue:8.1f} {classification.hue_deg:8.1f} {raw_undertone:>8s} {classification.undertone:>8s} "
            f"{classification.season:>7s} {'yes' if result.forehead_dropped else '-':>7s} {sclera_status}"
        )

    print("-" * len(header))
    print(f"{fallback_count}/{len(photo_paths)} photos fell back to uncorrected L*/a*/b* (sclera unreliable or unreadable)")

    if sclera_l_values:
        mean_l = sum(sclera_l_values) / len(sclera_l_values)
        mean_a = sum(sclera_a_values) / len(sclera_a_values)
        mean_b = sum(sclera_b_values) / len(sclera_b_values)
        print(
            f"Mean sclera Lab over {len(sclera_l_values)} successful readings: "
            f"L*={mean_l:.2f} a*={mean_a:.2f} b*={mean_b:.2f}"
        )
        print(
            "Mean L* is the calibration value for _SCLERA_REFERENCE_L in season_classifier.py. "
            "_SCLERA_REFERENCE_A/_B come from the daylight photo alone (see their comment), "
            "and the cast column is checked against _MAX_COLOR_CAST_AB."
        )
    else:
        print("No successful sclera readings in this batch - cannot compute reference constants.")


if __name__ == "__main__":
    main()
