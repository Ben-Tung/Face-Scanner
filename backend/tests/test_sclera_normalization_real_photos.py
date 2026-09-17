"""Sanity check: sclera-corrected depth should reorder real photos to match
informal human visual judgment, in cases where raw L* alone gets it wrong.

This is NOT proof of absolute correctness - no labeled ground-truth season
dataset exists for this app (see season_classifier.py's module docstring).
These are relative-ordering checks against a human's own visual read of a
handful of real photos: "this one looks lighter/deeper than that one." If a
pair regresses, it means the correction got *less* aligned with that human
judgment than before, which is worth knowing even without absolute labels.

Requires local, non-committed photo fixtures at
backend/tests/fixtures/photos/ (see .gitignore) and the MediaPipe model file
from backend/scripts/download_models.sh. Skips cleanly when either is
missing. Individual pairwise checks additionally skip if either named sample
isn't present, or if sclera correction wasn't actually applied to both (a
raw-L*-fallback photo can't test the correction itself).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.vision.skin_sampling import _MODEL_PATH, sample_skin_regions
from app.vision.season_classifier import SeasonClassification, classify_season

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "photos"
PHOTO_PATHS = sorted(FIXTURES_DIR.glob("*.jp*g")) if FIXTURES_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not PHOTO_PATHS or not _MODEL_PATH.exists(),
    reason=(
        "Local photo fixtures or the MediaPipe model aren't available - see "
        "backend/tests/fixtures/photos/ and backend/scripts/download_models.sh"
    ),
)


def _find_sample(name: str) -> Path | None:
    matches = [p for p in PHOTO_PATHS if p.stem == name]
    return matches[0] if matches else None


def _classify_corrected(photo_path: Path) -> SeasonClassification | None:
    """Runs the full pipeline; returns None if sclera correction wasn't
    actually applied (sampling failed, or the sclera itself was unreadable),
    since a pairwise check against uncorrected depth_lightness would prove
    nothing about the correction."""
    image_bgr = cv2.imread(str(photo_path))
    sample = sample_skin_regions(image_bgr)
    if not sample.success or sample.sclera is None or not sample.sclera.success:
        return None
    result = classify_season(
        sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb, sclera_rgb=sample.sclera.sclera_rgb
    )
    if not result.success:
        return None
    return result.classification


def _assert_corrected_ordering(lighter_name: str, deeper_name: str) -> None:
    lighter_path = _find_sample(lighter_name)
    deeper_path = _find_sample(deeper_name)
    if lighter_path is None or deeper_path is None:
        pytest.skip(f"{lighter_name}/{deeper_name} fixtures not present")

    lighter = _classify_corrected(lighter_path)
    deeper = _classify_corrected(deeper_path)
    if lighter is None or deeper is None:
        pytest.skip(f"sclera correction unavailable for {lighter_name} and/or {deeper_name} in this environment")

    assert lighter.depth_lightness > deeper.depth_lightness, (
        f"{lighter_name} (depth_lightness={lighter.depth_lightness:.1f}) should read lighter than "
        f"{deeper_name} (depth_lightness={deeper.depth_lightness:.1f}) after sclera correction"
    )


def test_sample2_reads_lighter_than_sample4_after_correction():
    _assert_corrected_ordering("Sample2", "Sample4")


def test_sample3_reads_lighter_than_sample9_after_correction():
    _assert_corrected_ordering("Sample3", "Sample9")


def test_sample6_reads_lighter_than_sample7_after_correction():
    _assert_corrected_ordering("Sample6", "Sample7")


def test_sample8_reads_lighter_than_sample7_after_correction():
    _assert_corrected_ordering("Sample8", "Sample7")
