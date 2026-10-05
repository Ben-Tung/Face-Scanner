"""Regression check: evenly lit real photos must not be rejected as
inconsistent_patches just because the skin reads pale/desaturated.

SampleEvenLightLowChroma is a real selfie under even daylight whose patches
read chroma ~7-10. The old hue-angle spread check (max(hue) - min(hue) > 25
degrees) rejected it twice over: no 0/360 wraparound (cheeks at 357.1 and
8.5 degrees read as a 348.6 degree spread), and even wrapped it's 43.6
degrees, because hue angle is mostly noise at that chroma. The ΔH* check
that replaced it (see season_classifier.hue_difference) reads it at ~6.75
color-managed (~6.07 before), under _HUE_DIFFERENCE_THRESHOLD.
test_season_classifier.py pins the same case with this photo's sampled
RGBs; this file checks the full detect -> sample -> classify pipeline end
to end, so a change to anchor placement, patch sampling or decoding that
reintroduces the failure gets caught too.

Its forehead patch also sits in the shadow of the subject's fringe
(forehead L* ~29 below the cheeks), so it's left out of the average (see
season_classifier._SHADED_FOREHEAD_L_GAP) - but it still counts toward the
hue check, and that shaded forehead is most of this photo's ΔH* (the cheeks
alone read ~2.0). Which season it lands on is covered by
test_same_subject_consistency_real_photos.py; this only asserts it
classifies.

Lives in its own fixtures/photos/consistency/ subdirectory so the
non-recursive photo globs in the other real-photo suites (and the
sclera-reference batch in scripts/classify_photo_batch.py) don't pick it up.
Requires local, non-committed photo fixtures (see .gitignore) and the
MediaPipe model file from backend/scripts/download_models.sh. Skips cleanly
when either is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.vision.image_decode import decode_image
from app.vision.season_classifier import _HUE_DIFFERENCE_THRESHOLD, classify_season
from app.vision.skin_sampling import _MODEL_PATH, sample_skin_regions

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "photos" / "consistency"
PHOTO_PATHS = sorted(FIXTURES_DIR.glob("*.jp*g")) if FIXTURES_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not PHOTO_PATHS or not _MODEL_PATH.exists(),
    reason=(
        "Local consistency photo fixtures or the MediaPipe model aren't available - see "
        "backend/tests/fixtures/photos/consistency/ and backend/scripts/download_models.sh"
    ),
)


@pytest.mark.parametrize("photo_path", PHOTO_PATHS, ids=lambda p: p.stem)
def test_evenly_lit_photo_classifies(photo_path: Path):
    sample = sample_skin_regions(decode_image(photo_path.read_bytes()))
    assert sample.success, f"sampling failed: {sample.error}"

    sclera_rgb = sample.sclera.sclera_rgb if sample.sclera is not None and sample.sclera.success else None
    result = classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb, sclera_rgb=sclera_rgb)

    assert result.success, (
        f"rejected as {result.error} with max ΔH* {result.max_hue_difference:.2f} "
        f"(threshold {_HUE_DIFFERENCE_THRESHOLD})"
    )
