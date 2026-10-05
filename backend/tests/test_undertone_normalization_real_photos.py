"""Sanity check: sclera-corrected undertone should agree across different
lighting conditions on the same face/pose, where raw (uncorrected) hue does
not - see season_classifier.normalize_undertone_ab.

This is NOT proof of absolute correctness - no labeled ground-truth season
dataset exists for this app (see season_classifier.py's module docstring).
Three checks below, each necessary but not individually sufficient alone:

(a) Mutual agreement: corrected undertone must be the same across all three
    lighting conditions (warm incandescent bulb, a "white" LED, outdoor
    overcast daylight) for the same face/pose. This alone isn't enough - a
    systematically biased correction could make all three conditions
    confidently agree on the WRONG undertone.
(b) Anchored agreement: each corrected undertone must also match the raw,
    uncorrected undertone read from the daylight photo specifically - not
    just each other in the abstract. Daylight is the one condition with a
    physical claim to being close to color-neutral (diffuse outdoor
    skylight, no artificial-light color cast, no indoor-wall bounce
    contamination), so its raw reading is the best available proxy for
    ground truth without a labeled dataset. This is what would catch "all
    three converged, but converged on the wrong answer together," which
    (a) alone cannot.
(c) Independent human judgment (recorded here, not asserted in code, since
    there's no way to automate "is this actually correct"): the test
    subject independently reported believing their own undertone is WARM
    (based on a jewelry/vein-color self-assessment), which agrees with what
    (a)/(b) converge on below. Caveat: this wasn't a blind check - the
    subject had already seen this pipeline repeatedly report "warm" earlier
    in the same session, so it's corroborating evidence, not a clean
    independent measurement.

Calibration note: _SCLERA_REFERENCE_A/_SCLERA_REFERENCE_B in
season_classifier.py were themselves calibrated FROM the SampleLightDaylight
photo used here (see that constant's comment), after the original
9-photo-batch-derived reference was shown, via this same real-photo
validation, to produce undertone DISAGREEMENT across these three lighting
conditions for this subject. So this test file is also the evidence that
justified that recalibration, not just a check run after the fact - if these
fixtures are ever replaced, rerun scripts/classify_photo_batch.py against
them and reconsider the reference constants, not just this test's pass/fail.

Requires local, non-committed photo fixtures at
backend/tests/fixtures/photos/lighting/ (see .gitignore) and the MediaPipe
model file from backend/scripts/download_models.sh. Skips cleanly when
either is missing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.vision.season_classifier import SeasonClassification, classify_season
from app.vision.skin_sampling import _MODEL_PATH, sample_skin_regions

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "photos" / "lighting"
PHOTO_PATHS = sorted(FIXTURES_DIR.glob("*.jp*g")) if FIXTURES_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    not PHOTO_PATHS or not _MODEL_PATH.exists(),
    reason=(
        "Local lighting-condition photo fixtures or the MediaPipe model aren't available - see "
        "backend/tests/fixtures/photos/lighting/ and backend/scripts/download_models.sh"
    ),
)

_CONDITION_NAMES = ("SampleLightWarm", "SampleLightWhite", "SampleLightDaylight")


def _find_sample(name: str) -> Path | None:
    matches = [p for p in PHOTO_PATHS if p.stem == name]
    return matches[0] if matches else None


def _classify_corrected(photo_path: Path) -> SeasonClassification | None:
    """Runs the full pipeline with sclera correction; returns None if the
    sclera itself wasn't reliably readable, since a check against an
    uncorrected fallback would prove nothing about the correction."""
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


def _classify_raw(photo_path: Path) -> SeasonClassification | None:
    """Runs the pipeline WITHOUT sclera correction - the pre-fix behavior."""
    image_bgr = cv2.imread(str(photo_path))
    sample = sample_skin_regions(image_bgr)
    if not sample.success:
        return None
    result = classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb)
    if not result.success:
        return None
    return result.classification


def _three_condition_paths() -> dict[str, Path]:
    paths = {name: _find_sample(name) for name in _CONDITION_NAMES}
    if any(p is None for p in paths.values()):
        pytest.skip("not all three lighting-condition fixtures are present")
    return paths


def test_undertone_agrees_across_lighting_conditions():
    """(a) Mutual agreement - necessary but not sufficient alone, see module
    docstring."""
    paths = _three_condition_paths()
    classifications = {name: _classify_corrected(path) for name, path in paths.items()}
    if any(c is None for c in classifications.values()):
        pytest.skip("sclera correction unavailable for one or more lighting-condition photos in this environment")

    undertones = {name: c.undertone for name, c in classifications.items()}
    assert len(set(undertones.values())) == 1, (
        f"expected undertone agreement across lighting conditions after sclera correction, got {undertones}"
    )


def test_undertone_matches_daylight_raw_reading():
    """(b) Anchored agreement - each corrected undertone must match
    daylight's own RAW (uncorrected) reading, not just agree with each other
    in the abstract. Catches a systematically-biased correction that makes
    all three confidently agree on the wrong answer, which (a) alone can't."""
    paths = _three_condition_paths()
    classifications = {name: _classify_corrected(path) for name, path in paths.items()}
    if any(c is None for c in classifications.values()):
        pytest.skip("sclera correction unavailable for one or more lighting-condition photos in this environment")

    daylight_raw = _classify_raw(paths["SampleLightDaylight"])
    if daylight_raw is None:
        pytest.skip("could not classify the daylight photo without sclera correction")

    for name, classification in classifications.items():
        assert classification.undertone == daylight_raw.undertone, (
            f"{name}'s corrected undertone ({classification.undertone}) should match daylight's raw, "
            f"uncorrected undertone ({daylight_raw.undertone})"
        )


def test_undertone_diverges_without_correction():
    """Companion to the above: demonstrates the bug this feature fixes - the
    same three photos disagree on undertone BEFORE sclera correction is
    applied. If this ever starts passing in the "agrees" direction (i.e.
    they stop disagreeing), it likely means the fixture photos changed
    rather than that the underlying confound went away - worth a second
    look either way."""
    paths = _three_condition_paths()
    raw_classifications = {name: _classify_raw(path) for name, path in paths.items()}
    if any(c is None for c in raw_classifications.values()):
        pytest.skip("could not classify one or more lighting-condition photos without sclera correction")

    raw_undertones = {name: c.undertone for name, c in raw_classifications.items()}
    assert len(set(raw_undertones.values())) > 1, (
        f"expected raw (uncorrected) undertone to disagree across lighting conditions "
        f"(demonstrating the bug this feature fixes), got unanimous {raw_undertones}"
    )
