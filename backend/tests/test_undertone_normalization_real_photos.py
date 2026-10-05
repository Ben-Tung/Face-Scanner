"""Sanity check: sclera-corrected undertone should agree across different
lighting conditions on the same face/pose, where raw (uncorrected) hue does
not - see season_classifier.normalize_undertone_ab. A cast too strong to
correct should be rejected (color_cast) rather than classified.

This is NOT proof of absolute correctness - no labeled ground-truth season
dataset exists for this app (see season_classifier.py's module docstring).
The fixtures are one subject under three conditions: a warm incandescent
bulb, a "white" LED, and outdoor overcast daylight. Checks below, each
necessary but not individually sufficient alone:

(a) Mutual agreement: corrected undertone must be the same across the
    conditions the pipeline accepts (white LED, daylight). This alone isn't
    enough - a systematically biased correction could make them confidently
    agree on the WRONG undertone.
(b) Anchored agreement: each accepted condition's corrected undertone must
    also match the raw, uncorrected undertone read from the daylight photo
    specifically. Daylight is the one condition with a physical claim to
    being close to color-neutral (diffuse outdoor skylight, no artificial-
    light cast, no indoor-wall bounce), so its raw reading is the best
    available proxy for ground truth without a labeled dataset. This is what
    would catch "converged, but on the wrong answer together," which (a)
    alone cannot.
(c) The warm-bulb photo is rejected as color_cast: its sclera reads a cast
    of ~21.9 (every other fixture photo is 12.0 or less), and no correction
    tried - this additive shift, a wider clamp, or full Bradford chromatic
    adaptation - brought it into line with the other two (it stays warm,
    ~54-60deg, while they read cool).

Human judgment, recorded rather than asserted: the subject self-assessed
their undertone as WARM (jewelry/vein-color check). That was never a blind
check - they'd watched this pipeline report "warm" repeatedly beforehand -
and, it turned out, that "warm" reading was an artifact: these iPhone photos
carry a Display P3 color profile that was being misread as sRGB, and the
a*/b* reference was calibrated from the same misread. Read color-managed
(see image_decode.py), daylight's raw hue is ~46deg - cool, about 4deg under
the 50deg warm/cool line - and 5 of this subject's 6 photos agree on cool
(see test_same_subject_consistency_real_photos.py). The self-assessment and
the pipeline now disagree; the 50deg threshold itself was only calibrated
against a synthetic Lab grid, so this subject sitting near it is expected
to be sensitive to small changes. Revisit with labeled subjects.

Calibration note: _SCLERA_REFERENCE_A/_SCLERA_REFERENCE_B in
season_classifier.py are calibrated FROM the SampleLightDaylight photo used
here (see that constant's comment). If these fixtures are ever replaced,
rerun scripts/classify_photo.py / classify_photo_batch.py against them and
reconsider the reference constants, not just this test's pass/fail.

Requires local, non-committed photo fixtures at
backend/tests/fixtures/photos/lighting/ (see .gitignore) and the MediaPipe
model file from backend/scripts/download_models.sh. Skips cleanly when
either is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.vision.image_decode import decode_image
from app.vision.season_classifier import SeasonClassification, SeasonClassificationResult, classify_season
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
_ACCEPTED_CONDITION_NAMES = ("SampleLightWhite", "SampleLightDaylight")


def _find_sample(name: str) -> Path | None:
    matches = [p for p in PHOTO_PATHS if p.stem == name]
    return matches[0] if matches else None


def _condition_paths(names: tuple[str, ...]) -> dict[str, Path]:
    paths = {name: _find_sample(name) for name in names}
    if any(p is None for p in paths.values()):
        pytest.skip("not all lighting-condition fixtures are present")
    return paths


def _classify_corrected(photo_path: Path) -> SeasonClassificationResult:
    """Runs the full pipeline with sclera correction. Skips if the sclera
    itself wasn't reliably readable, since a check against an uncorrected
    fallback would prove nothing about the correction - but returns the
    result as-is on a classification failure, so a rejection can't be
    mistaken for "correction unavailable"."""
    sample = sample_skin_regions(decode_image(photo_path.read_bytes()))
    assert sample.success, f"{photo_path.stem}: sampling failed ({sample.error})"
    if sample.sclera is None or not sample.sclera.success:
        pytest.skip(f"sclera unreadable for {photo_path.stem} in this environment")
    return classify_season(
        sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb, sclera_rgb=sample.sclera.sclera_rgb
    )


def _classify_accepted(photo_path: Path) -> SeasonClassification:
    result = _classify_corrected(photo_path)
    assert result.success, f"{photo_path.stem} was rejected as {result.error} (color cast {result.color_cast})"
    return result.classification


def _classify_raw(photo_path: Path) -> SeasonClassification | None:
    """Runs the pipeline WITHOUT sclera correction - the pre-fix behavior."""
    sample = sample_skin_regions(decode_image(photo_path.read_bytes()))
    if not sample.success:
        return None
    result = classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb)
    if not result.success:
        return None
    return result.classification


def test_undertone_agrees_across_accepted_lighting_conditions():
    """(a) Mutual agreement - necessary but not sufficient alone, see module
    docstring."""
    paths = _condition_paths(_ACCEPTED_CONDITION_NAMES)
    undertones = {name: _classify_accepted(path).undertone for name, path in paths.items()}

    assert len(set(undertones.values())) == 1, (
        f"expected undertone agreement across lighting conditions after sclera correction, got {undertones}"
    )


def test_undertone_matches_daylight_raw_reading():
    """(b) Anchored agreement - each accepted condition's corrected undertone
    must match daylight's own RAW (uncorrected) reading, not just agree with
    the others in the abstract. Catches a systematically-biased correction
    that makes them confidently agree on the wrong answer, which (a) alone
    can't."""
    paths = _condition_paths(_ACCEPTED_CONDITION_NAMES)
    daylight_raw = _classify_raw(paths["SampleLightDaylight"])
    if daylight_raw is None:
        pytest.skip("could not classify the daylight photo without sclera correction")

    for name, path in paths.items():
        classification = _classify_accepted(path)
        assert classification.undertone == daylight_raw.undertone, (
            f"{name}'s corrected undertone ({classification.undertone}) should match daylight's raw, "
            f"uncorrected undertone ({daylight_raw.undertone})"
        )


def test_warm_bulb_photo_is_rejected_as_color_cast():
    """(c) A cast this strong can't be corrected into line, so it must be
    rejected with a retake-in-daylight message rather than classified."""
    paths = _condition_paths(("SampleLightWarm",))

    result = _classify_corrected(paths["SampleLightWarm"])

    assert result.success is False
    assert result.error == "color_cast"


def test_undertone_diverges_without_correction():
    """Companion to the above: demonstrates the confound the correction
    addresses - the same three photos disagree on undertone BEFORE sclera
    correction is applied. If this ever starts passing in the "agrees"
    direction (i.e. they stop disagreeing), it likely means the fixture
    photos changed rather than that the underlying confound went away -
    worth a second look either way."""
    paths = _condition_paths(_CONDITION_NAMES)
    raw_classifications = {name: _classify_raw(path) for name, path in paths.items()}
    if any(c is None for c in raw_classifications.values()):
        pytest.skip("could not classify one or more lighting-condition photos without sclera correction")

    raw_undertones = {name: c.undertone for name, c in raw_classifications.items()}
    assert len(set(raw_undertones.values())) > 1, (
        f"expected raw (uncorrected) undertone to disagree across lighting conditions "
        f"(demonstrating the confound the correction addresses), got unanimous {raw_undertones}"
    )
