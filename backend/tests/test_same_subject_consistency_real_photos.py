"""Same person, different photos -> same result.

The goal the classifier's lighting corrections are tuned for (consistency
for a given user beats any single photo's theoretical accuracy), checked
directly: every photo below is the same subject, taken across different
days, rooms and lighting - a warm incandescent bulb, a "white" LED, outdoor
overcast daylight, even indoor light, and two everyday indoor selfies. Each
one must either classify, or be rejected as color_cast (lighting too
strongly tinted to correct - see season_classifier._MAX_COLOR_CAST_AB);
every photo that classifies must land on the same undertone and season.

Before color management (see image_decode.py) these same photos split 3
Spring / 3 Summer. With it, 5 agree on cool/Summer and the warm-bulb photo
is the one rejected. Like the other real-photo suites, this is evidence
about consistency for one subject, not proof of absolute correctness - no
labeled ground-truth dataset exists for this app.

Requires local, non-committed photo fixtures under backend/tests/fixtures/
photos/ (lighting/, consistency/ and same_subject/ - see .gitignore) and the
MediaPipe model file from backend/scripts/download_models.sh. Skips cleanly
when either is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.vision.image_decode import decode_image
from app.vision.season_classifier import SeasonClassificationResult, classify_season
from app.vision.skin_sampling import _MODEL_PATH, sample_skin_regions

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "photos"
SAME_SUBJECT_PHOTOS = [
    FIXTURES_DIR / "lighting" / "SampleLightDaylight.jpeg",
    FIXTURES_DIR / "lighting" / "SampleLightWhite.jpeg",
    FIXTURES_DIR / "lighting" / "SampleLightWarm.jpeg",
    FIXTURES_DIR / "consistency" / "SampleEvenLightLowChroma.jpeg",
    FIXTURES_DIR / "same_subject" / "SampleSameSubject1.jpeg",
    FIXTURES_DIR / "same_subject" / "SampleSameSubject2.jpeg",
]
_MIN_CLASSIFIED = 4

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in SAME_SUBJECT_PHOTOS) or not _MODEL_PATH.exists(),
    reason=(
        "Local same-subject photo fixtures or the MediaPipe model aren't available - see "
        "backend/tests/fixtures/photos/ and backend/scripts/download_models.sh"
    ),
)


def _classify(photo_path: Path) -> SeasonClassificationResult:
    sample = sample_skin_regions(decode_image(photo_path.read_bytes()))
    assert sample.success, f"{photo_path.stem}: sampling failed ({sample.error})"
    sclera_rgb = sample.sclera.sclera_rgb if sample.sclera is not None and sample.sclera.success else None
    return classify_season(sample.forehead_rgb, sample.left_cheek_rgb, sample.right_cheek_rgb, sclera_rgb=sclera_rgb)


def test_same_subject_gets_the_same_result_across_photos():
    results = {path.stem: _classify(path) for path in SAME_SUBJECT_PHOTOS}

    unexpected_failures = {
        name: (r.error, r.max_hue_difference, r.color_cast)
        for name, r in results.items()
        if not r.success and r.error != "color_cast"
    }
    assert not unexpected_failures, f"expected each photo to classify or be rejected as color_cast: {unexpected_failures}"

    classified = {name: r.classification for name, r in results.items() if r.success}
    assert len(classified) >= _MIN_CLASSIFIED, (
        f"only {len(classified)} of {len(results)} photos classified - too few to check consistency"
    )

    readings = {name: (c.undertone, c.season, round(c.hue_deg, 1)) for name, c in classified.items()}
    assert len({c.undertone for c in classified.values()}) == 1, f"undertone disagrees across photos: {readings}"
    assert len({c.season for c in classified.values()}) == 1, f"season disagrees across photos: {readings}"
