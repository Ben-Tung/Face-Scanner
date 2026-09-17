"""Deterministic Lab-based color season classification.

No LLM involved anywhere in this module — see CLAUDE.md's classifier rule.
Consumes the three sampled skin-patch RGB triples from skin_sampling.py and
returns a rule-based season classification (Spring/Summer/Autumn/Winter).
Pure math, no image/model dependency, so it's trivially unit-testable with
hand-picked RGB values.

Thresholds were calibrated against a synthetic Lab-space grid (fixed chroma,
hue swept across six depth levels from very fair to very deep, round-tripped
through sRGB to confirm plausibility) rather than a labeled photo dataset —
none exists yet. They're grouped here so they're easy to retune later
against real user feedback without touching the classification logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from app.vision.skin_sampling import RGB

Season = Literal["Spring", "Summer", "Autumn", "Winter"]
Undertone = Literal["warm", "cool"]
Depth = Literal["light", "deep"]
Clarity = Literal["clear", "muted"]
Lab = tuple[float, float, float]  # (L*, a*, b*) in real CIE Lab units, D65 white point

_HUE_WARM_COOL_THRESHOLD_DEG = 50.0
_DEPTH_LIGHT_DEEP_THRESHOLD_L = 58.0
_DEPTH_AMBIGUITY_BAND = 5.0
_CHROMA_CLEAR_MUTED_THRESHOLD = 25.0
# Cross-patch consistency is checked on HUE, not brightness (L*). A single
# directional light (common in ordinary selfies/portraits) routinely swings
# L* by 25-35 between a well-lit forehead and a shadowed cheek without
# changing the actual skin color much - brightness is expected to vary with
# lighting. Hue is what determines undertone and stays far more stable under
# that same lighting: across a real batch of 10 photos (including one with
# a strong single-direction light and shiny/glare-prone skin), hue spread
# topped out at 11.9 degrees even where L* spread hit 33.6. 25 sits well
# clear of that, so patches disagreeing by more than this aren't reading the
# same underlying skin color (e.g. hair, a colored light cast, or a patch
# whose chroma is too low for hue to mean anything - see hue_and_chroma).
_HUE_SPREAD_THRESHOLD_DEG = 25.0

# Depth is normalized against the sclera (whites of the eyes) when a reading
# is available (see skin_sampling.sample_sclera and normalize_depth_lightness
# below), since raw skin L* alone is confounded with each photo's own
# lighting/exposure - two subjects with very different real depth can read
# nearly identical raw L* if one photo happens to be a brighter, more evenly
# lit shot. _SCLERA_REFERENCE_L is the mean sclera L* observed across the 9
# (of 10) real fixture photos with a successful sclera reading under that
# sampling recipe - a first-pass value calibrated against that one small,
# unlabeled batch, not a rigorously validated constant; recompute it (and
# reconfirm the pairwise-ordering checks) with scripts/classify_photo_batch.py
# whenever the fixture batch changes, and refine it against a larger set over
# time. _SCLERA_CORRECTION_CLAMP_L caps the correction (how far a photo's own
# exposure is allowed to shift skin_l - see normalize_depth_lightness, NOT
# the skin-to-sclera contrast itself) so a still-somewhat-unreliable sclera
# reading can't swing the depth decision by an implausible amount. It's
# already binding on the most extreme real sample in this batch (a bright,
# evenly-lit photo whose sclera reads ~22 above the reference, clamped down
# to the +-20 cap) rather than sitting comfortably above every observed
# correction - worth widening (~22-25) next time this gets recalibrated.
_SCLERA_REFERENCE_L = 67.0
_SCLERA_CORRECTION_CLAMP_L = 20.0

_D65_WHITE = (0.95047, 1.0, 1.08883)  # Xn, Yn, Zn

_SEASON_BY_UNDERTONE_AND_DEPTH: dict[tuple[Undertone, Depth], Season] = {
    ("warm", "light"): "Spring",
    ("cool", "light"): "Summer",
    ("warm", "deep"): "Autumn",
    ("cool", "deep"): "Winter",
}


ClassificationFailureReason = Literal["inconsistent_patches"]


@dataclass(frozen=True)
class SeasonClassification:
    season: Season
    undertone: Undertone
    depth: Depth
    clarity: Clarity
    avg_lab: Lab
    hue_deg: float
    chroma: float
    # The lightness value actually used for the depth decision: raw avg_lab[0]
    # when no sclera reading was available, sclera-normalized otherwise (see
    # normalize_depth_lightness). avg_lab itself always stays the raw,
    # uncorrected average.
    depth_lightness: float


@dataclass(frozen=True)
class SeasonClassificationResult:
    success: bool
    classification: SeasonClassification | None = None
    error: ClassificationFailureReason | None = None


def _srgb_to_linear(channel: int) -> float:
    c = channel / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _f(t: float) -> float:
    delta = 6.0 / 29.0
    if t > delta**3:
        return t ** (1.0 / 3.0)
    return t / (3 * delta**2) + 4.0 / 29.0


def rgb_to_lab(rgb: RGB) -> Lab:
    """Convert an 8-bit sRGB triple to real-unit CIE Lab (D65 white point).

    L* is 0-100; a*/b* are signed (roughly -128..127 for real colors). This
    is standard sRGB -> linear -> XYZ -> Lab math, done in pure `math` since
    the input here is a single scalar triple, not an image array.
    """
    r, g, b = (_srgb_to_linear(c) for c in rgb)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    xn, yn, zn = _D65_WHITE
    fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)

    lightness = 116 * fy - 16
    a_axis = 500 * (fx - fy)
    b_axis = 200 * (fy - fz)
    return (lightness, a_axis, b_axis)


def _average_lab(labs: list[Lab]) -> Lab:
    n = len(labs)
    return (
        sum(lab[0] for lab in labs) / n,
        sum(lab[1] for lab in labs) / n,
        sum(lab[2] for lab in labs) / n,
    )


def hue_and_chroma(lab: Lab) -> tuple[float, float]:
    _, a, b = lab
    # Near-zero chroma makes hue numerically unstable (atan2(0,0)=0, i.e.
    # defaults to "cool") — a non-issue in practice since real skin never
    # averages to ~0 chroma.
    hue_deg = math.degrees(math.atan2(b, a)) % 360
    chroma = math.hypot(a, b)
    return hue_deg, chroma


def normalize_depth_lightness(skin_l: float, sclera_l: float) -> float:
    """Re-express skin lightness relative to this photo's own sclera reading.

    A photo's exposure/lighting is expected to shift both skin_l and sclera_l
    by roughly the same amount, so a `correction` of `_SCLERA_REFERENCE_L -
    sclera_l` estimates that shared shift, and adding it to skin_l cancels it
    back out - re-anchoring onto `_SCLERA_REFERENCE_L` expresses the result
    on the original L* scale, so `_DEPTH_LIGHT_DEEP_THRESHOLD_L`/
    `_DEPTH_AMBIGUITY_BAND` keep applying unchanged. A no-op when
    `sclera_l == _SCLERA_REFERENCE_L` (i.e. "typical" exposure for the batch
    that constant was derived from).

    `_SCLERA_CORRECTION_CLAMP_L` bounds `correction` itself (how far this
    photo's exposure is allowed to shift skin_l) - NOT `skin_l - sclera_l`
    (the skin-to-sclera contrast), which is deliberately left unclamped:
    that contrast is expected to be large for genuinely deep skin tones even
    under perfect lighting, and clamping it would cap how deep the
    classifier could ever read someone. Concretely: normalize_depth_lightness
    (25.0, 67.0) - very deep skin under exactly reference-quality lighting,
    sclera_l == _SCLERA_REFERENCE_L - correctly returns 25.0 unchanged, even
    though skin_l - sclera_l is -42, far past the clamp. If the clamp instead
    bounded skin_l - sclera_l directly, that same input would incorrectly
    get dragged up to 47.0 (67.0 - 20.0) - not because anything was wrong
    with the photo's lighting, but purely because this person's skin is
    naturally much darker than their sclera. That's the failure mode this
    function exists to avoid re-introducing by another route.
    """
    # correction is (reference - sclera_l), a pure read on how atypical THIS
    # PHOTO's exposure was - not (skin_l - sclera_l), which mixes in this
    # PERSON's actual skin tone and has nothing to do with lighting. Clamping
    # must happen here, on correction alone, before skin_l ever enters the
    # expression - clamping skin_l + correction as a whole (or equivalently
    # skin_l - sclera_l) would instead cap real skin-to-sclera contrast,
    # silently flattening deep skin tones toward "light" regardless of
    # lighting quality - see the docstring's worked example.
    correction = _SCLERA_REFERENCE_L - sclera_l
    correction = max(-_SCLERA_CORRECTION_CLAMP_L, min(_SCLERA_CORRECTION_CLAMP_L, correction))
    return max(0.0, min(100.0, skin_l + correction))


def classify_season(
    forehead_rgb: RGB,
    left_cheek_rgb: RGB,
    right_cheek_rgb: RGB,
    sclera_rgb: RGB | None = None,
) -> SeasonClassificationResult:
    """Classify a face's color season from three sampled skin-patch RGBs.

    Takes plain RGB tuples rather than a `SkinSampleResult` so it stays
    decoupled from the sampling module and trivially unit-testable. Callers
    are responsible for checking `SkinSampleResult.success` first.

    Before averaging, checks the three patches' hue against each other —
    patches that disagree in color (not just brightness) by more than
    `_HUE_SPREAD_THRESHOLD_DEG` aren't read as one skin tone, so this
    returns a failure instead of silently averaging incompatible readings.
    Ordinary directional lighting is expected to make patches disagree on
    brightness; it isn't a sign of a bad sample on its own.

    `sclera_rgb`, when provided, normalizes the depth decision against this
    photo's own lighting via `normalize_depth_lightness` — see that function.
    Undertone and clarity are always computed from the raw averaged Lab
    either way; only the depth axis is affected, since hue (unlike L*) is
    already stable under ordinary lighting variance (see
    `_HUE_SPREAD_THRESHOLD_DEG` above).
    """
    labs = [rgb_to_lab(forehead_rgb), rgb_to_lab(left_cheek_rgb), rgb_to_lab(right_cheek_rgb)]
    hue_values = [hue_and_chroma(lab)[0] for lab in labs]
    if max(hue_values) - min(hue_values) > _HUE_SPREAD_THRESHOLD_DEG:
        return SeasonClassificationResult(success=False, error="inconsistent_patches")

    avg_lab = _average_lab(labs)
    hue_deg, chroma = hue_and_chroma(avg_lab)

    undertone: Undertone = "warm" if hue_deg >= _HUE_WARM_COOL_THRESHOLD_DEG else "cool"
    clarity: Clarity = "clear" if chroma >= _CHROMA_CLEAR_MUTED_THRESHOLD else "muted"

    if sclera_rgb is not None:
        sclera_l = rgb_to_lab(sclera_rgb)[0]
        depth_lightness = normalize_depth_lightness(avg_lab[0], sclera_l)
    else:
        depth_lightness = avg_lab[0]

    if abs(depth_lightness - _DEPTH_LIGHT_DEEP_THRESHOLD_L) <= _DEPTH_AMBIGUITY_BAND:
        depth: Depth = "light" if clarity == "clear" else "deep"
    else:
        depth = "light" if depth_lightness >= _DEPTH_LIGHT_DEEP_THRESHOLD_L else "deep"

    season = _SEASON_BY_UNDERTONE_AND_DEPTH[(undertone, depth)]

    classification = SeasonClassification(
        season=season,
        undertone=undertone,
        depth=depth,
        clarity=clarity,
        avg_lab=avg_lab,
        hue_deg=hue_deg,
        chroma=chroma,
        depth_lightness=depth_lightness,
    )
    return SeasonClassificationResult(success=True, classification=classification)
