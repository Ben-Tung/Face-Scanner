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

import itertools
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
# that same lighting.
#
# Measured as the largest pairwise CIE ΔH* (see hue_difference), in Lab
# units - NOT the raw hue-angle spread this replaced (max(hue) - min(hue) >
# 25 degrees). That angle-only check misfired on a real, evenly-lit photo
# (newdaylight) twice over: it had no 0/360 wraparound (cheeks at 357.1 and
# 8.5 degrees, ~11 apart, read as a 348.6 degree spread), and even wrapped
# correctly it read 43.6 degrees because the skin there was pale/desaturated
# (chroma ~7-10), where hue angle is mostly noise - a 1-unit a*/b* wobble
# swings it ~8 degrees. ΔH* scales the angle by chroma, so near-neutral
# patches can't disagree by much. 10.0 equals the old 25-degree cutoff at
# chroma ~23 (typical skin), so behavior there barely changes. Across the
# 16 color-managed fixture photos the max is 0.9-4.5, except newdaylight
# (SampleEvenLightLowChroma) at 6.75; a real blown-highlight case that must
# still be rejected (see
# test_classify_season_flags_inconsistent_patches_as_low_confidence) is 16.0.
_HUE_DIFFERENCE_THRESHOLD = 10.0

# A forehead patch reading more than this many L* darker than even the
# DARKER cheek is in shadow - a fringe or a hat brim - and is left out of
# the average skin reading. One-sided against the darker cheek on purpose:
# directional side light darkens one cheek and leaves the forehead in
# between, so it never trips this; a forehead darker than both is the
# signature of a shadow cast from above. Measured on color-managed
# fixtures, forehead minus darker cheek reads -8.4 to +33.6 for every other
# subject, and -0.8 / -4.0 for the fringe-wearing test subject's unshaded
# photos versus -13.4 to -29.1 for their shaded ones; 11.0 sits in that gap,
# but on one fringe-wearing subject - forehead_dropped is logged on scan
# events so this can be tuned against real traffic.
#
# Only the AVERAGE drops the forehead. The cross-patch hue check still
# compares all three patches: a forehead far darker than both cheeks can
# also mean both cheeks are the bad patches (see
# test_classify_season_flags_inconsistent_patches_as_low_confidence, where
# two glare-washed cheeks agree with each other but not with the forehead).
_SHADED_FOREHEAD_L_GAP = 11.0

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
#
# Off by default (classify_season's correct_depth, driven by the
# SCLERA_DEPTH_CORRECTION setting). On real phone selfies the sclera sits in
# the shade of the lid and brow, so its L* tracks eye shape and shading
# more than exposure: one subject's six selfies read sclera L* 41-55 against
# this reference, which was averaged mostly from studio-style stock
# portraits. The correction hit the clamp, pushed that subject's depth to
# 88-100, and widened the spread across their photos (14.8 -> 19.4) instead
# of narrowing it. Kept, not deleted, so it can be re-enabled once
# recalibrated against phone selfies.
_SCLERA_REFERENCE_L = 67.0
_SCLERA_CORRECTION_CLAMP_L = 20.0

# Undertone is normalized against the sclera's own a*/b* the same way depth is
# normalized against its L* above: a uniform color-temperature cast (warm
# incandescent vs. cool LED vs. daylight) shifts skin and sclera hue together,
# and neither the clipping checks (nothing is blown out) nor
# _HUE_DIFFERENCE_THRESHOLD (every patch shifts together, so they still agree
# with each other) catch it - see normalize_undertone_ab.
#
# _SCLERA_REFERENCE_A/_SCLERA_REFERENCE_B are the sclera a*/b* of one
# outdoor overcast-daylight photo (SampleLightDaylight), not the mean over
# the 9-photo fixture batch the way _SCLERA_REFERENCE_L is. Overcast
# daylight is the one condition with a real physical claim to being
# color-neutral (diffuse skylight, no artificial-light cast, no indoor-wall
# bounce), while the batch's lighting provenance is unknown and, when
# tried, it reproduced the cross-lighting inconsistency this correction
# exists to fix.
#
# Read color-managed (see image_decode.py). The previous values (3.72, 5.63)
# were this same sclera with the photo's Display P3 pixels misread as sRGB;
# that misread, plus a reference calibrated from it, is what made one
# subject's warm-bulb, white-LED and daylight photos appear to agree on
# "warm". Read correctly, 5 of that subject's 6 photos agree on cool
# (36.7-46.2deg corrected hue), and the warm-bulb photo can't be corrected
# into line by any variant tried (this additive shift, a 30 clamp, or full
# Bradford chromatic adaptation). It's rejected instead - see
# _MAX_COLOR_CAST_AB. Caveat this carries forward: single-subject,
# single-photo calibration. Revisit with more subjects and outdoor photos,
# the same "first-pass, recompute as it grows" spirit as _SCLERA_REFERENCE_L
# - scripts/classify_photo.py on the daylight photo prints the value.
_SCLERA_REFERENCE_A = 4.91
_SCLERA_REFERENCE_B = 6.07

# A photo whose sclera sits this far from the reference in a*/b* has too
# strong a color cast to correct reliably, so it's rejected with a "retake
# in daylight" message rather than classified. Measured as the magnitude of
# the correction vector (see sclera_color_cast), never the skin-to-sclera
# gap. Against the reference above, every accepted fixture photo reads 12.0
# or less (Sample9 12.0, Sample3 11.7, everything else under 9); the one
# warm-incandescent-bulb photo reads 21.9 and stays warm after correction
# while the same subject's other five photos read cool. 16.0 sits between
# the two, but the rejection side rests on that single photo - the
# magnitude is logged on every scan event so this can be tuned against
# real traffic.
_MAX_COLOR_CAST_AB = 16.0

_D65_WHITE = (0.95047, 1.0, 1.08883)  # Xn, Yn, Zn

_SEASON_BY_UNDERTONE_AND_DEPTH: dict[tuple[Undertone, Depth], Season] = {
    ("warm", "light"): "Spring",
    ("cool", "light"): "Summer",
    ("warm", "deep"): "Autumn",
    ("cool", "deep"): "Winter",
}

# Depth when lightness falls inside _DEPTH_AMBIGUITY_BAND, broken on clarity.
# Spring and Winter are the clear seasons, Autumn and Summer the muted ones,
# so the same clarity points to opposite depths on each side of the
# undertone split.
_DEPTH_BY_UNDERTONE_AND_CLARITY: dict[tuple[Undertone, Clarity], Depth] = {
    ("warm", "clear"): "light",  # Spring
    ("warm", "muted"): "deep",  # Autumn
    ("cool", "clear"): "deep",  # Winter
    ("cool", "muted"): "light",  # Summer
}


ClassificationFailureReason = Literal["inconsistent_patches", "color_cast"]


@dataclass(frozen=True)
class SeasonClassification:
    season: Season
    undertone: Undertone
    depth: Depth
    clarity: Clarity
    # Raw average of the patches actually used: all three, or just the
    # cheeks when the forehead was dropped as shaded (see
    # _SHADED_FOREHEAD_L_GAP).
    avg_lab: Lab
    # hue_deg/chroma actually used for the undertone/clarity decision: raw
    # avg_lab-derived when no sclera reading was available, sclera-normalized
    # otherwise (see normalize_undertone_ab). avg_lab itself always stays the
    # raw, uncorrected average.
    hue_deg: float
    chroma: float
    # The lightness value actually used for the depth decision: sclera-
    # normalized (see normalize_depth_lightness) only when depth_corrected,
    # raw avg_lab[0] otherwise. avg_lab itself always stays the raw,
    # uncorrected average.
    depth_lightness: float
    # Whether depth_lightness was sclera-normalized: correct_depth was on
    # AND a sclera reading was available.
    depth_corrected: bool = False


@dataclass(frozen=True)
class SeasonClassificationResult:
    success: bool
    classification: SeasonClassification | None = None
    error: ClassificationFailureReason | None = None
    # Largest pairwise ΔH* across the three raw patches (see
    # hue_difference), populated on success AND on an inconsistent_patches
    # failure, so real scans can be logged against
    # _HUE_DIFFERENCE_THRESHOLD to keep tuning it.
    max_hue_difference: float | None = None
    # This photo's lighting cast (see sclera_color_cast), whenever a sclera
    # reading was available - success or failure - logged against
    # _MAX_COLOR_CAST_AB the same way.
    color_cast: float | None = None
    # Whether the forehead patch was left out of the average as shaded (see
    # _SHADED_FOREHEAD_L_GAP), success or failure, logged for tuning.
    forehead_dropped: bool = False


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
    # defaults to "cool"). Real skin does get close: pale, desaturated
    # patches in daylight have read chroma ~7, where a 1-unit a*/b* wobble
    # swings hue ~8 degrees - so compare patches with hue_difference, never
    # by raw hue angle.
    hue_deg = math.degrees(math.atan2(b, a)) % 360
    chroma = math.hypot(a, b)
    return hue_deg, chroma


def hue_difference(lab1: Lab, lab2: Lab) -> float:
    """CIE ΔH*: the hue component of the color difference between two Lab
    colors, in Lab units (the same scale as ΔE*ab).

    2 * sqrt(C1 * C2) * |sin(Δh / 2)| - a hue-angle gap weighted by both
    colors' chroma, so two near-neutral colors can't disagree by much even
    when their raw hue angles look far apart (hue is meaningless at ~0
    chroma), and a gap straddling 0/360 is measured the short way round.
    Symmetric, and 0 for identical hues.
    """
    hue1, chroma1 = hue_and_chroma(lab1)
    hue2, chroma2 = hue_and_chroma(lab2)
    delta_hue_deg = (hue2 - hue1 + 180) % 360 - 180
    return 2 * math.sqrt(chroma1 * chroma2) * abs(math.sin(math.radians(delta_hue_deg) / 2))


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


def normalize_undertone_ab(skin_a: float, skin_b: float, sclera_a: float, sclera_b: float) -> tuple[float, float]:
    """Re-express skin (a*, b*) relative to this photo's own sclera reading.

    Same conceptual move as normalize_depth_lightness, generalized from a 1D
    scalar (L*) to a 2D vector (a*, b*): a uniform color-temperature cast
    shifts the whole photo's white balance, so skin (a, b) and sclera (a, b)
    are expected to shift by roughly the same vector. The sclera - expected to
    read close to (_SCLERA_REFERENCE_A, _SCLERA_REFERENCE_B) under
    color-neutral light, not necessarily literal (0, 0) - see that constant's
    comment - estimates that shared shift and cancels it back out of skin
    (a, b), the same way sclera L* estimates and cancels a shared exposure
    shift for depth. A no-op when sclera (a, b) already equals the reference.

    Unclamped: casts too strong to trust are rejected before this runs (see
    sclera_color_cast and _MAX_COLOR_CAST_AB). Nor is the result clamped to
    a fixed output range afterward - a*/b* have no natural bound anywhere in
    this module (hue_and_chroma accepts any real-valued input), unlike L*'s
    physical 0-100 range.
    """
    # correction is (reference - sclera), a pure read on how atypical THIS
    # PHOTO's lighting cast was - not (skin - sclera), which mixes in this
    # PERSON's actual coloring and has nothing to do with lighting.
    correction_a = _SCLERA_REFERENCE_A - sclera_a
    correction_b = _SCLERA_REFERENCE_B - sclera_b
    return skin_a + correction_a, skin_b + correction_b


def sclera_color_cast(sclera_lab: Lab) -> float:
    """How strong a color cast this photo's lighting has, in a*/b* units:
    the magnitude of the correction normalize_undertone_ab would apply.

    Judged from the sclera's own deviation from the reference alone - NEVER
    from the skin-to-sclera gap (skin_a - sclera_a, skin_b - sclera_b). That
    gap is expected to be large for genuinely warm, high-chroma skin even
    under perfectly neutral light, so judging it would reject (or, as the
    clamp this replaced risked, flatten) exactly the warmest, most saturated
    skin tones - the same failure mode normalize_depth_lightness's clamp
    avoids on the lightness axis. Concretely: skin at (35.0, 45.0) under a
    sclera reading exactly (_SCLERA_REFERENCE_A, _SCLERA_REFERENCE_B) has a
    cast of 0, however far the skin sits from the sclera.
    """
    _, sclera_a, sclera_b = sclera_lab
    return math.hypot(_SCLERA_REFERENCE_A - sclera_a, _SCLERA_REFERENCE_B - sclera_b)


def classify_season(
    forehead_rgb: RGB,
    left_cheek_rgb: RGB,
    right_cheek_rgb: RGB,
    sclera_rgb: RGB | None = None,
    *,
    correct_depth: bool = False,
) -> SeasonClassificationResult:
    """Classify a face's color season from three sampled skin-patch RGBs.

    Takes plain RGB tuples rather than a `SkinSampleResult` so it stays
    decoupled from the sampling module and trivially unit-testable. Callers
    are responsible for checking `SkinSampleResult.success` first.

    Before averaging, checks the three patches' hue against each other —
    patches where any pair disagrees in color (not just brightness) by more
    than `_HUE_DIFFERENCE_THRESHOLD` (as ΔH*, see `hue_difference`) aren't
    read as one skin tone, so this returns a failure instead of silently
    averaging incompatible readings.
    Ordinary directional lighting is expected to make patches disagree on
    brightness; it isn't a sign of a bad sample on its own. The exception is
    a forehead far darker than both cheeks - shaded from above, by a fringe
    or hat brim - which is left out of the average (but not out of the hue
    check; see `_SHADED_FOREHEAD_L_GAP`).

    `sclera_rgb`, when provided, normalizes the undertone/clarity decision
    (`normalize_undertone_ab`) against this photo's own lighting, and the
    depth decision too (`normalize_depth_lightness`) when `correct_depth` is
    on - off by default, see `_SCLERA_REFERENCE_L`. Both fall back to the raw
    averaged Lab when no sclera reading is available. The cross-patch hue
    check above runs on each raw patch's own Lab, so it's unaffected either
    way: a uniform color cast shifts all three patches together and doesn't
    change their agreement with each other, which is exactly why that check
    can't catch it and a separate sclera-based correction is needed.

    Inside `_DEPTH_AMBIGUITY_BAND`, depth is broken on clarity per undertone
    (see `_DEPTH_BY_UNDERTONE_AND_CLARITY`).

    A cast too strong to correct (see `sclera_color_cast`) fails as
    `color_cast`, checked before the hue check: dragging patches can fix
    mixed patch readings, but nothing but a retake fixes the lighting.
    """
    labs = [rgb_to_lab(forehead_rgb), rgb_to_lab(left_cheek_rgb), rgb_to_lab(right_cheek_rgb)]
    forehead_lab, *cheek_labs = labs
    max_hue_difference = max(hue_difference(a, b) for a, b in itertools.combinations(labs, 2))
    sclera_lab = rgb_to_lab(sclera_rgb) if sclera_rgb is not None else None
    color_cast = sclera_color_cast(sclera_lab) if sclera_lab is not None else None
    forehead_dropped = forehead_lab[0] < min(lab[0] for lab in cheek_labs) - _SHADED_FOREHEAD_L_GAP
    measurements = {
        "max_hue_difference": max_hue_difference,
        "color_cast": color_cast,
        "forehead_dropped": forehead_dropped,
    }

    if color_cast is not None and color_cast > _MAX_COLOR_CAST_AB:
        return SeasonClassificationResult(success=False, error="color_cast", **measurements)
    if max_hue_difference > _HUE_DIFFERENCE_THRESHOLD:
        return SeasonClassificationResult(success=False, error="inconsistent_patches", **measurements)

    avg_lab = _average_lab(cheek_labs if forehead_dropped else labs)

    if sclera_lab is not None:
        corrected_a, corrected_b = normalize_undertone_ab(avg_lab[1], avg_lab[2], sclera_lab[1], sclera_lab[2])
        hue_deg, chroma = hue_and_chroma((avg_lab[0], corrected_a, corrected_b))
    else:
        hue_deg, chroma = hue_and_chroma(avg_lab)

    depth_corrected = correct_depth and sclera_lab is not None
    depth_lightness = normalize_depth_lightness(avg_lab[0], sclera_lab[0]) if depth_corrected else avg_lab[0]

    undertone: Undertone = "warm" if hue_deg >= _HUE_WARM_COOL_THRESHOLD_DEG else "cool"
    clarity: Clarity = "clear" if chroma >= _CHROMA_CLEAR_MUTED_THRESHOLD else "muted"

    if abs(depth_lightness - _DEPTH_LIGHT_DEEP_THRESHOLD_L) <= _DEPTH_AMBIGUITY_BAND:
        depth: Depth = _DEPTH_BY_UNDERTONE_AND_CLARITY[(undertone, clarity)]
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
        depth_corrected=depth_corrected,
    )
    return SeasonClassificationResult(success=True, classification=classification, **measurements)
