import math

import pytest

from app.vision.season_classifier import (
    _HUE_DIFFERENCE_THRESHOLD,
    _MAX_COLOR_CAST_AB,
    _SCLERA_REFERENCE_A,
    _SCLERA_REFERENCE_B,
    classify_season,
    hue_and_chroma,
    hue_difference,
    normalize_depth_lightness,
    normalize_undertone_ab,
    rgb_to_lab,
    sclera_color_cast,
)


def test_rgb_to_lab_matches_known_reference_values():
    white = rgb_to_lab((255, 255, 255))
    black = rgb_to_lab((0, 0, 0))

    assert white == pytest.approx((100.0, 0.0, 0.0), abs=0.01)
    assert black == pytest.approx((0.0, 0.0, 0.0), abs=0.01)


def test_classify_season_summer_cool_light():
    result = classify_season((216, 165, 152), (216, 165, 152), (216, 165, 152))

    assert result.success is True
    classification = result.classification
    assert classification.season == "Summer"
    assert classification.undertone == "cool"
    assert classification.depth == "light"
    assert classification.clarity == "muted"


def test_classify_season_spring_warm_light():
    result = classify_season((208, 169, 142), (208, 169, 142), (208, 169, 142))

    assert result.success is True
    classification = result.classification
    assert classification.season == "Spring"
    assert classification.undertone == "warm"
    assert classification.depth == "light"
    assert classification.clarity == "muted"


def test_classify_season_winter_cool_deep():
    result = classify_season((133, 88, 77), (133, 88, 77), (133, 88, 77))

    assert result.success is True
    classification = result.classification
    assert classification.season == "Winter"
    assert classification.undertone == "cool"
    assert classification.depth == "deep"
    assert classification.clarity == "muted"


def test_classify_season_autumn_warm_deep():
    result = classify_season((126, 92, 68), (126, 92, 68), (126, 92, 68))

    assert result.success is True
    classification = result.classification
    assert classification.season == "Autumn"
    assert classification.undertone == "warm"
    assert classification.depth == "deep"
    assert classification.clarity == "muted"


def test_classify_season_ambiguous_depth_high_chroma_breaks_toward_light():
    result = classify_season((194, 120, 102), (194, 120, 102), (194, 120, 102))

    assert result.success is True
    classification = result.classification
    assert classification.clarity == "clear"
    assert classification.depth == "light"
    assert classification.season == "Summer"


def test_classify_season_ambiguous_depth_low_chroma_breaks_toward_deep():
    result = classify_season((165, 132, 123), (165, 132, 123), (165, 132, 123))

    assert result.success is True
    classification = result.classification
    assert classification.clarity == "muted"
    assert classification.depth == "deep"
    assert classification.season == "Winter"


def test_classify_season_warm_side_boundary_pair():
    clear_result = classify_season((182, 128, 86), (182, 128, 86), (182, 128, 86))
    muted_result = classify_season((160, 134, 117), (160, 134, 117), (160, 134, 117))

    assert clear_result.success is True
    assert clear_result.classification.season == "Spring"
    assert clear_result.classification.clarity == "clear"

    assert muted_result.success is True
    assert muted_result.classification.season == "Autumn"
    assert muted_result.classification.clarity == "muted"


def test_classify_season_averages_across_differing_patches():
    result = classify_season(
        forehead_rgb=(216, 165, 152),
        left_cheek_rgb=(200, 150, 140),
        right_cheek_rgb=(230, 180, 165),
    )

    assert result.success is True
    classification = result.classification
    assert classification.avg_lab == pytest.approx((71.99, 16.77, 13.73), abs=0.01)
    assert classification.season == "Summer"
    assert classification.undertone == "cool"
    assert classification.depth == "light"


def test_classify_season_flags_inconsistent_patches_as_low_confidence():
    # Real bug case: a blown highlight on the right cheek pegged its R
    # channel at 255, pulling that patch toward white and its hue wildly
    # off the other two - a max pairwise ΔH* of ~16.0 (the cheeks sit 68
    # degrees of hue apart at real chroma), not just a brighter patch.
    result = classify_season(
        forehead_rgb=(123, 102, 99),
        left_cheek_rgb=(193, 185, 144),
        right_cheek_rgb=(255, 247, 227),
    )

    assert result.success is False
    assert result.error == "inconsistent_patches"
    assert result.classification is None


def test_classify_season_tolerates_natural_patch_variation():
    # Mild, realistic lighting variation across forehead/cheeks (~4.6 L*
    # spread, max pairwise ΔH* ~0.7) should still classify - the
    # consistency check shouldn't be so strict it rejects normal photos.
    result = classify_season(
        forehead_rgb=(216, 165, 152),
        left_cheek_rgb=(208, 158, 145),
        right_cheek_rgb=(222, 170, 158),
    )

    assert result.success is True
    assert result.classification is not None


def test_classify_season_tolerates_low_chroma_hue_wraparound():
    # Real photo (newdaylight): evenly lit, pale/desaturated skin (chroma
    # ~7-10). Patch hues read 40.7 / 357.1 / 8.5 degrees - the old
    # max(hue) - min(hue) check called that a 348.6 degree spread (no
    # 0/360 wraparound), and even wrapped it's 43.6 degrees, because hue
    # angle is mostly noise at this chroma. Max pairwise ΔH* is only ~6.07.
    result = classify_season(
        forehead_rgb=(146, 124, 118),
        left_cheek_rgb=(220, 203, 208),
        right_cheek_rgb=(225, 204, 207),
    )

    assert result.success is True
    assert result.max_hue_difference == pytest.approx(6.07, abs=0.01)


def test_classify_season_flags_hue_disagreement_at_typical_chroma():
    # Guards against the ΔH* check simply being looser than the angle check
    # it replaced: at ordinary skin chroma (~27), a ~47 degree hue gap
    # between the forehead and one cheek (max pairwise ΔH* ~21.5) must still
    # be rejected.
    result = classify_season(
        forehead_rgb=(200, 150, 120),
        left_cheek_rgb=(205, 140, 150),
        right_cheek_rgb=(200, 150, 120),
    )

    assert result.success is False
    assert result.error == "inconsistent_patches"
    assert result.max_hue_difference > _HUE_DIFFERENCE_THRESHOLD


def test_hue_difference_is_zero_for_identical_colors():
    assert hue_difference((60.0, 15.0, 20.0), (60.0, 15.0, 20.0)) == pytest.approx(0.0)


def test_hue_difference_is_symmetric():
    first, second = (60.0, 15.0, 20.0), (70.0, 22.0, 8.0)
    assert hue_difference(first, second) == pytest.approx(hue_difference(second, first))


def test_hue_difference_wraps_around_zero_degrees():
    def at_hue(hue_deg: float, chroma: float) -> tuple[float, float, float]:
        return (60.0, chroma * math.cos(math.radians(hue_deg)), chroma * math.sin(math.radians(hue_deg)))

    across_zero = hue_difference(at_hue(359.0, 20.0), at_hue(1.0, 20.0))
    same_gap_elsewhere = hue_difference(at_hue(44.0, 20.0), at_hue(46.0, 20.0))

    # 359 vs 1 degree is a 2 degree gap the short way round, not 358.
    assert across_zero == pytest.approx(same_gap_elsewhere)
    assert across_zero < 1.0


def test_hue_difference_scales_with_chroma():
    def at_hue(hue_deg: float, chroma: float) -> tuple[float, float, float]:
        return (60.0, chroma * math.cos(math.radians(hue_deg)), chroma * math.sin(math.radians(hue_deg)))

    # Same 30 degree hue gap: near-neutral colors barely differ, saturated
    # ones differ a lot - the whole reason this replaced a raw angle check.
    low_chroma = hue_difference(at_hue(30.0, 5.0), at_hue(60.0, 5.0))
    high_chroma = hue_difference(at_hue(30.0, 30.0), at_hue(60.0, 30.0))

    assert high_chroma == pytest.approx(low_chroma * 6)
    assert low_chroma < _HUE_DIFFERENCE_THRESHOLD < high_chroma


def test_normalize_depth_lightness_is_noop_at_reference_sclera():
    assert normalize_depth_lightness(50.0, 67.0) == pytest.approx(50.0)


def test_normalize_depth_lightness_boosts_skin_when_sclera_reads_dim():
    # Dim sclera signals an underexposed photo - skin should be corrected
    # upward to compensate.
    assert normalize_depth_lightness(50.0, 57.0) == pytest.approx(60.0)


def test_normalize_depth_lightness_lowers_skin_when_sclera_reads_bright():
    # Bright sclera signals an overexposed/well-lit photo - skin should be
    # corrected downward, this is the case that motivated the fix (a
    # brighter, more evenly-lit shot inflating raw skin L* independent of
    # the subject's real depth).
    assert normalize_depth_lightness(50.0, 77.0) == pytest.approx(40.0)


def test_normalize_depth_lightness_clamps_extreme_corrections():
    # Sclera far dimmer than reference: correction clamped at +20, not +48.
    assert normalize_depth_lightness(50.0, 20.0) == pytest.approx(70.0)
    # Sclera far brighter than reference: correction clamped at -20, not -40.
    assert normalize_depth_lightness(50.0, 108.0) == pytest.approx(30.0)


def test_normalize_depth_lightness_clamp_applies_to_correction_not_skin_sclera_gap():
    # A large skin-to-sclera gap alone must NOT trigger the clamp - only the
    # sclera's own deviation from the reference does. skin_l is 22.89 below
    # sclera_l here (comfortably past +-20), but the sclera itself reads
    # only 3.75 below the reference, so no clamping should occur: the result
    # should match the plain unclamped arithmetic, not a gap-clamped value.
    skin_l, sclera_l = 48.86, 71.75
    assert abs(skin_l - sclera_l) > 20.0  # the gap alone would suggest clamping applies
    result = normalize_depth_lightness(skin_l, sclera_l)
    assert result == pytest.approx(skin_l + (67.0 - sclera_l), abs=0.01)


def test_normalize_depth_lightness_leaves_deep_skin_unchanged_under_reference_lighting():
    # The failure mode this function exists to avoid re-introducing: if the
    # clamp bounded skin_l - sclera_l instead of (reference - sclera_l), a
    # genuinely very deep skin tone photographed under exactly
    # reference-quality lighting would get dragged up toward "light" purely
    # because it's naturally far darker than sclera - nothing to do with
    # lighting. sclera_l == _SCLERA_REFERENCE_L means zero lighting anomaly,
    # so skin_l must pass through completely unchanged, no matter how large
    # skin_l - sclera_l is.
    skin_l, sclera_l = 25.0, 67.0
    assert skin_l - sclera_l < -20.0  # gap alone would suggest clamping, if that were what's clamped
    assert normalize_depth_lightness(skin_l, sclera_l) == pytest.approx(skin_l)


def test_normalize_depth_lightness_clamps_output_to_valid_lab_range():
    assert normalize_depth_lightness(95.0, 20.0) == pytest.approx(100.0)
    assert normalize_depth_lightness(5.0, 108.0) == pytest.approx(0.0)


def test_normalize_undertone_ab_is_noop_at_reference_sclera():
    assert normalize_undertone_ab(15.0, 20.0, _SCLERA_REFERENCE_A, _SCLERA_REFERENCE_B) == pytest.approx((15.0, 20.0))


def test_normalize_undertone_ab_shifts_warmer_when_sclera_reads_cool():
    # A sclera reading below the reference in both a*/b* signals a cool-cast
    # light source - skin should be corrected warmer (up) to compensate.
    # Expressed relative to the reference constants (rather than hardcoded
    # absolute numbers) so this test doesn't silently go stale if the
    # reference is recalibrated again.
    sclera_a, sclera_b = _SCLERA_REFERENCE_A - 5.0, _SCLERA_REFERENCE_B - 3.0
    skin_a, skin_b = 10.0, 12.0
    assert normalize_undertone_ab(skin_a, skin_b, sclera_a, sclera_b) == pytest.approx((15.0, 15.0))


def test_normalize_undertone_ab_shifts_cooler_when_sclera_reads_warm():
    # A sclera reading above the reference in both a*/b* signals a warm-cast
    # light source (e.g. an incandescent bulb) - skin should be corrected
    # cooler (down) to compensate. This is the case that motivated the fix.
    sclera_a, sclera_b = _SCLERA_REFERENCE_A + 5.0, _SCLERA_REFERENCE_B + 3.0
    skin_a, skin_b = 10.0, 12.0
    assert normalize_undertone_ab(skin_a, skin_b, sclera_a, sclera_b) == pytest.approx((5.0, 9.0))


def test_normalize_undertone_ab_leaves_high_chroma_skin_unchanged_under_reference_lighting():
    # sclera == reference means zero inferred lighting cast, so even skin
    # far more saturated than the sclera must pass through unchanged - it's
    # naturally that warm, not lit that way.
    skin_a, skin_b = 45.0, 55.0
    assert normalize_undertone_ab(skin_a, skin_b, _SCLERA_REFERENCE_A, _SCLERA_REFERENCE_B) == pytest.approx(
        (skin_a, skin_b)
    )


def test_sclera_color_cast_measures_sclera_deviation_from_reference():
    assert sclera_color_cast((60.0, _SCLERA_REFERENCE_A, _SCLERA_REFERENCE_B)) == pytest.approx(0.0)
    assert sclera_color_cast((60.0, _SCLERA_REFERENCE_A + 3.0, _SCLERA_REFERENCE_B + 4.0)) == pytest.approx(5.0)


# Real sclera reading from a photo lit by a warm incandescent bulb (color-
# managed): a cast of ~21.9, too strong for the sclera correction to bring
# that photo's undertone into line with the same subject's other photos.
_WARM_BULB_SCLERA_RGB = (152, 104, 73)
# The same cast direction at ~15.0 - strong, but just under the cutoff.
_STRONG_BUT_CORRECTABLE_SCLERA_RGB = (145, 106, 83)


def test_classify_season_rejects_a_color_cast_too_strong_to_correct():
    skin = (241, 150, 92)  # that same photo's (warm-lit) cheek

    result = classify_season(skin, skin, skin, sclera_rgb=_WARM_BULB_SCLERA_RGB)

    assert result.success is False
    assert result.error == "color_cast"
    assert result.classification is None
    assert result.color_cast == pytest.approx(21.9, abs=0.1)
    assert result.color_cast > _MAX_COLOR_CAST_AB


def test_classify_season_corrects_a_cast_just_under_the_cutoff():
    skin = (241, 150, 92)

    result = classify_season(skin, skin, skin, sclera_rgb=_STRONG_BUT_CORRECTABLE_SCLERA_RGB)

    assert result.success is True
    assert result.color_cast == pytest.approx(15.0, abs=0.1)


def test_classify_season_judges_color_cast_by_sclera_not_skin_to_sclera_gap():
    # Naturally warm, high-chroma skin under reference-neutral light: the
    # skin sits far from the sclera in a*/b*, but the sclera itself shows no
    # cast, so this must classify - rejecting on the skin-to-sclera gap
    # would reject exactly the warmest, most saturated skin tones.
    skin = (230, 140, 80)
    neutral_sclera = (124, 108, 101)  # a*/b* at the reference
    skin_lab, sclera_lab = rgb_to_lab(skin), rgb_to_lab(neutral_sclera)
    assert math.hypot(skin_lab[1] - sclera_lab[1], skin_lab[2] - sclera_lab[2]) > _MAX_COLOR_CAST_AB

    result = classify_season(skin, skin, skin, sclera_rgb=neutral_sclera)

    assert result.success is True
    assert result.color_cast < 1.0


def test_classify_season_reports_color_cast_before_inconsistent_patches():
    # Dragging the patches can fix mixed patch readings; only a retake fixes
    # the lighting - so a photo with both problems is sent to retake.
    result = classify_season(
        forehead_rgb=(200, 150, 120),
        left_cheek_rgb=(205, 140, 150),
        right_cheek_rgb=(200, 150, 120),
        sclera_rgb=_WARM_BULB_SCLERA_RGB,
    )

    assert result.success is False
    assert result.error == "color_cast"
    assert result.max_hue_difference > _HUE_DIFFERENCE_THRESHOLD


def test_classify_season_without_sclera_reports_no_color_cast():
    result = classify_season((126, 92, 68), (126, 92, 68), (126, 92, 68))

    assert result.success is True
    assert result.color_cast is None


def test_classify_season_without_sclera_matches_raw_behavior():
    result = classify_season((126, 92, 68), (126, 92, 68), (126, 92, 68), sclera_rgb=None)

    assert result.success is True
    classification = result.classification
    assert classification.depth_lightness == pytest.approx(classification.avg_lab[0])
    raw_hue, raw_chroma = hue_and_chroma(classification.avg_lab)
    assert classification.hue_deg == pytest.approx(raw_hue)
    assert classification.chroma == pytest.approx(raw_chroma)
    assert classification.depth == "deep"
    assert classification.season == "Autumn"


def test_classify_season_sclera_correction_flips_depth_and_season():
    # Real bug case this fix addresses: a skin patch that reads confidently
    # "deep" under raw L* (51.3, well past the ambiguity band) should flip
    # to "light" once corrected against a dim sclera reading, which signals
    # the raw L* was itself depressed by this photo's own underexposure.
    skin = (150, 115, 92)

    raw_result = classify_season(skin, skin, skin)
    assert raw_result.classification.depth == "deep"
    assert raw_result.classification.season == "Autumn"

    # (124, 108, 101) is deliberately chosen to have a*/b* equal to
    # (_SCLERA_REFERENCE_A, _SCLERA_REFERENCE_B) - i.e. zero inferred color
    # cast - and differ from the reference only in L*. This isolates the
    # depth correction this test is about: now that undertone is also
    # corrected from the same sclera reading (normalize_undertone_ab), a
    # sclera_rgb picked without controlling for a*/b* would also perturb
    # undertone/clarity as a side effect unrelated to what this test checks.
    corrected_result = classify_season(skin, skin, skin, sclera_rgb=(124, 108, 101))
    assert corrected_result.classification.depth == "light"
    assert corrected_result.classification.season == "Spring"
    # Undertone/clarity are untouched by the correction - only depth changes.
    assert corrected_result.classification.undertone == raw_result.classification.undertone
    assert corrected_result.classification.clarity == raw_result.classification.clarity
    assert corrected_result.classification.avg_lab == raw_result.classification.avg_lab


def test_classify_season_sclera_correction_flips_undertone_and_season():
    # Symmetric to the depth-flip test above, on the other axis: a skin
    # patch that reads "cool" under raw hue (49.95deg, just barely under the
    # 50deg warm/cool threshold) should flip to "warm" once corrected
    # against a sclera reading whose b* sits well below the reference -
    # signaling this photo's own light source had a cool color-temperature
    # cast that pulled the whole shot (skin included) cooler than it really
    # is.
    skin = (185, 150, 135)

    raw_result = classify_season(skin, skin, skin)
    assert raw_result.classification.undertone == "cool"
    assert raw_result.classification.season == "Summer"

    # (171, 160, 166) is chosen to have L* at the depth reference (so depth
    # is left isolated/untouched, mirroring how the depth-flip test above
    # neutralizes a*/b*) and a* at the undertone reference, with only b*
    # shifted well below reference - the same axis a real incandescent-vs-
    # LED color-temperature cast lands on.
    corrected_result = classify_season(skin, skin, skin, sclera_rgb=(171, 160, 166))
    assert corrected_result.classification.undertone == "warm"
    assert corrected_result.classification.season == "Spring"
    # Depth is untouched by the undertone correction - only hue/chroma
    # change. depth_lightness has a tiny (~0.04) residual shift since this
    # sclera's L* isn't exactly at the reference (integer RGB rounding), not
    # because the undertone correction leaked into the depth axis.
    assert corrected_result.classification.depth == raw_result.classification.depth
    assert corrected_result.classification.depth_lightness == pytest.approx(
        raw_result.classification.depth_lightness, abs=0.1
    )
    assert corrected_result.classification.avg_lab == raw_result.classification.avg_lab


def test_classify_season_ambiguity_band_tie_break_applies_to_corrected_lightness():
    # A patch whose raw L* sits outside the ambiguity band (58 +/- 5,
    # confidently "light") but whose sclera-corrected depth_lightness lands
    # inside it should be tie-broken by clarity, exactly like the existing
    # raw-L* ambiguity tests - proving the band applies to whichever
    # lightness was actually used for the decision, not always the raw one.
    skin = (200, 155, 148)
    raw_result = classify_season(skin, skin, skin)
    assert raw_result.classification.depth_lightness == pytest.approx(67.92, abs=0.01)
    assert raw_result.classification.clarity == "muted"
    assert raw_result.classification.depth == "light"  # unambiguous: above the band entirely

    # A moderately bright sclera pulls the corrected value down into the band.
    # (200, 182, 175) is chosen the same way as in the depth-flip test above:
    # a*/b* equal to the reference, so only L* differs, isolating the depth
    # correction from the now-coupled undertone correction.
    corrected_result = classify_season(skin, skin, skin, sclera_rgb=(200, 182, 175))
    depth_lightness = corrected_result.classification.depth_lightness
    assert abs(depth_lightness - 58.0) <= 5.0  # now inside the ambiguity band
    assert corrected_result.classification.clarity == "muted"
    assert corrected_result.classification.depth == "deep"  # muted chroma breaks the tie toward deep


def test_classify_season_tolerates_strong_directional_lighting():
    # Real photo: a single light in front of the face, angled slightly
    # upward, on shiny/glare-prone skin - forehead reads much brighter than
    # either cheek (L* spread ~33.6, well past the old brightness-based
    # threshold this replaced), but all three patches agree on hue (max
    # pairwise ΔH* ~2.0), so this should still classify rather than being
    # rejected for a lighting pattern that's extremely common in selfies.
    result = classify_season(
        forehead_rgb=(192, 156, 138),
        left_cheek_rgb=(146, 97, 76),
        right_cheek_rgb=(116, 66, 45),
    )

    assert result.success is True
    assert result.classification is not None
