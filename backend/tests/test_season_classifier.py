import pytest

from app.vision.season_classifier import classify_season, rgb_to_lab


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
    # off the other two (near-zero chroma makes hue unstable - see
    # hue_and_chroma) - a 68 degree hue spread, not just a brighter patch.
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
    # spread, ~1.9 degree hue spread) should still classify - the
    # consistency check shouldn't be so strict it rejects normal photos.
    result = classify_season(
        forehead_rgb=(216, 165, 152),
        left_cheek_rgb=(208, 158, 145),
        right_cheek_rgb=(222, 170, 158),
    )

    assert result.success is True
    assert result.classification is not None


def test_classify_season_tolerates_strong_directional_lighting():
    # Real photo: a single light in front of the face, angled slightly
    # upward, on shiny/glare-prone skin - forehead reads much brighter than
    # either cheek (L* spread ~33.6, well past the old brightness-based
    # threshold this replaced), but all three patches agree on hue (~5.1
    # degree spread), so this should still classify rather than being
    # rejected for a lighting pattern that's extremely common in selfies.
    result = classify_season(
        forehead_rgb=(192, 156, 138),
        left_cheek_rgb=(146, 97, 76),
        right_cheek_rgb=(116, 66, 45),
    )

    assert result.success is True
    assert result.classification is not None
