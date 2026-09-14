import re

from app.beauty_guidance import GOLD_SWATCH, GUIDANCE_BY_SEASON, SILVER_SWATCH

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SEASONS = {"Spring", "Summer", "Autumn", "Winter"}
_METALS = {"Gold", "Silver", "Both"}


def test_all_seasons_are_present():
    assert set(GUIDANCE_BY_SEASON.keys()) == _SEASONS


def test_gold_and_silver_reference_swatches_have_valid_hex():
    assert _HEX_RE.match(GOLD_SWATCH.hex)
    assert _HEX_RE.match(SILVER_SWATCH.hex)


def test_each_season_has_complete_makeup_guidance():
    for season, guidance in GUIDANCE_BY_SEASON.items():
        makeup = guidance.makeup
        assert makeup.foundation_undertone, season
        assert makeup.foundation_tip, season
        assert 3 <= len(makeup.lip_shades) <= 4, season
        assert 2 <= len(makeup.blush_shades) <= 3, season
        for swatch in (*makeup.lip_shades, *makeup.blush_shades):
            assert swatch.name
            assert _HEX_RE.match(swatch.hex), (season, swatch)


def test_each_season_has_valid_jewelry_guidance():
    for season, guidance in GUIDANCE_BY_SEASON.items():
        assert guidance.jewelry.metal in _METALS, season
        assert guidance.jewelry.tip, season


def test_each_season_has_shopping_guidance():
    for season, guidance in GUIDANCE_BY_SEASON.items():
        assert guidance.shopping_guidance, season
