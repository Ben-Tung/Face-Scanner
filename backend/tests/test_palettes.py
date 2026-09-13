import re

from app.palettes import SWATCHES_BY_SEASON

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SEASONS = {"Spring", "Summer", "Autumn", "Winter"}


def test_all_seasons_are_present():
    assert set(SWATCHES_BY_SEASON.keys()) == _SEASONS


def test_each_season_has_four_or_five_swatches_with_valid_hex():
    for season, swatches in SWATCHES_BY_SEASON.items():
        assert 4 <= len(swatches) <= 5, season
        for swatch in swatches:
            assert swatch.name
            assert _HEX_RE.match(swatch.hex), (season, swatch)
