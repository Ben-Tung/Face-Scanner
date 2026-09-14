import re

from app.palettes import FULL_PALETTE_BY_SEASON, SWATCHES_BY_SEASON

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


def test_full_palette_has_all_seasons():
    assert set(FULL_PALETTE_BY_SEASON.keys()) == _SEASONS


def test_full_palette_categories_have_valid_hex_and_names():
    for season, palette in FULL_PALETTE_BY_SEASON.items():
        for category in (palette.best, palette.good, palette.avoid):
            for swatch in category:
                assert swatch.name
                assert _HEX_RE.match(swatch.hex), (season, swatch)


def test_full_palette_category_sizes_and_totals_are_in_range():
    for season, palette in FULL_PALETTE_BY_SEASON.items():
        assert 10 <= len(palette.best) <= 12, season
        assert 8 <= len(palette.good) <= 10, season
        assert 6 <= len(palette.avoid) <= 8, season
        total = len(palette.best) + len(palette.good) + len(palette.avoid)
        assert 20 <= total <= 30, (season, total)


def test_full_palette_avoid_is_disjoint_from_best_and_good():
    for season, palette in FULL_PALETTE_BY_SEASON.items():
        recommended_hexes = {s.hex for s in palette.best} | {s.hex for s in palette.good}
        avoid_hexes = {s.hex for s in palette.avoid}
        assert not (recommended_hexes & avoid_hexes), season


def test_full_palette_best_is_superset_of_free_tier_swatches():
    for season, free_swatches in SWATCHES_BY_SEASON.items():
        best_hexes = {s.hex for s in FULL_PALETTE_BY_SEASON[season].best}
        for swatch in free_swatches:
            assert swatch.hex in best_hexes, (season, swatch)
