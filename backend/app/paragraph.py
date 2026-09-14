"""Best-effort AI paragraph generation.

See CLAUDE.md's critical rule: this is the ONLY place in the app an LLM
call is allowed. Face detection and season/undertone/depth/clarity
classification stay fully deterministic Python (season_classifier.py) —
this module only turns an already-computed result into a short paragraph.

Mirrors app/analytics.py's best-effort shape: a scan's deterministic season
and swatches must never fail because Anthropic is unreachable, unconfigured,
or erroring. Every failure here is caught and logged; callers get None.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache

import anthropic

from app.beauty_guidance import BeautyGuidance
from app.config import get_settings
from app.palettes import Swatch
from app.vision.season_classifier import Season, SeasonClassification

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 300
# This runs inline in the request path (both /api/scan and /api/scan/manual
# await it before responding) — bounded well under the SDK's 10-minute
# default so a hung network call can't hang the whole scan request.
_REQUEST_TIMEOUT_SECONDS = 8.0
_MAX_RETRIES = 1  # default is 2; bounds worst-case wall-clock to ~16s, not ~24s

_SYSTEM_PROMPT = (
    "You are a warm, knowledgeable color consultant. A user has just gotten "
    "their color season result from a quick photo-based analysis. Write a "
    "short paragraph describing what their season means for them, in a "
    "warm, encouraging, personal tone — as if speaking directly to them. "
    "Reference at least one or two of their actual palette colors by name "
    "so it feels specific to them, not generic. Keep it to 2-3 short "
    "sentences, no more than 50 words total — prefer short, direct "
    "sentences over long compound ones. Do not use headers, bullet points, "
    "emoji, or quotation marks. Do not include any preamble or sign-off — "
    "output only the paragraph itself."
)

_USER_MESSAGE_TEMPLATE = (
    "My color season is {season} ({undertone} undertone, {depth} depth, "
    "{clarity} clarity).\nMy example palette includes: {swatch_names}."
)

_FULL_REPORT_SYSTEM_PROMPT = (
    "You are a warm, knowledgeable color consultant. A user has just paid to "
    "unlock their full color season report. Write a short paragraph "
    "celebrating their season and tying together their best colors with "
    "their beauty guidance, in a warm, encouraging, personal tone — as if "
    "speaking directly to them. Reference one or two of their actual best "
    "palette colors by name, and mention either their foundation undertone "
    "or their recommended jewelry metal, so it feels specific to them, not "
    "generic. Keep it to 2-4 short sentences, no more than 70 words total — "
    "prefer short, direct sentences over long compound ones. Do not invent "
    "colors or advice beyond what's given to you. Do not use headers, bullet "
    "points, emoji, or quotation marks. Do not include any preamble or "
    "sign-off — output only the paragraph itself."
)

_FULL_REPORT_USER_MESSAGE_TEMPLATE = (
    "My color season is {season}.\n"
    "My best palette colors include: {swatch_names}.\n"
    "My foundation undertone is {foundation_undertone}.\n"
    "My recommended jewelry metal is {jewelry_metal}."
)


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=get_settings().anthropic_api_key,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )


def _build_user_message(classification: SeasonClassification, swatches: Sequence[Swatch]) -> str:
    return _USER_MESSAGE_TEMPLATE.format(
        season=classification.season,
        undertone=classification.undertone,
        depth=classification.depth,
        clarity=classification.clarity,
        swatch_names=", ".join(swatch.name for swatch in swatches),
    )


def _build_full_report_user_message(
    season: Season, best_swatches: Sequence[Swatch], guidance: BeautyGuidance
) -> str:
    return _FULL_REPORT_USER_MESSAGE_TEMPLATE.format(
        season=season,
        swatch_names=", ".join(swatch.name for swatch in best_swatches),
        foundation_undertone=guidance.makeup.foundation_undertone,
        jewelry_metal=guidance.jewelry.metal,
    )


def _extract_text(response) -> str | None:
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or None


def _complete(system: str, user_message: str) -> str | None:
    """Run one best-effort Claude completion.

    Returns None (never raises) if ANTHROPIC_API_KEY isn't configured or the
    API call fails for any reason. Shared by every paragraph-generating
    function in this module — callers must not treat their caller's larger
    request as failed just because this returns None.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None

    try:
        response = _client().messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.RateLimitError:
        logger.warning("Anthropic rate-limited the paragraph request", exc_info=True)
        return None
    except anthropic.AuthenticationError:
        logger.warning("Anthropic rejected the configured API key", exc_info=True)
        return None
    except anthropic.APIStatusError:
        logger.warning("Anthropic returned an error status for the paragraph request", exc_info=True)
        return None
    except anthropic.APIConnectionError:
        logger.warning("Network error calling Anthropic for the paragraph request", exc_info=True)
        return None
    except Exception:
        logger.warning("Unexpected failure generating the AI paragraph", exc_info=True)
        return None

    return _extract_text(response)


def generate_paragraph(classification: SeasonClassification, swatches: Sequence[Swatch]) -> str | None:
    """Turn an already-computed season classification into a short warm paragraph.

    Best-effort: returns None (never raises) if ANTHROPIC_API_KEY isn't
    configured or the API call fails for any reason. Callers must not treat
    a scan as failed just because this returns None.
    """
    return _complete(_SYSTEM_PROMPT, _build_user_message(classification, swatches))


def generate_full_report_paragraph(
    season: Season, best_swatches: Sequence[Swatch], guidance: BeautyGuidance
) -> str | None:
    """Turn the paid tier's curated palette + beauty guidance into a short
    warm paragraph.

    Takes a bare `season` rather than a `SeasonClassification` because only
    `season` is persisted for a scan (see app/scans_repo.py) — undertone,
    depth, and clarity are computed at scan time but discarded, so they're
    no longer available by the time a paid report is viewed.

    Best-effort, same contract as generate_paragraph: never raises.
    """
    return _complete(
        _FULL_REPORT_SYSTEM_PROMPT, _build_full_report_user_message(season, best_swatches, guidance)
    )
