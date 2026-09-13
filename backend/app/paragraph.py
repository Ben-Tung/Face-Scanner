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

from app.config import get_settings
from app.palettes import Swatch
from app.vision.season_classifier import SeasonClassification

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


def _extract_text(response) -> str | None:
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or None


def generate_paragraph(classification: SeasonClassification, swatches: Sequence[Swatch]) -> str | None:
    """Turn an already-computed season classification into a short warm paragraph.

    Best-effort: returns None (never raises) if ANTHROPIC_API_KEY isn't
    configured or the API call fails for any reason. Callers must not treat
    a scan as failed just because this returns None.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None

    try:
        response = _client().messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(classification, swatches)}],
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
