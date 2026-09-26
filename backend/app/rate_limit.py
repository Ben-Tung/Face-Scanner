"""Per-IP rate limiting for the unauthenticated, cost-bearing scan endpoints.

POST /api/scan and POST /api/scan/manual each run a MediaPipe face-landmark
pipeline plus a billed Anthropic call before returning a response, with no
auth in front of them -- see CLAUDE.md's description of the free scan as
the viral, TikTok-driven entry point. slowapi's own key functions
(get_remote_address / get_ipaddr) trust X-Forwarded-For naively, which is
exactly the spoofable approach app/client_ip.py exists to avoid -- so we
supply get_client_ip as a custom key_func instead.

In-memory, single-process storage: fine pre-launch with no real traffic.
Would need a shared store (e.g. Redis) if this is ever scaled to multiple
Render instances/processes -- see app/config.py's scan_rate_limit_* comment.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from app.client_ip import get_client_ip
from app.config import get_settings


def scan_limit_value() -> str:
    """Re-read from Settings on every request (slowapi re-evaluates a
    callable limit value per-request, not once at import time), so the
    threshold is tunable via env var without a code change."""
    settings = get_settings()
    return f"{settings.scan_rate_limit_per_minute}/minute;{settings.scan_rate_limit_per_day}/day"


limiter = Limiter(key_func=get_client_ip, headers_enabled=True, storage_uri="memory://")


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Replaces slowapi's own default handler, which responds with
    {"error": "Rate limit exceeded: ..."} -- a shape the frontend's error
    parsing (lib/api.ts's parseScanErrorBody) doesn't recognize, since every
    other error path here replies with FastAPI's usual {"detail": ...}. That
    mismatch meant a real rate-limited user saw the same generic "something
    went wrong" message as an actual bad photo, with no indication they'd
    just been throttled. exc.detail already names which window was hit
    (e.g. "5 per 1 minute"), so it's threaded straight into the message.
    """
    response = JSONResponse(
        {"detail": f"You've hit the free-scan limit ({exc.detail}). Please try again shortly."},
        status_code=429,
    )
    return limiter._inject_headers(response, request.state.view_rate_limit)
