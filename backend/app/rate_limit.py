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

from slowapi import Limiter

from app.client_ip import get_client_ip
from app.config import get_settings


def scan_limit_value() -> str:
    """Re-read from Settings on every request (slowapi re-evaluates a
    callable limit value per-request, not once at import time), so the
    threshold is tunable via env var without a code change."""
    settings = get_settings()
    return f"{settings.scan_rate_limit_per_minute}/minute;{settings.scan_rate_limit_per_day}/day"


limiter = Limiter(key_func=get_client_ip, headers_enabled=True, storage_uri="memory://")
