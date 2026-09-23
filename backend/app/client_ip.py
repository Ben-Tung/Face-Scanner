"""Spoofing-resistant client IP extraction behind Render's edge proxy.

Render's edge sits in front of this app (it's never directly reachable) and
APPENDS the real peer IP as the last entry of X-Forwarded-For -- but it does
not strip or validate anything a client already put in that header. So
trusting the first (leftmost) entry is spoofable: a client can prepend
arbitrary fake IPs ahead of the one Render appends. We trust the rightmost
entry instead, regardless of how many attacker-controlled entries precede
it -- confirmed via python-ipware's leftmost=False behavior, which (after
reversing the parsed list) always returns the true last entry with no
dependency on the list's total length. Do NOT use IpWare's `proxy_count`
option here: it assumes a fixed total list length (client + exactly N
trusted proxies), which breaks the moment a client injects extra entries.

client_ip_trust_leftmost exists only as an env-var-flippable escape hatch,
in case an empirical check against the live Render deployment (see the
project's rollout plan) ever shows the opposite of this default.
"""

from functools import lru_cache

from python_ipware import IpWare
from starlette.requests import Request

from app.config import get_settings


@lru_cache
def _ipware() -> IpWare:
    settings = get_settings()
    return IpWare(leftmost=settings.client_ip_trust_leftmost)


def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if not xff:
        # Local dev / direct connections: nothing in front to trust but the
        # raw socket peer.
        return request.client.host if request.client else "unknown"

    ip, _trusted_route = _ipware().get_client_ip(meta={"HTTP_X_FORWARDED_FOR": xff})
    if ip is None:
        return request.client.host if request.client else "unknown"
    return str(ip)
