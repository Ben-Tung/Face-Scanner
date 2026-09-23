"""Spoofing-resistant client IP extraction behind Render's edge.

Confirmed empirically on 2026-09-23 by sending `X-Forwarded-For: 1.2.3.4`
from outside Render and reading back what the app actually received:

    raw_xff = "1.2.3.4, 174.74.246.29, 104.23.160.18, 10.197.84.32"
    client_host = "10.197.84.32"

client_host (the ASGI-level socket peer) matched the LAST entry exactly, and
that last entry is a private RFC1918 address -- Render's own internal load
balancer, not a real client. The second-to-last entry (104.23.160.18) falls
in Cloudflare's published 104.16.0.0/13 range. So the real path is:

    client --(1)--> Cloudflare edge --(2)--> Cloudflare/Render tunnel hop --(3)--> Render's internal LB --> this app

Three hops append an entry each (the IP of whoever is talking to them
directly), none of them strip what's already there -- so trusting the
rightmost entry (or request.client.host directly, which is the same value)
resolves to Render's own infrastructure for every request, not the client.
The real client IP is always exactly 3 positions from the right
(`ip_list[-3]`), regardless of how many fake entries a client prepends
first, since negative indexing counts from the end and each of the 3
trusted hops appends exactly one entry, no more, no less.

We deliberately do NOT use IpWare's `proxy_count`/`get_client_ip()`
orchestration: its validity check (`len(ip_list) - 1 >= proxy_count`)
assumes the client always contributes exactly one entry of its own before
the trusted chain, which isn't how X-Forwarded-For works -- a browser
client contributes zero entries, so that check would reject perfectly
ordinary requests. We use IpWare only for its well-tested comma/port/IP
parsing (get_ips_from_string) and do the "N hops from the right" indexing
ourselves.

client_ip_trust_hops (Settings) is the env-var-flippable escape hatch if
Render's proxy chain ever changes shape.
"""

from python_ipware import IpWare
from starlette.requests import Request

from app.config import get_settings

_ipware = IpWare()


def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if not xff:
        # Local dev / direct connections: nothing in front to trust but the
        # raw socket peer.
        return request.client.host if request.client else "unknown"

    ip_list = _ipware.get_ips_from_string(xff)
    trust_hops = get_settings().client_ip_trust_hops
    if not ip_list or len(ip_list) < trust_hops:
        return request.client.host if request.client else "unknown"

    return str(ip_list[-trust_hops])
