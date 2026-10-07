"""Session-wide test isolation for external-dependency settings.

Settings() reads backend/.env when tests run with cwd=backend/ (both
`cd backend && pytest` and `docker compose exec backend pytest`, the latter
via docker-compose's bind mount of ./backend:/app). That file holds a real
ANTHROPIC_API_KEY and a real DATABASE_URL (the dev Supabase project) for
local dev. Without these guards, every /api/scan success-path test would
make a real, billed Anthropic call, try to persist a scan against whatever
database backend/.env points at, and write analytics events into that
database's events table - instead of exercising the deterministic
season/swatch path they're meant to test. test_paragraph.py,
test_analytics.py and test_scans_repo.py/test_payments.py override these
per-test via their own monkeypatches, which layer on top of these fixtures
correctly - as do tests that spy on scan_module.log_event directly.

Every test file also shares one TestClient(app) hitting the real
app.rate_limit.limiter singleton, and TestClient requests all resolve to
the same request.client.host -- so without a reset, any test file that
calls /api/scan or /api/scan/manual more than a handful of times (e.g.
parametrized real-photo tests) would start tripping the rate limit and
fail with unrelated 429s.
"""

import pytest

from app import analytics, paragraph
from app.config import Settings
from app.rate_limit import limiter
from app.routers import scan as scan_module


@pytest.fixture(autouse=True)
def _disable_ai_paragraph_by_default(monkeypatch):
    monkeypatch.setattr(paragraph, "get_settings", lambda: Settings(anthropic_api_key=None))


@pytest.fixture(autouse=True)
def _disable_analytics_by_default(monkeypatch):
    # Patched at analytics' own settings lookup rather than at each module's
    # imported log_event, so every caller (scan, payments, anything added
    # later) gets the no-op without being listed here.
    monkeypatch.setattr(analytics, "get_settings", lambda: Settings(database_url=None))


@pytest.fixture(autouse=True)
def _stub_scan_persistence_by_default(monkeypatch):
    monkeypatch.setattr(scan_module, "create_scan", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    limiter.reset()
    yield
    limiter.reset()
