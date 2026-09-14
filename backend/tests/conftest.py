"""Session-wide test isolation for external-dependency settings.

Settings() reads backend/.env when tests run with cwd=backend/ (both
`cd backend && pytest` and `docker compose exec backend pytest`, the latter
via docker-compose's bind mount of ./backend:/app). That file holds a real
ANTHROPIC_API_KEY and a real (currently pgbouncer-broken, see project
memory) DATABASE_URL for local dev. Without these guards, every /api/scan
success-path test would either make a real, billed Anthropic call, or try
to persist a scan against whatever database backend/.env points at,
instead of exercising the deterministic season/swatch path they're meant
to test. test_paragraph.py and test_scans_repo.py/test_payments.py
override these per-test via their own monkeypatches, which layer on top of
these fixtures correctly.
"""

import pytest

from app import paragraph
from app.config import Settings
from app.routers import scan as scan_module


@pytest.fixture(autouse=True)
def _disable_ai_paragraph_by_default(monkeypatch):
    monkeypatch.setattr(paragraph, "get_settings", lambda: Settings(anthropic_api_key=None))


@pytest.fixture(autouse=True)
def _stub_scan_persistence_by_default(monkeypatch):
    monkeypatch.setattr(scan_module, "create_scan", lambda *args, **kwargs: None)
