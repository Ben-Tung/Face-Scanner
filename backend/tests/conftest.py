"""Session-wide test isolation for external-dependency settings.

Settings() reads backend/.env when tests run with cwd=backend/ (both
`cd backend && pytest` and `docker compose exec backend pytest`, the latter
via docker-compose's bind mount of ./backend:/app). That file holds a real
ANTHROPIC_API_KEY for local dev. Without this guard, every /api/scan
success-path test would make a real, billed Anthropic call instead of
exercising the deterministic season/swatch path it's meant to test.
test_paragraph.py overrides this per-test via its own monkeypatch, which
layers on top of this fixture correctly.
"""

import pytest

from app import paragraph
from app.config import Settings


@pytest.fixture(autouse=True)
def _disable_ai_paragraph_by_default(monkeypatch):
    monkeypatch.setattr(paragraph, "get_settings", lambda: Settings(anthropic_api_key=None))
