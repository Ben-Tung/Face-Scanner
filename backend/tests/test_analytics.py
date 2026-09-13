from app import analytics
from app.config import Settings


def test_log_event_noops_without_database_url(monkeypatch):
    monkeypatch.setattr(analytics, "get_settings", lambda: Settings(database_url=None))

    analytics.log_event("scan_started")  # must not raise


def test_log_event_swallows_unreachable_database(monkeypatch):
    monkeypatch.setattr(
        analytics,
        "get_settings",
        lambda: Settings(database_url="postgresql://bad:bad@localhost:1/nope"),
    )

    analytics.log_event("scan_started", {"season": "Winter"})  # must not raise
