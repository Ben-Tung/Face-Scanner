from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables (or a local .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "palette-api"
    app_version: str = "0.1.0"
    environment: str = "development"

    # Origins allowed to call this API from a browser.
    cors_origins: list[str] = ["http://localhost:3000"]

    # Read here so the value is validated at startup, but nothing connects to it
    # yet — the database is wired up when the first feature needs it.
    database_url: str | None = None

    # Best-effort: app/paragraph.py no-ops (returns None) when this is unset.
    anthropic_api_key: str | None = None

    # Stripe Checkout for the full-report unlock. Left unset means payments
    # are disabled: app/routers/payments.py returns a clear 500 rather than
    # letting the Stripe SDK fail with an opaque auth error.
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None

    # Where the Next.js frontend is served — used to build Stripe Checkout
    # success_url/cancel_url. Has a real default (unlike the secrets above)
    # since Checkout needs *some* value even in a bare local run.
    frontend_base_url: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
