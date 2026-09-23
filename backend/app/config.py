import json
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables (or a local .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "palette-api"
    app_version: str = "0.1.0"
    environment: str = "development"

    # Origins allowed to call this API from a browser. Stored as the raw env
    # string (not list[str]) because pydantic-settings tries to json.loads()
    # any list-typed env value before validators even run, which used to
    # crash startup with an opaque SettingsError on a bare comma-separated
    # value. Parsed on demand via cors_origins_list().
    cors_origins: str = "http://localhost:3000"

    # Per-IP rate limits for POST /api/scan and /api/scan/manual (see
    # app/rate_limit.py) -- each request runs a MediaPipe face-landmark
    # pipeline plus a billed Anthropic call. Single-process, in-memory
    # limiter -- fine pre-launch with no real traffic; would need a shared
    # store (e.g. Redis) if this is ever scaled to multiple Render
    # instances/processes, since in-memory state isn't shared across
    # processes.
    scan_rate_limit_per_minute: int = 5
    scan_rate_limit_per_day: int = 60

    # See app/client_ip.py. Default (False) trusts the RIGHTMOST entry of
    # X-Forwarded-For -- matches standard reverse-proxy append behavior.
    # Flip via env var only if an empirical check against the live
    # deployment shows Render puts the real client IP somewhere else -- no
    # code change needed.
    client_ip_trust_leftmost: bool = False

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

    # Best-effort: app/confirmation_email.py no-ops (returns False) when
    # this is unset. A purchase confirmation email must never block or
    # fail the Stripe webhook that already confirmed payment.
    resend_api_key: str | None = None

    def cors_origins_list(self) -> list[str]:
        """Parse cors_origins as either a comma-separated list or a JSON array."""
        value = self.cors_origins.strip()
        if not value:
            raise ValueError(
                "CORS_ORIGINS must not be empty -- set a comma-separated list "
                '(e.g. "https://example.com,https://foo.com") or a JSON array '
                '(e.g. ["https://example.com"]).'
            )
        if value.startswith("["):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"CORS_ORIGINS looks like a JSON array but failed to parse ({exc}). "
                    "Use a JSON array or a comma-separated list of origins."
                ) from exc
            if not isinstance(parsed, list) or not all(isinstance(o, str) for o in parsed):
                raise ValueError("CORS_ORIGINS JSON array must contain only strings.")
            return parsed
        return [origin.strip() for origin in value.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
