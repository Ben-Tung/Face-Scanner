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


@lru_cache
def get_settings() -> Settings:
    return Settings()
