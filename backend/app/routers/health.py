from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config import Settings, get_settings

router = APIRouter(prefix="/api", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness check.

    Deliberately does not touch Postgres — nothing depends on it yet, and a
    health check that fails on an unused dependency is just noise.
    """
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
