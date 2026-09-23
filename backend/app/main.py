import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.rate_limit import limiter
from app.routers import health, payments, scan

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # TEMPORARY -- remove once Render's X-Forwarded-For behavior is
    # empirically confirmed (see the rate-limiting rollout plan). Logs both
    # the raw header and the ASGI-level peer address so we can tell whether
    # Render appends the real client IP to X-Forwarded-For (expected) or
    # already rewrites request.client.host directly.
    @app.middleware("http")
    async def _log_client_ip_diagnostic(request, call_next):
        logger.info(
            "xff_diagnostic path=%s raw_xff=%r client_host=%r",
            request.url.path,
            request.headers.get("x-forwarded-for"),
            request.client.host if request.client else None,
        )
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(scan.router)
    app.include_router(payments.router)

    return app


app = create_app()
