import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.client_ip import get_client_ip
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
    # empirically confirmed (see the rate-limiting rollout plan). Logs the
    # raw header, the ASGI-level peer address, AND what app.client_ip.
    # get_client_ip() actually resolves from them -- so confirming the
    # extraction logic is correct doesn't require re-deriving it by hand
    # from a raw header dump each time.
    @app.middleware("http")
    async def _log_client_ip_diagnostic(request, call_next):
        # .warning, not .info: nothing in this app calls logging.basicConfig,
        # so the root logger's effective level defaults to WARNING and an
        # INFO-level call here would be silently dropped before ever
        # reaching a handler (see the other app.* loggers in this codebase,
        # which all use .warning/.exception for the same reason).
        logger.warning(
            "xff_diagnostic path=%s raw_xff=%r client_host=%r resolved_client_ip=%r",
            request.url.path,
            request.headers.get("x-forwarded-for"),
            request.client.host if request.client else None,
            get_client_ip(request),
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
