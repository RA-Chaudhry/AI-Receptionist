from contextlib import asynccontextmanager
from pathlib import Path
import threading

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

from app.api.v1.api import api_router
from app.api.v1 import voice_ws
from app.core.config import settings
from app.core.logging import setup_logging
import asyncio
from app.core.middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from app.services.groq_service import GroqServiceError, close_shared_client
from app.services.voice_service import load_models, prewarm_voice_cache

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.environment == "production":
        if not settings.admin_api_key or not settings.admin_api_key.strip():
            raise RuntimeError("Production startup failure: ADMIN_API_KEY must be set and non-empty.")
        if not settings.voice_websocket_token or not settings.voice_websocket_token.strip():
            raise RuntimeError("Production startup failure: VOICE_WEBSOCKET_TOKEN must be set and non-empty.")

    # Schema changes are applied exclusively by Alembic.  Calling create_all at
    # runtime masks missing migrations and can create a schema different from
    # the one production deployments actually run.
    thread = threading.Thread(target=load_models, daemon=True)
    thread.start()
    asyncio.create_task(prewarm_voice_cache())
    yield
    await close_shared_client()


setup_logging()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI Receptionist, review reply generator, and basic CRM backend.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)

app.include_router(api_router, prefix="/api/v1")
if settings.VOICE_ENABLED:
    app.include_router(voice_ws.router, prefix="/api/v1")


@app.exception_handler(GroqServiceError)
async def groq_service_error_handler(request: Request, exc: GroqServiceError):
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


@app.get("/", tags=["Frontend"])
async def root():
    frontend_file = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    response = FileResponse(frontend_file)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
