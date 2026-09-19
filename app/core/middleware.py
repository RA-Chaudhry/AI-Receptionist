from __future__ import annotations

from collections import defaultdict, deque
from time import monotonic

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.core.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=()"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path.startswith(("/docs", "/redoc", "/openapi.json", "/static")):
            return await call_next(request)

        client_host = request.client.host if request.client else "anonymous"
        now = monotonic()
        window = settings.rate_limit_window_seconds
        limit = settings.rate_limit_requests
        hits = self._hits[client_host]

        while hits and now - hits[0] > window:
            hits.popleft()

        if len(hits) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
            )

        hits.append(now)
        return await call_next(request)
