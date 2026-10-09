"""
tests/test_security_and_auth.py — Middleware, Rate Limiting, Security Headers & Auth Tests
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.middleware import RateLimitMiddleware
from app.main import app


def _clear_rate_limiter():
    curr = getattr(app, "middleware_stack", None)
    while curr:
        if isinstance(curr, RateLimitMiddleware):
            curr._hits.clear()
        curr = getattr(curr, "app", None)


def test_security_headers_present(client: TestClient):
    """Verify security headers are attached to responses."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Referrer-Policy") == "same-origin"
    assert "microphone=(self)" in response.headers.get("Permissions-Policy", "")


def test_cors_headers(client: TestClient):
    """Verify CORS middleware responds with correct access-control headers for allowed origins."""
    response = client.get(
        "/health",
        headers={"Origin": "http://127.0.0.1:8000"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"


def test_rate_limit_middleware_exceeded(client: TestClient, monkeypatch):
    """Verify rate limit middleware returns HTTP 429 when limit is exceeded."""
    _clear_rate_limiter()
    # Temporarily lower rate limit for test
    monkeypatch.setattr(settings, "rate_limit_requests", 3)
    monkeypatch.setattr(settings, "rate_limit_window_seconds", 60)

    # 3 allowed requests
    for _ in range(3):
        res = client.get("/health")
        assert res.status_code == 200

    # 4th request should be blocked with 429
    res_blocked = client.get("/health")
    assert res_blocked.status_code == 429
    assert "Too many requests" in res_blocked.json()["detail"]


def test_rate_limit_bypasses_docs(client: TestClient, monkeypatch):
    """Verify OpenAPI and documentation routes bypass rate limiting."""
    _clear_rate_limiter()
    monkeypatch.setattr(settings, "rate_limit_requests", 1)
    monkeypatch.setattr(settings, "rate_limit_window_seconds", 60)

    # Consume single rate limit credit
    client.get("/health")

    # Documentation route should still succeed
    docs_res = client.get("/openapi.json")
    assert docs_res.status_code == 200


def test_production_startup_fails_without_admin_api_key(monkeypatch):
    """In production environment, fail fast if ADMIN_API_KEY is empty."""
    import asyncio
    from app.main import lifespan

    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "admin_api_key", "")
    monkeypatch.setattr(settings, "voice_websocket_token", "valid_ws_token")

    async def _test():
        with pytest.raises(RuntimeError, match="ADMIN_API_KEY must be set"):
            async with lifespan(app):
                pass

    asyncio.run(_test())


def test_production_startup_fails_without_websocket_token(monkeypatch):
    """In production environment, fail fast if VOICE_WEBSOCKET_TOKEN is empty."""
    import asyncio
    from app.main import lifespan

    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "admin_api_key", "valid_admin_key")
    monkeypatch.setattr(settings, "voice_websocket_token", "")

    async def _test():
        with pytest.raises(RuntimeError, match="VOICE_WEBSOCKET_TOKEN must be set"):
            async with lifespan(app):
                pass

    asyncio.run(_test())


def test_development_startup_succeeds_without_keys(monkeypatch):
    """In development/test, server starts cleanly even if keys are unset."""
    import asyncio
    from app.main import lifespan

    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "admin_api_key", "")
    monkeypatch.setattr(settings, "voice_websocket_token", "")

    async def _test():
        async with lifespan(app):
            pass

    asyncio.run(_test())


def test_call_language_cache_cleared_on_disconnect():
    """Verify _call_language_cache is cleared in finally block on disconnect."""
    import asyncio
    from unittest.mock import AsyncMock
    from app.services.voice_service import process_voice_websocket, _call_language_cache

    mock_ws = AsyncMock()
    # Mock receive to disconnect immediately
    from fastapi.websockets import WebSocketDisconnect
    mock_ws.receive.side_effect = WebSocketDisconnect()

    async def _test():
        initial_count = len(_call_language_cache)
        try:
            await process_voice_websocket(mock_ws, phone_number=None)
        except WebSocketDisconnect:
            pass
        # Ensure no residual entries leaked
        assert len(_call_language_cache) == initial_count

    asyncio.run(_test())

