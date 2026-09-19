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
