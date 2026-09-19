"""Small, explicit security gates for operator and voice-provider endpoints."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Query, status

from app.core.config import settings


def _is_production() -> bool:
    return settings.environment.strip().lower() == "production"


async def require_admin_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Protect CRM data in production without forcing a development secret."""
    expected = settings.admin_api_key
    if not expected:
        if _is_production():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Administrative API authentication is not configured.",
            )
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")


def validate_voice_websocket_token(token: str | None) -> bool:
    """Allow local development, but fail closed for provider streams in production."""
    expected = settings.voice_websocket_token
    if not expected:
        return not _is_production()
    return token is not None and secrets.compare_digest(token, expected)


async def require_voice_websocket_token(token: str | None = Query(default=None)) -> None:
    if not validate_voice_websocket_token(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid voice stream token.")
