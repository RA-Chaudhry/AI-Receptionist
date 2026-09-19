"""
tests/test_provider_manager.py — Unit Tests for Groq Provider Manager & Multi-Layer Dispatch
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.core.config import settings
from app.services.groq_service import GroqService, GroqServiceError
from app.services.ai_service import get_ai_response
from app.services.receptionist_router import ReceptionistRouter


def test_groq_service_missing_api_key(monkeypatch):
    """Verify GroqService raises GroqServiceError when GROQ_API_KEY is empty."""
    monkeypatch.setattr(settings, "groq_api_key", "")
    with pytest.raises(GroqServiceError, match="GROQ_API_KEY is missing"):
        GroqService()


def test_groq_service_http_status_error(monkeypatch):
    """Verify HTTP status errors are converted to GroqServiceError."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk_testkey")
    service = GroqService()

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("500 Internal Error", request=MagicMock(), response=mock_resp)

    async def _run():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(GroqServiceError, match="Groq API request failed"):
                await service.chat_completion([{"role": "user", "content": "Hello"}])

    asyncio.run(_run())


def test_groq_service_network_request_error(monkeypatch):
    """Verify network connection errors are converted to GroqServiceError."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk_testkey")
    service = GroqService()

    async def _run():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(GroqServiceError, match="Unable to connect to Groq API"):
                await service.chat_completion([{"role": "user", "content": "Hello"}])

    asyncio.run(_run())


def test_groq_service_invalid_json_shape(monkeypatch):
    """Verify malformed JSON responses raise GroqServiceError."""
    monkeypatch.setattr(settings, "groq_api_key", "gsk_testkey")
    service = GroqService()

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"unexpected": "data"}
    mock_resp.raise_for_status = MagicMock()

    async def _run():
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(GroqServiceError, match="Unexpected response shape"):
                await service.chat_completion([{"role": "user", "content": "Hello"}])

    asyncio.run(_run())


def test_ai_service_layer1_router_bypass_groq():
    """Verify Layer 1 local router handles standard inquiries with $0 API call."""
    async def _run():
        # 1. Greeting
        reply1 = await get_ai_response("Sara", "sara@example.com", "Hi there")
        assert "Logic Pulse" in reply1

        # 2. Timing
        reply2 = await get_ai_response("Sara", "sara@example.com", "What are your hours?")
        assert "9 AM to 5 PM" in reply2

        # 3. CEO
        reply3 = await get_ai_response("Sara", "sara@example.com", "Who is the CEO?")
        assert "Muhammad Ramzan" in reply3

    asyncio.run(_run())


def test_ai_service_fallback_on_groq_failure():
    """Verify ai_service returns polite human fallback message if Groq fails."""
    async def _run():
        with patch.object(GroqService, "chat_completion_with_usage", new_callable=AsyncMock, side_effect=RuntimeError("Groq Down")):
            reply = await get_ai_response(
                "Sara",
                "sara@example.com",
                "Can we integrate custom webhooks?",
                system_prompt="Custom technical agent",
            )
            assert "Our assistant is currently unavailable" in reply

    asyncio.run(_run())
