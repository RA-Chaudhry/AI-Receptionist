"""
tests/test_groq_client_safety.py — Phase D Tests
Verifies Groq client safety:
1. Fallback to HTTP/1.1 if h2 is unavailable.
2. Shared client lifecycle, connection retry, and streaming fallback.
3. HTTP 429/5xx status-only handling.
4. delta.reasoning suppression (only clean delta.content is yielded).
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.core.config import settings
from app.services.groq_service import (
    GroqService,
    GroqServiceError,
    close_shared_client,
    get_shared_client,
)


def test_http1_fallback_when_h2_missing(monkeypatch):
    """Verify get_shared_client gracefully creates an HTTP/1.1 client if h2 is missing."""
    import sys
    monkeypatch.setitem(sys.modules, "h2", None)

    async def _run():
        await close_shared_client()
        client = get_shared_client()
        assert client is not None
        assert not client.is_closed
        await close_shared_client()

    asyncio.run(_run())


def test_groq_stream_suppresses_reasoning_deltas(monkeypatch):
    """Verify that delta.reasoning is ignored and ONLY delta.content is yielded."""
    monkeypatch.setattr("app.core.config.settings.groq_api_key", "gsk_test_12345")

    # Simulate SSE lines with reasoning deltas mixed with content deltas
    sse_lines = [
        b'data: {"choices": [{"delta": {"reasoning": "Thinking step 1..."}}]}',
        b'data: {"choices": [{"delta": {"reasoning": "Thinking step 2..."}}]}',
        b'data: {"choices": [{"delta": {"content": "Hello, "}}]}',
        b'data: {"choices": [{"delta": {"reasoning": "Thinking step 3..."}}]}',
        b'data: {"choices": [{"delta": {"content": "how may I assist you?"}}]}',
        b'data: [DONE]',
    ]

    class FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass
        async def aiter_lines(self):
            for line in sse_lines:
                yield line.decode("utf-8")
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    async def _run():
        groq_svc = GroqService()
        with patch.object(httpx.AsyncClient, "stream", return_value=FakeResponse()):
            yielded = []
            async for token in groq_svc.chat_completion_stream([{"role": "user", "content": "hi"}]):
                yielded.append(token)

            full_text = "".join(yielded)
            assert full_text == "Hello, how may I assist you?"
            assert "Thinking" not in full_text
            assert "reasoning" not in full_text

    asyncio.run(_run())


def test_groq_streaming_connection_error_fallback(monkeypatch, caplog):
    """Verify that if streaming encounters a connection error, a clean fallback is yielded."""
    monkeypatch.setattr("app.core.config.settings.groq_api_key", "gsk_test_12345")

    class FailingResponseContext:
        async def __aenter__(self):
            raise httpx.ConnectError("Simulated connection failure")
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    async def _run():
        groq_svc = GroqService()
        with patch.object(httpx.AsyncClient, "stream", return_value=FailingResponseContext()):
            yielded = []
            with caplog.at_level(logging.WARNING):
                async for token in groq_svc.chat_completion_stream([{"role": "user", "content": "hi"}]):
                    yielded.append(token)

            full_text = "".join(yielded)
            assert len(yielded) > 0
            assert "apologize" in full_text.lower() or "trouble" in full_text.lower()
            # Ensure no API keys were logged
            for record in caplog.records:
                assert "gsk_test" not in record.message

    asyncio.run(_run())


def test_groq_http_status_error_no_secret_leak(monkeypatch, caplog):
    """Verify that HTTP 429/500 errors log status-only and do not leak authorization tokens."""
    monkeypatch.setattr("app.core.config.settings.groq_api_key", "gsk_secret_token_never_log")

    mock_resp = MagicMock()
    mock_resp.status_code = 429

    async def _run():
        groq_svc = GroqService()
        with patch.object(httpx.AsyncClient, "post", side_effect=httpx.HTTPStatusError("429 Too Many Requests", request=MagicMock(), response=mock_resp)):
            with caplog.at_level(logging.WARNING):
                with pytest.raises(GroqServiceError):
                    await groq_svc.chat_completion_with_usage([{"role": "user", "content": "hi"}])

            # Verify no secret leaked into log records
            for record in caplog.records:
                assert "gsk_secret" not in record.message
                assert "Authorization" not in record.message

    asyncio.run(_run())
