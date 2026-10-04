"""
tests/test_health_and_demo_mode.py — Phase H: Voice Health Endpoint & Demo Mode Tests
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.voice_service import send_vonage_sms


def test_voice_health_endpoint(client: TestClient):
    """Verify /api/v1/health/voice returns system readiness and cache telemetry."""
    resp = client.get("/api/v1/health/voice")
    assert resp.status_code == 200
    data = resp.json()

    assert "status" in data
    assert "groq_reachable" in data
    assert "whisper_provider" in data
    assert "caches_warm" in data
    assert "web_tts_cache_count" in data["caches_warm"]
    assert "vonage_pcm_cache_count" in data["caches_warm"]
    assert "last_5_turn_totals" in data
    assert isinstance(data["last_5_turn_totals"], list)


def test_demo_mode_suppresses_sms(monkeypatch):
    """Verify DEMO_MODE=true disables outbound SMS sending for demo safety."""
    import asyncio
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "vonage_api_key", "mock_key")
    monkeypatch.setattr(settings, "vonage_api_secret", "mock_secret")
    monkeypatch.setattr(settings, "vonage_from_number", "12345")

    async def _run():
        status = await send_vonage_sms("+923001234567", "Your appointment is confirmed.")
        assert status == "disabled"

    asyncio.run(_run())
