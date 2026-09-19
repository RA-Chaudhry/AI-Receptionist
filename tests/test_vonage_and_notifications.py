import asyncio
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.voice_service import send_vonage_sms, _sms_history
from app.services.email_service import EmailService


@pytest.fixture
def client():
    old_env = settings.environment
    settings.environment = "testing"
    with TestClient(app) as c:
        yield c
    settings.environment = old_env


def test_vonage_answer_webhook(client):
    response = client.post("/api/v1/vonage/answer")
    assert response.status_code == 200
    ncco = response.json()
    assert isinstance(ncco, list)
    assert len(ncco) == 1
    assert ncco[0]["action"] == "connect"
    assert ncco[0]["endpoint"][0]["type"] == "websocket"
    assert "vonage/stream" in ncco[0]["endpoint"][0]["uri"]
    assert ncco[0]["endpoint"][0]["content-type"] == "audio/l16;rate=16000"


def test_vonage_event_webhook(client):
    response = client.post("/api/v1/vonage/event", json={"status": "answered"})
    assert response.status_code == 200


def test_vonage_sms_truncation_and_rate_limiting():
    async def _run():
        # Clear SMS rate limit history
        _sms_history.clear()
        
        # Configure Vonage settings for tests
        settings.vonage_api_key = "test_key"
        settings.vonage_api_secret = "test_secret"
        settings.vonage_from_number = "15005550006"
        settings.sms_rate_limit_count = 2
        settings.sms_rate_limit_window_seconds = 60
        
        long_msg = "A" * 500
        
        # Mock HTTP response from Vonage API
        async def mock_post(*args, **kwargs):
            data = kwargs.get("data", {})
            assert data.get("api_key") == "test_key"
            assert data.get("api_secret") == "test_secret"
            assert len(data.get("text", "")) <= 450
            if data.get("text", "").startswith("A" * 100):
                assert data.get("text", "").endswith("...")
                
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json = lambda: {
                "messages": [{"status": "0", "message-id": "12345"}]
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            # 1st SMS
            status1 = await send_vonage_sms("+1234567890", long_msg)
            assert status1 == "sent"
            
            # 2nd SMS
            status2 = await send_vonage_sms("+1234567890", "Hello 2")
            assert status2 == "sent"
            
            # 3rd SMS (should be rate limited)
            status3 = await send_vonage_sms("+1234567890", "Hello 3")
            assert status3 == "rate_limited"

    asyncio.run(_run())


def test_email_service_smtp():
    async def _run():
        settings.email_provider = "smtp"
        settings.from_email = "test@logicpulse.com"
        
        email_service = EmailService()
        
        mock_smtp = MagicMock()
        with patch("smtplib.SMTP", return_value=mock_smtp):
            success = await email_service.send_email("user@example.com", "Test Subject", "Test Body")
            assert success is True
            mock_smtp.__enter__.return_value.send_message.assert_called_once()

    asyncio.run(_run())


def test_email_service_resend():
    async def _run():
        settings.email_provider = "resend"
        settings.resend_api_key = "re_testkey"
        settings.from_email = "test@logicpulse.com"
        
        email_service = EmailService()
        
        async def mock_post(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            return mock_resp
            
        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            success = await email_service.send_email("user@example.com", "Test Subject", "Test Body")
            assert success is True

    asyncio.run(_run())


def test_email_service_sendgrid():
    async def _run():
        settings.email_provider = "sendgrid"
        settings.sendgrid_api_key = "sg_testkey"
        settings.from_email = "test@logicpulse.com"
        
        email_service = EmailService()
        
        async def mock_post(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 202
            mock_resp.raise_for_status = MagicMock()
            return mock_resp
            
        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            success = await email_service.send_email("user@example.com", "Test Subject", "Test Body")
            assert success is True

    asyncio.run(_run())

