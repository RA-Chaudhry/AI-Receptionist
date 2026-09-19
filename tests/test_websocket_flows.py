"""
tests/test_websocket_flows.py — End-to-End WebSocket Testing for Browser & Vonage Pipelines
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.services.voice_service import VADProcessor


def test_browser_voice_websocket_speech_turn(client: TestClient, mock_voice_environment, mock_groq):
    """
    Test full voice turn on /api/v1/voice/stream:
    Binary audio -> speech_ended -> transcript -> response_start -> audio_chunk -> response_end
    """
    with client.websocket_connect("/api/v1/voice/stream?phone_number=03001234567") as ws:
        # 1. Send binary audio chunks (listening event)
        dummy_chunk = b"\x00\x00" * 400
        ws.send_bytes(dummy_chunk)
        msg1 = ws.receive_json()
        assert msg1["type"] == "listening"

        # 2. Trigger speech_ended event
        ws.send_text(json.dumps({"type": "speech_ended"}))

        # 3. Receive transcript event
        transcript_msg = ws.receive_json()
        assert transcript_msg["type"] == "transcript"
        assert len(transcript_msg["data"]) > 0

        # 4. Receive response_start event
        resp_start_msg = ws.receive_json()
        assert resp_start_msg["type"] == "response_start"

        # 5. Receive audio_chunk events
        chunk_msg = ws.receive_json()
        assert chunk_msg["type"] == "audio_chunk"
        assert "data" in chunk_msg
        assert chunk_msg["index"] == 0

        # Receive remaining chunks until response_end
        next_msg = ws.receive_json()
        while next_msg["type"] == "audio_chunk":
            next_msg = ws.receive_json()

        assert next_msg["type"] == "response_end"
        assert len(next_msg.get("data", "")) > 0


def test_browser_voice_websocket_interrupt(client: TestClient):
    """Test sending interrupt signal over /api/v1/voice/stream."""
    with client.websocket_connect("/api/v1/voice/stream") as ws:
        ws.send_text(json.dumps({"type": "interrupt"}))
        resp = ws.receive_json()
        assert resp["type"] == "voice_interrupted"


def test_browser_voice_websocket_goodbye(client: TestClient, monkeypatch, mock_voice_environment):
    """Test that saying goodbye triggers goodbye event and disconnect."""
    # Configure mock whisper to return 'goodbye'
    mock_voice_environment["whisper"].transcript = "goodbye"

    with client.websocket_connect("/api/v1/voice/stream") as ws:
        ws.send_bytes(b"\x00\x00" * 400)
        ws.receive_json()  # listening

        ws.send_text(json.dumps({"type": "speech_ended"}))
        transcript_msg = ws.receive_json()
        assert transcript_msg["type"] == "transcript"
        assert transcript_msg["data"] == "goodbye"

        goodbye_msg = ws.receive_json()
        assert goodbye_msg["type"] == "goodbye"


def test_vonage_websocket_initial_handshake_and_greeting(client: TestClient, mock_voice_environment):
    """Test Vonage audio WebSocket connection, initial metadata, and welcome greeting."""
    with client.websocket_connect("/api/v1/vonage/stream") as ws:
        # Send Vonage metadata handshake
        meta = {"uuid": "vonage-test-session-001", "from": "+923001234567"}
        ws.send_text(json.dumps(meta))

        # First incoming message to caller should be binary PCM frames or clear control
        # Receive at least 1 binary frame from welcome greeting stream
        data = ws.receive_bytes()
        assert isinstance(data, bytes)
        assert len(data) == 640  # 20ms frame of 16kHz 16-bit mono PCM
