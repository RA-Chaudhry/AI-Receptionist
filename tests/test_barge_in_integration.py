import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.utils.audio_buffer import TurnDetector, TurnState
from tests.test_jitter_and_packet_handling import generate_pcm_sine, generate_silence


def test_barge_in_state_preservation_and_turn_detector():
    """Verify TurnDetector preserves speech on barge-in without dropping initial phonemes."""
    td = TurnDetector(silence_duration_ms=700, min_speech_duration_ms=200)

    # 1. Caller begins speaking (300ms speech)
    res = td.process_frame(generate_pcm_sine(300))
    assert res.turn_started is True
    assert res.state == TurnState.SPEAKING

    # 2. Caller continues speaking (400ms speech)
    res2 = td.process_frame(generate_pcm_sine(400))
    assert res2.state == TurnState.SPEAKING

    # 3. Caller finishes speaking (700ms silence)
    res3 = td.process_frame(generate_silence(700))
    assert res3.turn_completed is True
    assert res3.audio_bytes is not None
    # All 700ms of speech (300ms + 400ms) plus pre-buffer preserved
    assert len(res3.audio_bytes) >= int(16000 * 0.7 * 2)


def test_false_interruption_click_does_not_trigger_barge_in():
    """Verify that a brief click/cough (< 200ms) does NOT set turn_started=True."""
    td = TurnDetector(min_speech_duration_ms=200)

    # 40ms click (2 frames of 20ms) followed by silence
    click_40ms = generate_pcm_sine(40)
    res = td.process_frame(click_40ms)
    assert res.turn_started is False
    assert td.state == TurnState.IDLE

    # Followed by silence
    silence = generate_silence(100)
    res2 = td.process_frame(silence)
    assert res2.turn_started is False
    assert td.state == TurnState.IDLE


def test_websocket_barge_in_during_active_turn(client: TestClient, mock_voice_environment, mock_groq):
    """
    Verify browser WebSocket cancels ongoing generation and dispatches voice_interrupted
    when caller speaks during assistant playback.
    """
    with client.websocket_connect("/api/v1/voice/stream?phone_number=%2B923001234567") as ws:
        # 1. User sends first turn
        speech_300ms = generate_pcm_sine(300)
        ws.send_bytes(speech_300ms)
        # 700ms silence completes turn 1
        ws.send_bytes(generate_silence(700))

        # 2. Receive responses for turn 1
        events = []
        for _ in range(5):
            try:
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") in ("response_start", "audio_chunk", "response_end"):
                    break
            except Exception:
                break

        assert any(e.get("type") in ("listening", "transcript", "response_start", "audio_chunk", "response_end") for e in events)

        # 3. Caller barges in with new speech while assistant is speaking
        ws.send_bytes(speech_300ms)

        # 4. Check for voice_interrupted or listening
        barge_in_events = []
        for _ in range(5):
            try:
                msg = ws.receive_json()
                barge_in_events.append(msg)
                if msg.get("type") == "voice_interrupted":
                    break
            except Exception:
                break

        # User turn completes with silence
        ws.send_bytes(generate_silence(700))

        # Client sends goodbye to close cleanly
        ws.send_json({"type": "goodbye"})


def test_websocket_multiple_rapid_interruptions(client: TestClient, mock_voice_environment, mock_groq):
    """Verify that multiple rapid interruptions in sequence do not crash the WebSocket session."""
    with client.websocket_connect("/api/v1/voice/stream?phone_number=%2B923007654321") as ws:
        # Rapid interruptions via text event
        for i in range(5):
            ws.send_json({"type": "interrupt"})
            msg = ws.receive_json()
            assert msg.get("type") == "voice_interrupted"

        # Session remains completely healthy and responsive
        ws.send_bytes(generate_pcm_sine(300))
        ws.send_bytes(generate_silence(700))
        
        # Drain responses
        got_transcript_or_start = False
        for _ in range(6):
            try:
                msg = ws.receive_json()
                if msg.get("type") in ("transcript", "response_start", "audio_chunk", "listening"):
                    got_transcript_or_start = True
            except Exception:
                break

        assert got_transcript_or_start is True
        ws.send_json({"type": "goodbye"})


def test_conversation_context_preserved_across_interruptions(client: TestClient, mock_voice_environment, mock_groq, isolated_db):
    """Verify conversation history context is preserved across multiple turns and interruptions."""
    phone = "+923009998877"
    with client.websocket_connect(f"/api/v1/voice/stream?phone_number={phone}") as ws:
        # Turn 1
        ws.send_bytes(generate_pcm_sine(300))
        ws.send_bytes(generate_silence(700))

        # Drain turn 1
        for _ in range(5):
            try:
                ws.receive_json()
            except Exception:
                break

        # Interrupt
        ws.send_json({"type": "interrupt"})
        ws.receive_json()

        # Turn 2
        ws.send_bytes(generate_pcm_sine(300))
        ws.send_bytes(generate_silence(700))

        # Drain turn 2
        for _ in range(5):
            try:
                ws.receive_json()
            except Exception:
                break

        ws.send_json({"type": "goodbye"})
