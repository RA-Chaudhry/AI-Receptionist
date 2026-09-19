"""
tests/test_streaming_pipeline.py — Phase 3 Streaming LLM & Voice Pipeline Tests

Tests:
1. stream_sentence_chunks token buffering and boundary detection
2. GroqService.chat_completion_stream Server-Sent Events (SSE) parsing
3. get_ai_response_stream incremental generation and TTFT recording
4. Full-duplex WebSocket streaming audio transmission and interruption cancellation
5. Empirical TTFA (Time-To-First-Audio) latency comparison: Batch vs Streaming
"""

import asyncio
import base64
import json
import time
import pytest
from fastapi.testclient import TestClient

from app.services.ai_service import get_ai_response, get_ai_response_stream
from app.services.groq_service import GroqService, GroqServiceError
from app.services.telemetry import VoiceTelemetryRecorder
from app.services.voice_service import generate_tts_chunks
from app.utils.stream_utils import stream_sentence_chunks


# ──────────────────────────────────────────────────────────────────────────────
# 1. Sentence Buffering & Token Splitting Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_stream_sentence_chunks_basic_punctuation():
    """Verify sentence boundaries on periods, exclamation marks, and question marks."""
    async def _run():
        async def _token_generator():
            tokens = ["Hello ", "there! ", "How ", "can I ", "help you ", "today? ", "We offer ", "cloud ", "AI."]
            for t in tokens:
                yield t

        results = []
        async for sentence in stream_sentence_chunks(_token_generator()):
            results.append(sentence)

        assert results == [
            "Hello there!",
            "How can I help you today?",
            "We offer cloud AI.",
        ]

    asyncio.run(_run())


def test_stream_sentence_chunks_abbreviations_and_decimals():
    """Ensure abbreviations (Dr., CEO.) and decimals (3.5) are not prematurely split."""
    async def _run():
        async def _token_generator():
            tokens = ["Please ", "meet ", "Dr. ", "Muhammad ", "Ramzan ", "at 3.5 ", "PM. ", "He is ", "our CEO."]
            for t in tokens:
                yield t

        results = []
        async for sentence in stream_sentence_chunks(_token_generator()):
            results.append(sentence)

        assert len(results) == 2
        assert "Dr. Muhammad Ramzan at 3.5 PM." in results[0]
        assert "He is our CEO." in results[1]

    asyncio.run(_run())


def test_stream_sentence_chunks_clause_splitting():
    """Verify long sentences split on clause commas when exceeding word threshold."""
    async def _run():
        async def _token_generator():
            tokens = [
                "We build custom enterprise cloud architectures, ",
                "and we also deliver high performance voice assistants, ",
                "for hospitals and businesses.",
            ]
            for t in tokens:
                yield t

        results = []
        async for sentence in stream_sentence_chunks(_token_generator(), min_clause_words=3):
            results.append(sentence)

        assert len(results) == 3
        assert "cloud architectures," in results[0]
        assert "voice assistants," in results[1]
        assert "businesses." in results[2]

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────────────────
# 2. GroqService Streaming Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_groq_service_chat_completion_stream_offline(mock_groq):
    """Verify FakeGroqService streams tokens progressively."""
    async def _run():
        messages = [{"role": "user", "content": "Tell me about dental software."}]

        tokens = []
        async for token in mock_groq.chat_completion_stream(messages):
            tokens.append(token)

        full_text = "".join(tokens).strip()
        assert len(tokens) > 1
        assert "multi-tenant cloud software" in full_text

    asyncio.run(_run())


def test_groq_service_streaming_error_handling(monkeypatch):
    """Verify GroqServiceError is raised if streaming connection fails."""
    async def _run():
        fake_service = GroqService()
        monkeypatch.setattr(fake_service, "base_url", "http://invalid-unreachable-host:9999")

        with pytest.raises(GroqServiceError):
            async for _ in fake_service.chat_completion_stream([{"role": "user", "content": "hi"}]):
                pass

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────────────────
# 3. AI Service Streaming & TTFT Telemetry Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_get_ai_response_stream_local_router(mock_groq):
    """Verify local router responses are streamed instantly with 0 tokens recorded."""
    async def _run():
        recorder = VoiceTelemetryRecorder(call_id="test-router-stream", channel="test")
        sentences = []

        async for sentence in get_ai_response_stream(
            user_name="Test Caller",
            user_email="test@example.com",
            message="What are your office hours?",
            telemetry_recorder=recorder,
        ):
            sentences.append(sentence)

        assert len(sentences) >= 1
        assert any("Monday to Friday" in s or "9 AM" in s for s in sentences)
        assert recorder.cost.provider == "local_router"
        assert recorder.cost.estimated_cost_usd == 0.0
        assert recorder.latency.llm_first_token_time is not None

    asyncio.run(_run())


def test_get_ai_response_stream_groq_layer(mock_groq):
    """Verify LLM path streams sentence chunks and records TTFT."""
    async def _run():
        recorder = VoiceTelemetryRecorder(call_id="test-groq-stream", channel="test")
        sentences = []

        async for sentence in get_ai_response_stream(
            user_name="Ali",
            user_email="ali@example.com",
            message="Can you build custom dental clinic systems?",
            telemetry_recorder=recorder,
        ):
            sentences.append(sentence)

        assert len(sentences) >= 1
        assert any("multi-tenant cloud software" in s for s in sentences)
        assert recorder.cost.provider == "groq_stream"
        assert recorder.latency.llm_ttft_ms is not None
        assert recorder.latency.llm_total_latency_ms is not None

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────────────────
# 4. WebSocket Full-Duplex Streaming & Interruption Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_websocket_streaming_speech_turn(client: TestClient, mock_groq, mock_voice_environment):
    """Verify full-duplex WebSocket streams response_start, audio_chunk frames, and response_end."""
    with client.websocket_connect("/api/v1/voice/stream?phone_number=03001234567") as ws:
        # Send synthetic audio frames
        ws.send_bytes(b"\x00\x00" * 320)
        resp1 = ws.receive_json()
        assert resp1["type"] == "listening"

        # Signal speech ended
        ws.send_text(json.dumps({"type": "speech_ended"}))

        # Expect transcript
        transcript_msg = ws.receive_json()
        assert transcript_msg["type"] == "transcript"

        # Expect response_start
        start_msg = ws.receive_json()
        assert start_msg["type"] == "response_start"

        # Expect streaming audio_chunk frames with sequential indexing
        audio_chunks = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "response_end":
                break
            assert msg["type"] == "audio_chunk"
            assert isinstance(msg["index"], int)
            audio_chunks.append(msg["data"])

        assert len(audio_chunks) >= 1
        # Each chunk is valid base64
        for chunk in audio_chunks:
            decoded = base64.b64decode(chunk)
            assert len(decoded) > 0


def test_websocket_streaming_interruption(client: TestClient, mock_groq, mock_voice_environment):
    """Verify client interrupt event cancels ongoing streaming task without crashing connection."""
    with client.websocket_connect("/api/v1/voice/stream?phone_number=03001234567") as ws:
        ws.send_bytes(b"\x00\x00" * 320)
        ws.receive_json()  # listening

        # Send interrupt event
        ws.send_text(json.dumps({"type": "interrupt"}))
        interrupted_msg = ws.receive_json()
        assert interrupted_msg["type"] == "voice_interrupted"

        # Connection should remain open and healthy for subsequent turns
        ws.send_bytes(b"\x00\x00" * 320)
        resp = ws.receive_json()
        assert resp["type"] == "listening"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Empirical TTFA Latency Comparison (Batch vs Streaming)
# ──────────────────────────────────────────────────────────────────────────────

def test_ttfa_latency_comparison_batch_vs_streaming(mock_groq, mock_voice_environment):
    """
    Measures and compares Time-To-First-Audio (TTFA) between:
    1. Batch mode: full response text generated first -> entire TTS synthesized -> audio dispatched.
    2. Streaming mode: first sentence token boundary -> immediate TTS chunk dispatched.
    """
    async def _run():
        # 1. Measure Batch Mode TTFA
        recorder_batch = VoiceTelemetryRecorder(call_id="batch-measurement", channel="benchmark")
        recorder_batch.mark_audio_received()
        recorder_batch.mark_vad_start()
        recorder_batch.mark_vad_end()

        full_text = await get_ai_response(
            user_name="Benchmark User",
            user_email="bench@example.com",
            message="Tell me in detail about your cloud software development capabilities.",
            system_prompt="You are an enterprise AI concierge.",
            telemetry_recorder=recorder_batch,
        )
        batch_first_chunk = False
        async for _ in generate_tts_chunks(full_text, telemetry_recorder=recorder_batch):
            if not batch_first_chunk:
                recorder_batch.mark_audio_sent()
                batch_first_chunk = True
        recorder_batch.finalize()
        batch_ttfa = recorder_batch.latency.ttfa_ms

        # 2. Measure Streaming Mode TTFA
        recorder_stream = VoiceTelemetryRecorder(call_id="stream-measurement", channel="benchmark")
        recorder_stream.mark_audio_received()
        recorder_stream.mark_vad_start()
        recorder_stream.mark_vad_end()

        stream_first_chunk = False
        async for sentence in get_ai_response_stream(
            user_name="Benchmark User",
            user_email="bench@example.com",
            message="Tell me in detail about your cloud software development capabilities.",
            system_prompt="You are an enterprise AI concierge.",
            telemetry_recorder=recorder_stream,
        ):
            async for _ in generate_tts_chunks(sentence, telemetry_recorder=recorder_stream):
                if not stream_first_chunk:
                    recorder_stream.mark_audio_sent()
                    stream_first_chunk = True
                    break
            if stream_first_chunk:
                break

        recorder_stream.finalize()
        stream_ttfa = recorder_stream.latency.ttfa_ms

        # Assertions
        assert batch_ttfa is not None
        assert stream_ttfa is not None
        evaluation = recorder_stream.latency.evaluate_budget()
        assert evaluation["within_total_budget"] is True
        print(f"\n[TTFA Benchmark Result] Batch TTFA: {batch_ttfa:.2f}ms | Streaming TTFA: {stream_ttfa:.2f}ms")

    asyncio.run(_run())
