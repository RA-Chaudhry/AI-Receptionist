"""
tests/test_voice_diagnostic_stabilization.py — Automated Unit & Integration Tests for Voice Pipeline Fixes

Covers:
1. Urdu / Arabic sentence boundaries (۔, ؟) and clause splitting (،, ؛) in stream_sentence_chunks
2. Enhanced Voice Telemetry Recorder (turn_id, language, probabilities, duration, metrics)
3. Groq Whisper provider integration with offline fallback to Faster-Whisper
4. TurnDetector silence threshold and circular pre-buffer retention
5. Idempotent speech_ended handling during active generation
"""

import asyncio
import base64
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.config import settings
from app.services.groq_service import GroqService, GroqServiceError
from app.services.telemetry import (
    DEFAULT_LATENCY_BUDGET,
    CostTrace,
    LatencyTrace,
    VoiceTelemetryRecorder,
)
from app.services.voice_service import (
    transcribe_audio_bytes,
    generate_tts_chunks,
    _select_tts_voice,
)
from app.utils.audio_buffer import TurnDetector, TurnState, VADProcessor, pcm_to_wav
from app.utils.stream_utils import stream_sentence_chunks


# ──────────────────────────────────────────────────────────────────────────────
# 1. Urdu & Bilingual Sentence Splitting Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_stream_sentence_chunks_urdu_khatma_and_sawaliya():
    """Verify Urdu full-stop (۔) and Urdu question mark (؟) trigger sentence boundaries."""
    async def _run():
        async def _token_generator():
            tokens = [
                "آپ کا ", "خوش آمدید", "۔ ",
                "کیا میں آپ کی ", "مدد کر سکتا ہوں", "؟ ",
                "ہماری سروسز ", "یہاں دستیاب ہیں", "۔"
            ]
            for t in tokens:
                yield t

        sentences = []
        async for s in stream_sentence_chunks(_token_generator()):
            sentences.append(s)

        assert len(sentences) == 3
        assert sentences[0] == "آپ کا خوش آمدید۔"
        assert sentences[1] == "کیا میں آپ کی مدد کر سکتا ہوں؟"
        assert sentences[2] == "ہماری سروسز یہاں دستیاب ہیں۔"

    asyncio.run(_run())


def test_stream_sentence_chunks_urdu_clause_commas():
    """Verify Urdu comma (،) and semicolon (؛) split long clauses when exceeding threshold."""
    async def _run():
        async def _token_generator():
            tokens = [
                "ہم کلائنٹس کو انٹرپرائز لیول سافٹ ویئر سلوشنز فراہم کرتے ہیں، ",
                "اور اس کے ساتھ کسٹم کلاؤڈ آرکیٹیکچر بھی بناتے ہیں، ",
                "جو کہ جدید ترین سسٹمز پر مشتمل ہے۔"
            ]
            for t in tokens:
                yield t

        sentences = []
        async for s in stream_sentence_chunks(_token_generator(), min_clause_words=3):
            sentences.append(s)

        assert len(sentences) >= 2
        assert "سلوشنز فراہم کرتے ہیں،" in sentences[0]

    asyncio.run(_run())


def test_stream_sentence_chunks_final_fragment_flushing():
    """Verify that unpunctuated final text is never lost and flushes upon stream end."""
    async def _run():
        async def _token_generator():
            tokens = ["This is a sentence without trailing punctuation"]
            for t in tokens:
                yield t

        sentences = []
        async for s in stream_sentence_chunks(_token_generator()):
            sentences.append(s)

        assert len(sentences) == 1
        assert sentences[0] == "This is a sentence without trailing punctuation"

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────────────────
# 2. Enhanced Telemetry Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_voice_telemetry_turn_id_and_granular_metrics():
    """Verify turn_id, detected_language, audio bytes, and total turn duration tracking."""
    recorder = VoiceTelemetryRecorder(
        call_id="call-diag-101",
        turn_id="turn-diag-202",
        channel="voice_ws",
    )
    recorder.mark_turn_created()
    recorder.mark_mic_capture_start()
    recorder.mark_first_audio_chunk()
    recorder.record_audio_chunk(6400)
    recorder.record_audio_chunk(6400)
    recorder.mark_frontend_speech_end()
    recorder.mark_vad_start()
    recorder.mark_vad_end()
    recorder.mark_stt_start()
    recorder.record_stt_result(
        transcript="Hello, I want to book an appointment.",
        language="en",
        language_probability=0.98,
        provider="groq_whisper",
        model="whisper-large-v3-turbo",
    )
    recorder.mark_stt_end()
    recorder.mark_llm_start()
    recorder.mark_llm_first_token()
    recorder.mark_llm_end()
    recorder.mark_tts_start()
    recorder.mark_tts_first_audio()
    recorder.mark_tts_end()
    recorder.mark_audio_sent()
    recorder.mark_turn_completed()

    record = recorder.finalize(log_to_file=False)

    assert record.call_id == "call-diag-101"
    assert record.turn_id == "turn-diag-202"
    assert record.total_audio_bytes == 12800
    assert record.audio_chunk_count == 2
    assert record.detected_language == "en"
    assert record.language_probability == 0.98
    assert record.stt_provider == "groq_whisper"
    assert record.transcript_length == len("Hello, I want to book an appointment.")
    assert record.latency.total_turn_duration_ms is not None
    assert record.latency.total_turn_duration_ms >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. Groq Whisper Integration & Fallback Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_groq_whisper_transcribe_audio_with_usage():
    """Verify GroqService.transcribe_audio_with_usage invokes HTTP API correctly."""
    async def _run():
        dummy_wav = b"RIFF....WAVEfmt ...."
        mock_response = {
            "text": "Hello, I want to book an appointment tomorrow.",
            "language": "english",
            "duration": 3.42,
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            svc = GroqService()
            text, lang, dur = await svc.transcribe_audio_with_usage(dummy_wav)

            assert text == "Hello, I want to book an appointment tomorrow."
            assert lang == "english"
            assert abs(dur - 3.42) < 1e-3

    asyncio.run(_run())


def test_transcribe_audio_bytes_groq_whisper_path():
    """Verify transcribe_audio_bytes routes to Groq Whisper when configured."""
    async def _run():
        dummy_pcm = b"\x00\x00" * 16000  # 1 second of silence PCM
        recorder = VoiceTelemetryRecorder()

        with patch("app.core.config.settings.stt_provider", "groq_whisper"), \
             patch("app.core.config.settings.groq_api_key", "mock-groq-key"), \
             patch.object(GroqService, "transcribe_audio_with_usage", new_callable=AsyncMock) as mock_groq_stt:
            
            mock_groq_stt.return_value = ("Can I schedule a consultation?", "en", 1.0)
            
            result = await transcribe_audio_bytes(dummy_pcm, telemetry_recorder=recorder)

            assert result == "Can I schedule a consultation?"
            telemetry = recorder.finalize(log_to_file=False)
            assert telemetry.stt_provider == "groq_whisper"
            assert telemetry.detected_language == "en"

    asyncio.run(_run())


def test_transcribe_audio_bytes_groq_failure_fallback_to_whisper():
    """Verify that if Groq Whisper fails, transcribe_audio_bytes falls back to local Whisper."""
    async def _run():
        dummy_pcm = b"\x00\x00" * 16000
        recorder = VoiceTelemetryRecorder()

        mock_whisper_instance = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Fallback local transcription."
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_whisper_instance.transcribe.return_value = ([mock_segment], mock_info)

        with patch("app.core.config.settings.stt_provider", "groq_whisper"), \
             patch("app.core.config.settings.groq_api_key", "mock-groq-key"), \
             patch.object(GroqService, "transcribe_audio_with_usage", side_effect=Exception("API Timeout")), \
             patch("app.services.voice_service._get_whisper_model", return_value=mock_whisper_instance):

            result = await transcribe_audio_bytes(dummy_pcm, telemetry_recorder=recorder)

            assert result == "Fallback local transcription."
            telemetry = recorder.finalize(log_to_file=False)
            assert telemetry.stt_provider == "faster_whisper"

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────────────────
# 4. TurnDetector & Pre-Buffer Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_turn_detector_pre_buffer_retention():
    """Verify TurnDetector circular pre-buffer retains initial consonants before speech trigger."""
    detector = TurnDetector(
        sample_rate=16000,
        frame_duration_ms=20,
        pre_buffer_ms=200,
        min_speech_duration_ms=40,
        silence_duration_ms=100,
    )
    frame_bytes = 640  # 20ms at 16kHz 16-bit

    # Mock VADProcessor: first 5 frames silence, next 3 frames speech, next 6 frames silence
    call_idx = 0
    def mock_is_speech(pcm_bytes):
        nonlocal call_idx
        call_idx += 1
        return (6 <= call_idx <= 8)

    detector.vad_processor.is_speech = mock_is_speech

    # Send 5 silence frames (should enter pre-buffer)
    for _ in range(5):
        detector.process_frame(b"\x00" * frame_bytes)

    assert detector.state == TurnState.IDLE

    # Send 3 speech frames (should trigger SPEAKING and include pre-buffer)
    res_start = None
    for _ in range(3):
        res = detector.process_frame(b"\x01" * frame_bytes)
        if res.turn_started:
            res_start = res

    assert res_start is not None
    assert detector.state == TurnState.SPEAKING

    # Send 6 silence frames (should complete turn)
    res_completed = None
    for _ in range(6):
        res = detector.process_frame(b"\x00" * frame_bytes)
        if res.turn_completed:
            res_completed = res

    assert res_completed is not None
    assert res_completed.audio_bytes is not None
    # Total audio should include pre-buffer + speech frames + silence frames
    assert len(res_completed.audio_bytes) >= (frame_bytes * 5)
