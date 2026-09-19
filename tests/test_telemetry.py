"""
tests/test_telemetry.py — Unit and integration tests for Phase 2 Voice Telemetry & Latency Budget
"""

import asyncio
import os
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.telemetry import (
    DEFAULT_LATENCY_BUDGET,
    CostTrace,
    LatencyBudget,
    LatencyTrace,
    VoiceTelemetryRecorder,
    TELEMETRY_LOG_FILE,
)
from app.services.groq_service import GroqCompletionResult, GroqService
from app.services.ai_service import get_ai_response
from app.services.voice_service import transcribe_audio_bytes, generate_tts_chunks


# ──────────────────────────────────────────────────────────────────────────────
# 1. Latency Budget Configuration Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_latency_budget_targets():
    """Verify target latency budget values as specified in requirements."""
    budget = DEFAULT_LATENCY_BUDGET
    assert budget.vad_end_detection_target_ms == 150.0
    assert budget.stt_finalization_target_ms == 300.0
    assert budget.llm_ttft_target_ms == 400.0
    assert budget.tts_first_audio_target_ms == 250.0
    assert budget.network_buffering_target_ms == 100.0
    assert budget.total_target_ms == 1200.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. LatencyTrace Calculations & Delta Timings
# ──────────────────────────────────────────────────────────────────────────────

def test_latency_trace_calculations():
    """Verify delta calculation in milliseconds using mock perf_counter timestamps."""
    t0 = 100.000
    trace = LatencyTrace(
        audio_received_time=t0,
        vad_start_time=t0 + 0.010,
        vad_end_time=t0 + 0.150,        # vad_latency = 140ms
        stt_start_time=t0 + 0.155,
        stt_end_time=t0 + 0.450,        # stt_latency = 295ms
        llm_start_time=t0 + 0.455,
        llm_first_token_time=t0 + 0.755, # llm_ttft = 300ms
        llm_end_time=t0 + 0.800,        # llm_total = 345ms
        tts_start_time=t0 + 0.805,
        tts_first_audio_time=t0 + 1.005, # tts_ttfb = 200ms
        tts_end_time=t0 + 1.150,        # tts_total = 345ms
        audio_sent_time=t0 + 1.055,     # network_buffering = 50ms, ttfa = (1.055 - 0.150) = 905ms
    )

    metrics = trace.get_metrics_summary()
    assert abs(metrics["vad_latency_ms"] - 140.0) < 1e-3
    assert abs(metrics["stt_latency_ms"] - 295.0) < 1e-3
    assert abs(metrics["llm_ttft_ms"] - 300.0) < 1e-3
    assert abs(metrics["llm_total_latency_ms"] - 345.0) < 1e-3
    assert abs(metrics["tts_ttfb_ms"] - 200.0) < 1e-3
    assert abs(metrics["tts_total_latency_ms"] - 345.0) < 1e-3
    assert abs(metrics["network_buffering_ms"] - 50.0) < 1e-3
    assert abs(metrics["ttfa_ms"] - 905.0) < 1e-3


def test_latency_budget_evaluation_within_and_exceeded():
    """Verify budget compliance evaluation against target 1200ms."""
    # Case 1: Within budget (900ms < 1200ms)
    trace_pass = LatencyTrace(
        vad_end_time=10.0,
        audio_sent_time=10.9,
    )
    eval_pass = trace_pass.evaluate_budget()
    assert eval_pass["within_total_budget"] is True
    assert abs(eval_pass["measured_ttfa_ms"] - 900.0) < 1e-3
    assert abs(eval_pass["budget_delta_ms"] - (-300.0)) < 1e-3

    # Case 2: Exceeding budget (1500ms > 1200ms)
    trace_fail = LatencyTrace(
        vad_end_time=10.0,
        audio_sent_time=11.5,
    )
    eval_fail = trace_fail.evaluate_budget()
    assert eval_fail["within_total_budget"] is False
    assert abs(eval_fail["measured_ttfa_ms"] - 1500.0) < 1e-3
    assert abs(eval_fail["budget_delta_ms"] - 300.0) < 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cost Accounting Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_cost_trace_groq():
    """Verify Groq token pricing estimation."""
    cost = CostTrace(
        call_id="call-123",
        request_id="req-456",
        provider="groq",
        model="llama-3.1-8b-instant",
        input_tokens=1000,
        output_tokens=500,
    )
    est = cost.calculate_estimated_cost()
    # 1000 in * $0.05/1M = $0.00005, 500 out * $0.08/1M = $0.00004 -> Total $0.00009
    assert abs(est - 0.00009) < 1e-7


def test_cost_trace_local_router():
    """Verify local router has $0.00 cost."""
    cost = CostTrace(
        call_id="call-123",
        request_id="req-456",
        provider="local_router",
        model="regex_rules",
        input_tokens=0,
        output_tokens=0,
    )
    assert cost.calculate_estimated_cost() == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. VoiceTelemetryRecorder Lifecycle & Logging
# ──────────────────────────────────────────────────────────────────────────────

def test_telemetry_recorder_lifecycle_and_logging(tmp_path, monkeypatch):
    """Test full recorder flow using time.perf_counter() and outputting to log file."""
    test_log = tmp_path / "telemetry_test.jsonl"
    monkeypatch.setattr("app.services.telemetry.TELEMETRY_LOG_FILE", str(test_log))

    recorder = VoiceTelemetryRecorder(call_id="test-call-id", request_id="test-req-id", channel="voice_ws")
    recorder.mark_audio_received()
    recorder.mark_vad_start()
    recorder.mark_vad_end()
    recorder.mark_stt_start()
    recorder.record_stt_duration(2.5)
    recorder.mark_stt_end()
    recorder.mark_llm_start()
    recorder.record_llm_usage(provider="groq", model="llama-3.1-8b-instant", input_tokens=120, output_tokens=30)
    recorder.mark_llm_end()
    recorder.mark_tts_start()
    recorder.mark_tts_first_audio()
    recorder.mark_audio_sent()
    recorder.mark_tts_end()

    telemetry = recorder.finalize(log_to_file=True)

    assert telemetry.call_id == "test-call-id"
    assert telemetry.request_id == "test-req-id"
    assert telemetry.cost.input_tokens == 120
    assert telemetry.cost.output_tokens == 30
    assert telemetry.cost.stt_audio_duration_sec == 2.5
    assert telemetry.latency.ttfa_ms is not None
    assert telemetry.latency.ttfa_ms >= 0.0

    # Verify JSONL log entry written
    assert os.path.exists(test_log)
    with open(test_log, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["call_id"] == "test-call-id"
        assert "metrics_summary" in data
        assert "budget_evaluation" in data


# ──────────────────────────────────────────────────────────────────────────────
# 5. Service-Level Telemetry Integrations
# ──────────────────────────────────────────────────────────────────────────────

def test_groq_service_usage_capture():
    """Verify GroqService captures token usage from response."""
    async def _run():
        service = GroqService()
        service.api_key = "test_key"

        fake_response = {
            "choices": [{"message": {"content": "Hello world"}}],
            "model": "llama-3.1-8b-instant",
            "usage": {"prompt_tokens": 42, "completion_tokens": 10, "total_tokens": 52},
        }

        mock_resp = MagicMock()
        mock_resp.json = lambda: fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            res = await service.chat_completion_with_usage([{"role": "user", "content": "Hi"}])
            assert res.content == "Hello world"
            assert res.input_tokens == 42
            assert res.output_tokens == 10
            assert res.total_tokens == 52

    asyncio.run(_run())


def test_ai_service_telemetry_local_router():
    """Verify Layer 1 local router triggers zero-cost telemetry."""
    async def _run():
        recorder = VoiceTelemetryRecorder()
        reply = await get_ai_response(
            user_name="Ali",
            user_email="ali@example.com",
            message="What are your office hours?",
            telemetry_recorder=recorder,
        )
        assert "9 AM to 5 PM" in reply
        assert recorder.cost.provider == "local_router"
        assert recorder.cost.input_tokens == 0
        assert recorder.latency.llm_ttft_ms is not None

    asyncio.run(_run())


def test_ai_service_telemetry_groq_layer():
    """Verify Layer 2 Groq triggers token usage telemetry."""
    async def _run():
        recorder = VoiceTelemetryRecorder()
        fake_result = GroqCompletionResult(
            content="We specialize in custom web apps.",
            input_tokens=150,
            output_tokens=25,
            total_tokens=175,
            model="llama-3.1-8b-instant",
        )

        with patch.object(GroqService, "chat_completion_with_usage", new_callable=AsyncMock, return_value=fake_result):
            # Query with custom system prompt that invokes Groq
            reply = await get_ai_response(
                user_name="Sara",
                user_email="sara@example.com",
                message="Tell me more about your architecture.",
                system_prompt="You are a senior tech lead.",
                telemetry_recorder=recorder,
            )
            assert "custom web apps" in reply
            assert recorder.cost.provider == "groq"
            assert recorder.cost.input_tokens == 150
            assert recorder.cost.output_tokens == 25
            assert recorder.latency.llm_end_time is not None

    asyncio.run(_run())


def test_voice_transcribe_telemetry():
    """Verify transcribe_audio_bytes records STT timestamps and audio duration."""
    async def _run():
        recorder = VoiceTelemetryRecorder()
        # 1 second of 16kHz 16-bit PCM silence = 32000 bytes
        dummy_pcm = b"\x00" * 32000

        with patch("app.services.voice_service._whisper_model") as mock_whisper:
            mock_whisper.transcribe.return_value = ([], MagicMock())
            await transcribe_audio_bytes(dummy_pcm, telemetry_recorder=recorder)

        assert recorder.cost.stt_audio_duration_sec == 1.0
        assert recorder.latency.stt_start_time is not None
        assert recorder.latency.stt_end_time is not None
        assert recorder.latency.stt_latency_ms is not None

    asyncio.run(_run())


def test_voice_tts_telemetry():
    """Verify generate_tts_chunks records TTS start, first audio, and end timestamps."""
    async def _run():
        recorder = VoiceTelemetryRecorder()

        async def fake_stream(self):
            yield {"type": "audio", "data": b"chunk1"}
            yield {"type": "audio", "data": b"chunk2"}

        with patch("edge_tts.Communicate.stream", side_effect=fake_stream, autospec=True):
            chunks = []
            async for chunk in generate_tts_chunks("Test", telemetry_recorder=recorder):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert recorder.latency.tts_start_time is not None
        assert recorder.latency.tts_first_audio_time is not None
        assert recorder.latency.tts_end_time is not None
        assert recorder.latency.tts_ttfb_ms is not None

    asyncio.run(_run())

