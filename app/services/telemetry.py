"""
app/services/telemetry.py — Voice Latency Measurement & Latency Budget Instrumentation

Provides granular timestamp instrumentation using time.perf_counter(),
computes turn-around latency metrics (TTFT, TTFB, TTFA), defines target latency budgets,
and tracks token/audio usage costs per call_id / request_id / turn_id.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

TELEMETRY_LOG_FILE = os.getenv("TELEMETRY_LOG_FILE", "logs/telemetry.jsonl")


def get_telemetry_log_file() -> str:
    """Returns the effective telemetry log file path or 'disabled'."""
    global TELEMETRY_LOG_FILE
    if TELEMETRY_LOG_FILE:
        return TELEMETRY_LOG_FILE
    env_val = os.getenv("TELEMETRY_LOG_FILE")
    if env_val is not None:
        return env_val
    try:
        from app.core.config import settings
        return getattr(settings, "telemetry_log_file", "logs/telemetry.jsonl")
    except Exception:
        return "logs/telemetry.jsonl"


class LatencyBudget(BaseModel):
    """
    Target latency budget definitions (in milliseconds).
    IMPORTANT: These represent target budget allocations, not measured claims.
    """
    vad_end_detection_target_ms: float = 150.0
    stt_finalization_target_ms: float = 300.0
    llm_ttft_target_ms: float = 400.0
    tts_first_audio_target_ms: float = 250.0
    network_buffering_target_ms: float = 100.0
    total_target_ms: float = 1200.0

    model_config = ConfigDict(frozen=True)


DEFAULT_LATENCY_BUDGET = LatencyBudget()


class LatencyTrace(BaseModel):
    """
    Captures high-resolution timestamps via time.perf_counter() across the turn lifecycle
    and computes delta durations in milliseconds.
    """
    turn_created_time: Optional[float] = None
    mic_capture_start_time: Optional[float] = None
    first_audio_chunk_time: Optional[float] = None
    last_audio_chunk_time: Optional[float] = None
    frontend_speech_end_time: Optional[float] = None
    audio_received_time: Optional[float] = None
    vad_start_time: Optional[float] = None
    vad_end_time: Optional[float] = None
    speech_end_time: Optional[float] = None
    endpointing_delay_time_ms: Optional[float] = None
    stt_start_time: Optional[float] = None
    stt_end_time: Optional[float] = None
    llm_start_time: Optional[float] = None
    llm_first_token_time: Optional[float] = None
    first_clause_time: Optional[float] = None
    llm_end_time: Optional[float] = None
    tts_start_time: Optional[float] = None
    tts_first_audio_time: Optional[float] = None
    tts_end_time: Optional[float] = None
    send_start_time: Optional[float] = None
    audio_sent_time: Optional[float] = None
    frontend_playback_start_time: Optional[float] = None
    frontend_playback_end_time: Optional[float] = None
    turn_completed_time: Optional[float] = None

    @property
    def speech_duration_ms(self) -> Optional[float]:
        """Duration of speech utterance (from vad_start to speech_end)."""
        end = self.speech_end_time if self.speech_end_time is not None else self.vad_end_time
        if end is not None and self.vad_start_time is not None:
            return max(0.0, (end - self.vad_start_time) * 1000.0)
        return None

    @property
    def endpointing_delay_ms(self) -> Optional[float]:
        """Time from actual speech end until VAD silence detection triggers."""
        if self.endpointing_delay_time_ms is not None:
            return max(0.0, self.endpointing_delay_time_ms)
        if self.vad_end_time is not None and self.speech_end_time is not None:
            return max(0.0, (self.vad_end_time - self.speech_end_time) * 1000.0)
        return 0.0

    @property
    def vad_latency_ms(self) -> Optional[float]:
        """Backward-compatible alias for speech turn / vad duration."""
        if self.vad_end_time is not None and self.vad_start_time is not None:
            return max(0.0, (self.vad_end_time - self.vad_start_time) * 1000.0)
        return self.speech_duration_ms

    @property
    def stt_latency_ms(self) -> Optional[float]:
        """Duration of STT transcription."""
        if self.stt_end_time is not None and self.stt_start_time is not None:
            return max(0.0, (self.stt_end_time - self.stt_start_time) * 1000.0)
        return None

    @property
    def stt_duration(self) -> Optional[float]:
        """Alias for stt_latency_ms in seconds."""
        latency = self.stt_latency_ms
        return (latency / 1000.0) if latency is not None else None

    @property
    def llm_ttft_ms(self) -> Optional[float]:
        """Time to First Token (or full response generation in non-streaming mode)."""
        if self.llm_first_token_time is not None and self.llm_start_time is not None:
            return max(0.0, (self.llm_first_token_time - self.llm_start_time) * 1000.0)
        elif self.llm_end_time is not None and self.llm_start_time is not None:
            return max(0.0, (self.llm_end_time - self.llm_start_time) * 1000.0)
        return None

    @property
    def first_clause_ms(self) -> Optional[float]:
        """Time to first clause boundary emitted from stream."""
        if self.first_clause_time is not None and self.llm_start_time is not None:
            return max(0.0, (self.first_clause_time - self.llm_start_time) * 1000.0)
        return self.llm_ttft_ms

    @property
    def llm_total_latency_ms(self) -> Optional[float]:
        """Total LLM generation duration."""
        if self.llm_end_time is not None and self.llm_start_time is not None:
            return max(0.0, (self.llm_end_time - self.llm_start_time) * 1000.0)
        return None

    @property
    def tts_ttfb_ms(self) -> Optional[float]:
        """Time to first audio byte from TTS synthesis."""
        if self.tts_first_audio_time is not None and self.tts_start_time is not None:
            return max(0.0, (self.tts_first_audio_time - self.tts_start_time) * 1000.0)
        return None

    @property
    def tts_total_latency_ms(self) -> Optional[float]:
        """Total TTS synthesis duration."""
        if self.tts_end_time is not None and self.tts_start_time is not None:
            return max(0.0, (self.tts_end_time - self.tts_start_time) * 1000.0)
        return None

    @property
    def network_buffering_ms(self) -> Optional[float]:
        """Delay between first synthesized audio chunk and transmission to caller."""
        if self.audio_sent_time is not None and self.tts_first_audio_time is not None:
            return max(0.0, (self.audio_sent_time - self.tts_first_audio_time) * 1000.0)
        return None

    @property
    def send_ms(self) -> Optional[float]:
        """Alias for network send / buffering duration."""
        if self.audio_sent_time is not None and self.send_start_time is not None:
            return max(0.0, (self.audio_sent_time - self.send_start_time) * 1000.0)
        return self.network_buffering_ms

    @property
    def ttfa_ms(self) -> Optional[float]:
        """
        Time to First Audio (Total turn-around latency):
        From real end-of-speech (speech_end_time or vad_end_time - endpointing_delay) to audio sent to caller.
        """
        ref_start = (
            self.speech_end_time
            if self.speech_end_time is not None
            else self.vad_end_time
            if self.vad_end_time is not None
            else self.audio_received_time
        )
        if self.audio_sent_time is not None and ref_start is not None:
            return max(0.0, (self.audio_sent_time - ref_start) * 1000.0)
        return None

    @property
    def total_from_speech_end_ms(self) -> Optional[float]:
        """Alias for total latency from speech end."""
        return self.ttfa_ms

    @property
    def total_turn_duration_ms(self) -> Optional[float]:
        """Total end-to-end turn duration from turn_created to turn_completed."""
        start = self.turn_created_time or self.audio_received_time
        end = self.turn_completed_time or self.audio_sent_time
        if start is not None and end is not None:
            return max(0.0, (end - start) * 1000.0)
        return None

    def get_metrics_summary(self) -> Dict[str, Optional[float]]:
        return {
            "speech_duration_ms": self.speech_duration_ms,
            "endpointing_delay_ms": self.endpointing_delay_ms,
            "vad_latency_ms": self.vad_latency_ms,
            "stt_latency_ms": self.stt_latency_ms,
            "llm_ttft_ms": self.llm_ttft_ms,
            "first_clause_ms": self.first_clause_ms,
            "llm_total_latency_ms": self.llm_total_latency_ms,
            "tts_ttfb_ms": self.tts_ttfb_ms,
            "tts_total_latency_ms": self.tts_total_latency_ms,
            "network_buffering_ms": self.network_buffering_ms,
            "send_ms": self.send_ms,
            "ttfa_ms": self.ttfa_ms,
            "total_from_speech_end_ms": self.total_from_speech_end_ms,
            "total_turn_duration_ms": self.total_turn_duration_ms,
        }

    def evaluate_budget(self, budget: LatencyBudget = DEFAULT_LATENCY_BUDGET) -> Dict[str, Any]:
        """Compares measured metrics against target latency budgets."""
        ttfa = self.ttfa_ms
        return {
            "budget_targets_ms": budget.model_dump(),
            "within_total_budget": (ttfa <= budget.total_target_ms) if ttfa is not None else None,
            "measured_ttfa_ms": ttfa,
            "budget_delta_ms": (ttfa - budget.total_target_ms) if ttfa is not None else None,
        }


class CostTrace(BaseModel):
    """
    Usage and cost accounting for the turn.
    """
    call_id: str
    request_id: str
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    stt_provider: str = "faster_whisper"
    stt_model: str = "base"
    stt_audio_duration_sec: float = 0.0
    tts_provider: str = "edge_tts"
    tts_voice: str = "en-US-AriaNeural"
    tts_audio_duration_sec: float = 0.0
    estimated_cost_usd: float = 0.0

    def calculate_estimated_cost(self) -> float:
        """
        Estimates total turn cost based on provider pricing:
        - Groq LLaMA 3.1 8B: ~$0.05 / 1M prompt tokens, ~$0.08 / 1M completion tokens
        - Groq Whisper: ~$0.00018 / sec ($0.111 / hour)
        - Local router: $0.00
        - Faster-Whisper (local CPU): $0.00
        - Edge-TTS: $0.00
        """
        if self.provider == "groq":
            prompt_cost = (self.input_tokens / 1_000_000.0) * 0.05
            completion_cost = (self.output_tokens / 1_000_000.0) * 0.08
            stt_cost = (self.stt_audio_duration_sec * 0.00018) if self.stt_provider == "groq_whisper" else 0.0
            self.estimated_cost_usd = round(prompt_cost + completion_cost + stt_cost, 8)
        else:
            self.estimated_cost_usd = 0.0
        return self.estimated_cost_usd


class TurnTelemetry(BaseModel):
    """
    Complete telemetry event record for a single interaction turn.
    """
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    call_id: str
    request_id: str
    turn_id: str
    channel: str = "voice"  # "voice_ws", "vonage_voice", "chat_rest"
    
    # Audio metadata
    sample_rate: int = 16000
    channels: int = 1
    audio_format: str = "pcm_s16le"
    audio_chunk_count: int = 0
    total_audio_bytes: int = 0
    audio_duration_sec: float = 0.0
    
    # VAD metadata
    speech_frames_count: int = 0
    silence_frames_count: int = 0
    
    # STT metadata
    stt_provider: str = "faster_whisper"
    stt_model: str = "base"
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None
    transcript_length: int = 0
    
    # Interruption metadata
    interrupt_detected: bool = False
    interrupt_reason: Optional[str] = None
    
    latency: LatencyTrace = Field(default_factory=LatencyTrace)
    cost: CostTrace
    budget_evaluation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VoiceTelemetryRecorder:
    """
    Turn-level telemetry recorder that collects timestamps via time.perf_counter(),
    records token counts/costs, and writes structured telemetry JSONL records.
    """
    def __init__(
        self,
        call_id: Optional[str] = None,
        request_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        channel: str = "voice_ws",
    ) -> None:
        self.call_id = call_id or str(uuid4())
        self.request_id = request_id or str(uuid4())
        self.turn_id = turn_id or self.request_id
        self.channel = channel
        self.path: str = "llm"
        self.latency = LatencyTrace(turn_created_time=time.perf_counter())
        self.cost = CostTrace(call_id=self.call_id, request_id=self.request_id)
        
        # Audio / VAD metrics
        self.sample_rate: int = 16000
        self.channels: int = 1
        self.audio_format: str = "pcm_s16le"
        self.audio_chunk_count: int = 0
        self.total_audio_bytes: int = 0
        self.audio_duration_sec: float = 0.0
        self.speech_frames_count: int = 0
        self.silence_frames_count: int = 0
        
        # STT metrics
        self.stt_provider: str = "faster_whisper"
        self.stt_model: str = "base"
        self.detected_language: Optional[str] = None
        self.language_probability: Optional[float] = None
        self.transcript_length: int = 0
        
        # Interruption
        self.interrupt_detected: bool = False
        self.interrupt_reason: Optional[str] = None
        
        self.metadata: Dict[str, Any] = {}
        if os.getenv("TESTING", "").lower() in ("true", "1") or "pytest" in sys.modules:
            self.metadata["mock"] = True
        self._finalized = False

    # ── High-resolution timestamp setters (time.perf_counter) ─────────────────
    def mark_turn_created(self) -> None:
        if self.latency.turn_created_time is None:
            self.latency.turn_created_time = time.perf_counter()

    def mark_mic_capture_start(self) -> None:
        self.latency.mic_capture_start_time = time.perf_counter()

    def mark_first_audio_chunk(self) -> None:
        if self.latency.first_audio_chunk_time is None:
            self.latency.first_audio_chunk_time = time.perf_counter()

    def mark_last_audio_chunk(self) -> None:
        self.latency.last_audio_chunk_time = time.perf_counter()

    def mark_frontend_speech_end(self) -> None:
        self.latency.frontend_speech_end_time = time.perf_counter()

    def mark_audio_received(self) -> None:
        if self.latency.audio_received_time is None:
            self.latency.audio_received_time = time.perf_counter()

    def mark_vad_start(self) -> None:
        if self.latency.vad_start_time is None:
            self.latency.vad_start_time = time.perf_counter()

    def mark_vad_end(
        self,
        speech_end_time: Optional[float] = None,
        endpointing_delay_ms: Optional[float] = None,
    ) -> None:
        now = time.perf_counter()
        self.latency.vad_end_time = now
        if speech_end_time is not None:
            self.latency.speech_end_time = speech_end_time
        elif endpointing_delay_ms is not None:
            self.latency.speech_end_time = now - (endpointing_delay_ms / 1000.0)
        else:
            self.latency.speech_end_time = now

        if endpointing_delay_ms is not None:
            self.latency.endpointing_delay_time_ms = endpointing_delay_ms
        elif self.latency.speech_end_time is not None:
            self.latency.endpointing_delay_time_ms = max(0.0, (now - self.latency.speech_end_time) * 1000.0)

    def mark_stt_start(self) -> None:
        if self.latency.stt_start_time is None:
            self.latency.stt_start_time = time.perf_counter()

    def mark_stt_end(self) -> None:
        self.latency.stt_end_time = time.perf_counter()

    def mark_llm_start(self) -> None:
        if self.latency.llm_start_time is None:
            self.latency.llm_start_time = time.perf_counter()

    def mark_llm_first_token(self) -> None:
        if self.latency.llm_first_token_time is None:
            self.latency.llm_first_token_time = time.perf_counter()

    def mark_first_clause(self) -> None:
        if self.latency.first_clause_time is None:
            self.latency.first_clause_time = time.perf_counter()

    def mark_llm_end(self) -> None:
        self.latency.llm_end_time = time.perf_counter()
        if self.latency.llm_first_token_time is None:
            self.latency.llm_first_token_time = self.latency.llm_end_time
        if self.latency.first_clause_time is None:
            self.latency.first_clause_time = self.latency.llm_first_token_time

    def mark_tts_start(self) -> None:
        if self.latency.tts_start_time is None:
            self.latency.tts_start_time = time.perf_counter()

    def mark_tts_first_audio(self) -> None:
        if self.latency.tts_first_audio_time is None:
            self.latency.tts_first_audio_time = time.perf_counter()

    def mark_tts_end(self) -> None:
        self.latency.tts_end_time = time.perf_counter()

    def mark_send_start(self) -> None:
        if self.latency.send_start_time is None:
            self.latency.send_start_time = time.perf_counter()

    def mark_audio_sent(self) -> None:
        if self.latency.audio_sent_time is None:
            self.latency.audio_sent_time = time.perf_counter()

    def mark_frontend_playback_start(self) -> None:
        self.latency.frontend_playback_start_time = time.perf_counter()

    def mark_frontend_playback_end(self) -> None:
        self.latency.frontend_playback_end_time = time.perf_counter()

    def mark_turn_completed(self) -> None:
        self.latency.turn_completed_time = time.perf_counter()

    def log_latency_summary(self) -> None:
        m = self.latency.get_metrics_summary()
        path_val = self.path or (
            "router"
            if self.cost.provider in ("local_router", "regex_rules") or "greeting" in self.turn_id
            else ("tool" if self.cost.provider == "tool_service" else "llm")
        )
        stt_provider = self.stt_provider or "faster_whisper"
        llm_model = self.cost.model or "llama-3.1-8b-instant"
        endpointing_ms = m.get("endpointing_delay_ms") or 0.0
        stt_ms = m.get("stt_latency_ms") or 0.0
        llm_first_token_ms = m.get("llm_ttft_ms") or 0.0
        first_clause_ms = m.get("first_clause_ms") or 0.0
        tts_first_byte_ms = m.get("tts_ttfb_ms") or 0.0
        send_ms = m.get("send_ms") or 0.0
        total_from_speech_end_ms = m.get("total_from_speech_end_ms") or 0.0
        audio_len_s = self.audio_duration_sec or 0.0
        transcript_len = self.transcript_length or 0

        logger.info(
            "LATENCY path=%s, stt_provider=%s, llm_model=%s, endpointing_ms=%.1f, stt_ms=%.1f, llm_first_token_ms=%.1f, first_clause_ms=%.1f, tts_first_byte_ms=%.1f, send_ms=%.1f, total_from_speech_end_ms=%.1f, audio_len_s=%.2f, transcript_len=%d",
            path_val,
            stt_provider,
            llm_model,
            endpointing_ms,
            stt_ms,
            llm_first_token_ms,
            first_clause_ms,
            tts_first_byte_ms,
            send_ms,
            total_from_speech_end_ms,
            audio_len_s,
            transcript_len,
        )

    def record_interrupt(self, reason: str = "user_speech") -> None:
        self.interrupt_detected = True
        self.interrupt_reason = reason
        self.add_metadata("interrupt_reason", reason)

    # ── Audio & VAD metrics recording ─────────────────────────────────────────
    def record_audio_chunk(self, chunk_bytes: int) -> None:
        self.audio_chunk_count += 1
        self.total_audio_bytes += chunk_bytes
        self.mark_last_audio_chunk()

    def record_vad_decision(self, is_speech: bool) -> None:
        if is_speech:
            self.speech_frames_count += 1
        else:
            self.silence_frames_count += 1

    # ── STT metrics recording ─────────────────────────────────────────────────
    def record_stt_result(
        self,
        transcript: str,
        language: Optional[str] = None,
        language_probability: Optional[float] = None,
        provider: str = "faster_whisper",
        model: str = "base",
    ) -> None:
        self.detected_language = language
        self.language_probability = language_probability
        self.transcript_length = len(transcript)
        self.stt_provider = provider
        self.stt_model = model
        self.cost.stt_provider = provider
        self.cost.stt_model = model

    # ── Cost & usage recording ────────────────────────────────────────────────
    def record_llm_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.cost.provider = provider
        self.cost.model = model
        self.cost.input_tokens = input_tokens
        self.cost.output_tokens = output_tokens
        self.cost.total_tokens = input_tokens + output_tokens
        self.cost.calculate_estimated_cost()

    def record_stt_duration(self, duration_sec: float) -> None:
        self.audio_duration_sec = duration_sec
        self.cost.stt_audio_duration_sec = duration_sec

    def record_tts_duration(self, duration_sec: float) -> None:
        self.cost.tts_audio_duration_sec = duration_sec

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    # ── Finalization & Structured Logging ─────────────────────────────────────
    def finalize(self, log_to_file: bool = True) -> TurnTelemetry:
        self._finalized = True
        if self.latency.turn_completed_time is None:
            self.latency.turn_completed_time = time.perf_counter()
        self.cost.calculate_estimated_cost()
        budget_eval = self.latency.evaluate_budget(DEFAULT_LATENCY_BUDGET)

        # Sanity guards: Check for implausible latencies on STT+LLM turns
        path_val = self.path or (
            "router"
            if self.cost.provider in ("local_router", "regex_rules") or "greeting" in self.turn_id
            else ("tool" if self.cost.provider == "tool_service" else "llm")
        )
        if path_val == "llm" and (self.latency.stt_start_time is not None or self.latency.llm_start_time is not None):
            if not self.metadata.get("mock"):
                ep_ms = self.latency.endpointing_delay_ms or 0.0
                tot_ms = self.latency.total_from_speech_end_ms or 0.0
                if ep_ms < 100.0 or tot_ms < 300.0:
                    logger.warning(
                        "implausible latency: endpointing_delay_ms=%.1f (<100ms) or total_from_speech_end_ms=%.1f (<300ms) for STT+LLM turn. "
                        "Raw timestamps: vad_start=%.4f, vad_end=%.4f, speech_end=%.4f, stt_start=%.4f, stt_end=%.4f, "
                        "llm_start=%.4f, llm_first_token=%.4f, first_clause=%.4f, tts_start=%.4f, tts_first_audio=%.4f, audio_sent=%.4f",
                        ep_ms,
                        tot_ms,
                        self.latency.vad_start_time or 0.0,
                        self.latency.vad_end_time or 0.0,
                        self.latency.speech_end_time or 0.0,
                        self.latency.stt_start_time or 0.0,
                        self.latency.stt_end_time or 0.0,
                        self.latency.llm_start_time or 0.0,
                        self.latency.llm_first_token_time or 0.0,
                        self.latency.first_clause_time or 0.0,
                        self.latency.tts_start_time or 0.0,
                        self.latency.tts_first_audio_time or 0.0,
                        self.latency.audio_sent_time or 0.0,
                    )

        telemetry_record = TurnTelemetry(
            call_id=self.call_id,
            request_id=self.request_id,
            turn_id=self.turn_id,
            channel=self.channel,
            sample_rate=self.sample_rate,
            channels=self.channels,
            audio_format=self.audio_format,
            audio_chunk_count=self.audio_chunk_count,
            total_audio_bytes=self.total_audio_bytes,
            audio_duration_sec=self.audio_duration_sec,
            speech_frames_count=self.speech_frames_count,
            silence_frames_count=self.silence_frames_count,
            stt_provider=self.stt_provider,
            stt_model=self.stt_model,
            detected_language=self.detected_language,
            language_probability=self.language_probability,
            transcript_length=self.transcript_length,
            interrupt_detected=self.interrupt_detected,
            interrupt_reason=self.interrupt_reason,
            latency=self.latency,
            cost=self.cost,
            budget_evaluation=budget_eval,
            metadata=self.metadata,
        )

        if log_to_file:
            self._write_to_log(telemetry_record)

        return telemetry_record

    def _write_to_log(self, record: TurnTelemetry) -> None:
        log_file = get_telemetry_log_file()
        if not log_file or log_file.lower() == "disabled":
            return
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            entry = record.model_dump()
            # Include computed latency helpers in log output for instant observability
            entry["metrics_summary"] = self.latency.get_metrics_summary()

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write telemetry log: {exc}")
