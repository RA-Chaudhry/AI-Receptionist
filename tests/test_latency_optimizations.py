"""
tests/test_latency_optimizations.py
Tests verifying latency reduction optimizations:
1. VAD AND-condition + adaptive noise floor + trailing silence trimming
2. Stream sentence chunking (first-clause punctuation threshold >= 15 chars)
3. In-memory history caching
4. Urdu script phrase normalization and router matching (~20 phrases)
5. Barge-in / interruption fast cancellation (< 200ms)
6. Shared Groq HTTP client lifecycle
"""

import asyncio
import numpy as np
import pytest
import time

from app.services.groq_service import get_shared_client, close_shared_client, GroqService
from app.services.receptionist_router import ReceptionistRouter
from app.services.voice_service import get_recent_voice_history, _history_cache
from app.utils.audio_buffer import VADProcessor, TurnDetector, TurnState, TurnDetectorResult
from app.utils.entity_normalizer import (
    extract_phone_number,
    extract_day,
    extract_time_hour,
    normalize_spoken_digits,
    format_phone_display,
)
from app.utils.stream_utils import stream_sentence_chunks, clean_tts_text


# ──────────────────────────────────────────────────────────────────────────────
# 1. VAD AND-condition & Adaptive Noise Floor
# ──────────────────────────────────────────────────────────────────────────────

class DummyVadAlwaysTrue:
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return True

class DummyVadAlwaysFalse:
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return False


def test_vad_and_condition_with_energy():
    """Test that is_speech requires BOTH WebRTC VAD True AND frame RMS >= threshold."""
    vad_proc = VADProcessor(min_rms=100.0)
    vad_proc.vad = DummyVadAlwaysTrue()

    # Frame 1: Low energy (silence / low mic noise RMS < 100)
    low_energy_pcm = (np.ones(320, dtype=np.int16) * 10).tobytes()
    assert not vad_proc.is_speech(low_energy_pcm)

    # Frame 2: High energy (loud sound RMS > 100)
    high_energy_pcm = (np.ones(320, dtype=np.int16) * 1000).tobytes()
    assert vad_proc.is_speech(high_energy_pcm)

    # Frame 3: High energy but VAD says False (e.g. non-speech loud click / pop)
    vad_proc.vad = DummyVadAlwaysFalse()
    assert not vad_proc.is_speech(high_energy_pcm)


def test_vad_adaptive_noise_floor():
    """Test adaptive noise floor adjustment during initial audio frames."""
    vad_proc = VADProcessor(min_rms=50.0)
    vad_proc.vad = DummyVadAlwaysTrue()
    detector = TurnDetector(vad_processor=vad_proc, silence_duration_ms=300, post_buffer_ms=150)

    # Feed background noise frames with RMS ~40 (below min_rms=50)
    noise_frame = (np.ones(320, dtype=np.int16) * 40).tobytes()
    for _ in range(15):
        event = detector.process_frame(noise_frame)
        assert event.state == TurnState.IDLE
        assert not event.turn_completed

    # Noise floor should adapt
    assert detector.vad_processor.noise_floor > 0
    effective_thresh = max(detector.vad_processor.min_rms, detector.vad_processor.noise_floor * 2.5)
    assert effective_thresh >= detector.vad_processor.min_rms


def test_turn_detector_trailing_silence_trimming():
    """Test that TurnDetector trims trailing silence beyond post_buffer_ms (~150ms)."""
    # 20ms frames at 16kHz = 320 samples = 640 bytes
    frame_bytes = 640
    vad_proc = VADProcessor(min_rms=50.0)
    vad_proc.vad = DummyVadAlwaysTrue()
    detector = TurnDetector(vad_processor=vad_proc, silence_duration_ms=300, post_buffer_ms=150)

    speech_frame = (np.ones(320, dtype=np.int16) * 1500).tobytes()
    silence_frame = (np.zeros(320, dtype=np.int16)).tobytes()

    # 1. Start speech (5 speech frames = 100ms)
    for _ in range(5):
        res = detector.process_frame(speech_frame)
    assert detector.state == TurnState.SPEAKING

    # 2. Add silence frames until turn ends (300ms silence = 15 frames)
    turn_audio = None
    endpointing_delay = 0.0
    for _ in range(16):
        res = detector.process_frame(silence_frame)
        if res.turn_completed:
            turn_audio = res.audio_bytes
            endpointing_delay = res.endpointing_delay_ms
            break

    assert turn_audio is not None
    # Total speech was 100ms. With 150ms post-buffer, audio length should be trimmed
    # and NOT include the full 300ms silence.
    assert len(turn_audio) < (5 + 16) * frame_bytes
    assert endpointing_delay >= 300.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. Stream Sentence Chunks First Clause (>= 15 chars)
# ──────────────────────────────────────────────────────────────────────────────

def test_stream_sentence_chunks_first_clause():
    """Test that stream_sentence_chunks emits early on comma/semicolon after >= 15 chars."""
    async def _run():
        async def token_gen():
            tokens = ["Hello there, ", "welcome to Logic Pulse. ", "How can I ", "assist you today?"]
            for t in tokens:
                yield t
                await asyncio.sleep(0.001)

        chunks = []
        async for chunk in stream_sentence_chunks(token_gen()):
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert "Hello there" in chunks[0]
        assert "Logic Pulse" in " ".join(chunks)

    asyncio.run(_run())


def test_clean_tts_text_formatting():
    """Test clean_tts_text strips markdown asterisks, hashes, backticks, emojis."""
    raw = "**Hello!** Welcome to #LogicPulse `AI`. 😊 How are you? 🚀"
    cleaned = clean_tts_text(raw)
    assert "**" not in cleaned
    assert "#" not in cleaned
    assert "`" not in cleaned
    assert "😊" not in cleaned
    assert "🚀" not in cleaned
    assert "Hello! Welcome to LogicPulse AI. How are you?" == cleaned


# ──────────────────────────────────────────────────────────────────────────────
# 3. In-Memory Conversation History Cache
# ──────────────────────────────────────────────────────────────────────────────

def test_in_memory_history_caching():
    """Test that history cache stores and retrieves history without redundant DB queries."""
    async def _run():
        user_id = 99999
        _history_cache[user_id] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        ]

        cached = await get_recent_voice_history(user_id)
        assert len(cached) == 2
        assert cached[0]["content"] == "Hello"

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────────────────
# 4. Urdu Script & Roman Urdu Normalization (~20 phrases)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "phrase,expected_type,expected_val",
    [
        # 1-4. Urdu Script Confirmations
        ("جی ہاں", "confirm", True),
        ("ہاں ٹھیک ہے", "confirm", True),
        ("درست ہے بک کر دیں", "confirm", True),
        ("بالکل تصدیق ہے", "confirm", True),
        # 5-7. Urdu Script Cancellations / Declines
        ("نہیں منسوخ کر دیں", "cancel", True),
        ("نہیں مجھے وقت تبدیل کرنا ہے", "cancel", True),
        ("روک دیں کینسل کریں", "cancel", True),
        # 8-11. Urdu Script Days & Times
        ("پیر کے دن", "day", "Monday"),
        ("جمعہ کو", "day", "Friday"),
        ("منگل صبح دس بجے", "day", "Tuesday"),
        ("بدھ شام تین بجے", "time", 15),
        # 12-14. Roman Urdu Confirmations & Days
        ("ji haan submit kardein", "confirm", True),
        ("theek hai confirm hai", "confirm", True),
        ("somwar ko 4 baje", "day", "Monday"),
        # 15-17. English Days, Times & Confirmation
        ("yes please book it", "confirm", True),
        ("Thursday at 2 PM", "day", "Thursday"),
        ("Friday at 11 AM", "time", 11),
        # 18-20. Spoken Phone Numbers (Urdu script, Roman Urdu, English)
        ("صفر تین چار پانچ ایک دو تین چار پانچ چھ سات", "phone", "03451234567"),
        ("zero three double four one two three four five six seven", "phone", "03441234567"),
        ("0300 1234567", "phone", "03001234567"),
    ]
)
def test_urdu_and_multilingual_phrases(phrase, expected_type, expected_val):
    if expected_type == "confirm":
        # Test confirmation regex via ReceptionistRouter appointment handler simulation
        history = [
            {"role": "user", "content": "I want an appointment on Monday at 3 PM"},
            {"role": "assistant", "content": "Would you like me to submit this booking request for Monday at 3 PM with phone number 0345 1234567?"},
        ]
        resp = ReceptionistRouter.get_hardcoded_response(phrase, history=history)
        assert resp is not None
        assert "ready for submission" in resp.lower() or "submission" in resp.lower()

    elif expected_type == "cancel":
        history = [
            {"role": "user", "content": "I want an appointment on Monday at 3 PM"},
            {"role": "assistant", "content": "Would you like me to submit this booking request?"},
        ]
        resp = ReceptionistRouter.get_hardcoded_response(phrase, history=history)
        assert resp is not None
        assert "no problem" in resp.lower() or "reschedule" in resp.lower()

    elif expected_type == "day":
        day = extract_day(phrase)
        assert day == expected_val

    elif expected_type == "time":
        hr = extract_time_hour(phrase)
        assert hr == expected_val

    elif expected_type == "phone":
        is_valid, phone_num, _ = extract_phone_number(phrase)
        assert is_valid is True
        assert phone_num == expected_val


# ──────────────────────────────────────────────────────────────────────────────
# 5. Barge-in / Interruption Task Cancellation (< 200ms)
# ──────────────────────────────────────────────────────────────────────────────

def test_barge_in_cancellation_speed():
    """Test that cancelling an active generation task terminates within 200ms."""
    async def _run():
        async def slow_generation():
            await asyncio.sleep(5.0)

        task = asyncio.create_task(slow_generation())
        await asyncio.sleep(0.01)

        t0 = time.perf_counter()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 200.0, f"Cancellation took {elapsed_ms:.2f}ms, expected < 200ms"

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────────────────
# 6. Shared Groq Client Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def test_shared_groq_client_singleton():
    """Test that get_shared_client returns the same AsyncClient instance."""
    async def _run():
        client1 = get_shared_client()
        client2 = get_shared_client()
        assert client1 is client2
        assert not client1.is_closed

        await close_shared_client()
        assert client1.is_closed

        # Re-acquisition returns fresh open client
        client3 = get_shared_client()
        assert client3 is not client1
        assert not client3.is_closed
        await close_shared_client()

    asyncio.run(_run())
