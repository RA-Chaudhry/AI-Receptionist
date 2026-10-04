"""
tests/test_vad_barge_in_hardening.py — Phase F: VAD & Barge-In Hardening Tests

Covers:
1. Barge-in guard window (BARGE_IN_GUARD_MS) and minimum speech duration.
2. Noise floor adaptation (synthetic noise RMS 40, 120, 300 vs speech RMS 1500, clamping at 400).
3. Max turn duration guard (15s) with warning logging.
4. Duplicate race prevention between server VAD and client speech_ended.
"""

import asyncio
import logging
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.utils.audio_buffer import VADProcessor, TurnDetector, TurnState
from app.services.voice_service import process_voice_websocket
from app.core.config import settings


def generate_synthetic_pcm(duration_ms: int, rms_level: float, sample_rate: int = 16000) -> bytes:
    """Generates synthetic sine/noise PCM audio bytes with exact specified RMS level."""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    if rms_level <= 0:
        return b"\x00" * (num_samples * 2)

    # Sine wave with amplitude = rms * sqrt(2)
    amplitude = rms_level * (2 ** 0.5)
    t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * 440 * t)
    waveform = np.clip(waveform, -32767, 32767).astype(np.int16)
    return waveform.tobytes()


def test_noise_floor_clamping_and_speech_isolation():
    """
    Verify noise floor adapts only on quiet/noise frames (RMS 40, 120, 300)
    and strictly IGNORES high energy speech (RMS 1500) without learning it as noise.
    """
    vad = VADProcessor(aggressiveness=2, min_rms=100.0)
    vad.reset_noise_floor()
    assert vad.noise_floor == 40.0

    # 1. Feed low noise (RMS 40)
    for _ in range(5):
        vad.update_noise_floor(40.0)
    assert 20.0 <= vad.noise_floor <= 45.0

    # 2. Feed moderate noise (RMS 120)
    for _ in range(50):
        vad.update_noise_floor(120.0)
    assert vad.noise_floor > 40.0

    # 3. Feed higher background noise (RMS 300)
    for _ in range(50):
        vad.update_noise_floor(300.0)
    assert vad.noise_floor <= 400.0

    # 4. Feed loud speech (RMS 1500)
    # Even if fed directly to update_noise_floor, must NOT inflate noise floor past 400
    floor_before_speech = vad.noise_floor
    vad.update_noise_floor(1500.0)
    # Floor must NOT change on speech frame > 400 RMS
    assert vad.noise_floor == floor_before_speech
    assert vad.noise_floor <= 400.0

    # Verify speech detection works with loud frame
    pcm_speech_1500 = generate_synthetic_pcm(400, 1500.0)
    is_sp = vad.is_speech(pcm_speech_1500, update_noise=False)
    assert is_sp is True

    # 5. Reset per call
    vad.reset_noise_floor()
    assert vad.noise_floor == 40.0


def test_turn_detector_noise_floor_frozen_during_speech():
    """Verify TurnDetector freezes noise floor updates while in TurnState.SPEAKING."""
    vad = VADProcessor()
    td = TurnDetector(vad_processor=vad, min_speech_duration_ms=60)
    td.reset()

    # User speaks loudly (RMS 1200) for 100ms
    speech_frame = generate_synthetic_pcm(100, 1200.0)
    res = td.process_frame(speech_frame)
    assert res.state == TurnState.SPEAKING

    floor_during_speech = vad.noise_floor
    # Feed more loud speech while state is SPEAKING
    speech_frame_2 = generate_synthetic_pcm(100, 1400.0)
    td.process_frame(speech_frame_2)

    # Noise floor must not have been updated during SPEAKING
    assert vad.noise_floor == floor_during_speech


def test_max_turn_guard_15s_triggers_warning_and_truncates(caplog):
    """Verify 15-second max turn guard forces completion and logs WARNING."""
    td = TurnDetector(
        sample_rate=16000,
        frame_duration_ms=20,
        max_turn_duration_ms=500, # Shortened to 500ms for fast test execution
        silence_duration_ms=1000,
        min_speech_duration_ms=40,
    )
    td.reset()

    # Feed 100ms speech to enter SPEAKING
    speech_initial = generate_synthetic_pcm(100, 1000.0)
    res1 = td.process_frame(speech_initial)
    assert res1.state == TurnState.SPEAKING

    # Feed continuous speech exceeding max_turn_duration_ms (500ms total)
    speech_overflow = generate_synthetic_pcm(450, 1000.0)
    with caplog.at_level(logging.WARNING):
        res2 = td.process_frame(speech_overflow)

    assert res2.turn_completed is True
    assert res2.audio_bytes is not None
    # Verify WARNING log
    warning_logged = any("max turn duration guard" in rec.message.lower() for rec in caplog.records)
    assert warning_logged is True


def test_barge_in_guard_window_ignores_initial_echo():
    """Verify barge-in is ignored during the initial BARGE_IN_GUARD_MS window."""
    async def _run():
        sent_messages = []
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=lambda msg: sent_messages.append(msg))

        step = 0
        async def mock_receive():
            nonlocal step
            if step == 0:
                step += 1
                # User turn 1
                return {"text": '{"type": "speech_ended"}'}
            elif step == 1:
                step += 1
                # Echo arrived immediately (< 100ms) while assistant starts speaking
                await asyncio.sleep(0.02)
                # Send raw bytes frame
                return {"bytes": generate_synthetic_pcm(40, 1200.0)}
            elif step == 2:
                step += 1
                await asyncio.sleep(0.10)
                return {"type": "websocket.disconnect"}
            return {"type": "websocket.disconnect"}

        mock_ws.receive = mock_receive

        with patch("app.services.voice_service.transcribe_audio_bytes", new_callable=AsyncMock) as mock_stt, \
             patch("app.services.voice_service.get_or_create_user", new_callable=AsyncMock) as mock_user, \
             patch("app.services.voice_service.generate_tts_chunks") as mock_tts:

            user_obj = MagicMock()
            user_obj.id = 991
            user_obj.name = "BargeIn Test"
            user_obj.email = "bargein@example.com"
            mock_user.return_value = user_obj
            mock_stt.return_value = "Hello there"

            async def slow_tts(*args, **kwargs):
                await asyncio.sleep(0.20)
                yield b"FAKE_AUDIO"

            mock_tts.side_effect = slow_tts

            await process_voice_websocket(mock_ws, "+923001234567")

            # Verify voice_interrupted was NOT triggered by early echo frame (<300ms)
            interrupted_msgs = [m for m in sent_messages if m.get("type") == "voice_interrupted"]
            assert len(interrupted_msgs) == 0

    asyncio.run(_run())


def test_server_vad_and_speech_ended_race_condition():
    """
    Verify that when server VAD completes an utterance naturally,
    a subsequent duplicate 'speech_ended' event does NOT start a 2nd generation task.
    """
    async def _run():
        sent_messages = []
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=lambda msg: sent_messages.append(msg))

        generation_count = 0
        async def count_transcribe(*args, **kwargs):
            nonlocal generation_count
            generation_count += 1
            return "What are your services?"

        step = 0
        async def mock_receive():
            nonlocal step
            if step == 0:
                step += 1
                # 1. First send speech + silence to trigger server VAD natural completion
                speech = generate_synthetic_pcm(200, 1200.0)
                silence = b"\x00" * int(16000 * 0.400 * 2) # 400ms silence
                return {"bytes": speech + silence}
            elif step == 1:
                step += 1
                # 2. Immediately send redundant 'speech_ended' event
                return {"text": '{"type": "speech_ended"}'}
            elif step == 2:
                step += 1
                while not any(m.get("type") == "response_end" for m in sent_messages):
                    await asyncio.sleep(0.01)
                return {"type": "websocket.disconnect"}
            return {"type": "websocket.disconnect"}

        mock_ws.receive = mock_receive

        with patch("app.services.voice_service.transcribe_audio_bytes", side_effect=count_transcribe), \
             patch("app.services.voice_service.get_or_create_user", new_callable=AsyncMock) as mock_user:

            user_obj = MagicMock()
            user_obj.id = 990
            user_obj.name = "Race Test"
            user_obj.email = "race@example.com"
            mock_user.return_value = user_obj

            await process_voice_websocket(mock_ws, "+923001234567")

            # Generation must have executed EXACTLY once
            assert generation_count == 1

    asyncio.run(_run())
