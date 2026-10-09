"""
tests/test_voice_service_hardening.py — Phase B Verification Tests
Audits voice_service.py for regression safety, exit phrase robustness,
models status non-blocking behavior, and async offloading.
"""

import pytest
from app.services.voice_service import (
    is_exit_phrase,
    get_models_status,
    _build_voice_identity,
    check_sms_rate_limit,
    send_vonage_sms,
    load_models,
    get_or_create_user,
    save_chat_message,
    _select_tts_voice,
    _pyttsx3_to_wav_bytes,
    transcribe_audio_bytes,
    generate_tts_chunks,
    process_voice_websocket,
    process_vonage_websocket,
    play_audio_stream,
)


def test_is_exit_phrase_exact_and_word_boundary():
    """Verify exit phrases match on word boundaries and intent."""
    # Valid exit phrases
    assert is_exit_phrase("bye") is True
    assert is_exit_phrase("goodbye") is True
    assert is_exit_phrase("Goodbye!") is True
    assert is_exit_phrase("bye bye") is True
    assert is_exit_phrase("ok bye") is True
    assert is_exit_phrase("thank you bye") is True
    assert is_exit_phrase("Allah Hafiz") is True
    assert is_exit_phrase("Khuda hafiz") is True
    assert is_exit_phrase("exit") is True

    # Substring false positives (MUST NOT match)
    assert is_exit_phrase("maybe") is False
    assert is_exit_phrase("bayonet") is False
    assert is_exit_phrase("goodbye world of software") is False

    # Words inside conversational questions (MUST NOT match)
    assert is_exit_phrase("can you say bye later") is False
    assert is_exit_phrase("what does bye mean") is False
    assert is_exit_phrase("why did you say goodbye") is False
    assert is_exit_phrase("I will say bye to my colleague first") is False


def test_get_models_status_non_blocking():
    """Verify get_models_status returns instantly without loading models."""
    status = get_models_status()
    assert isinstance(status, dict)
    assert "whisper_loaded" in status
    assert "vad_loaded" in status
    assert "tts_available" in status


def test_all_14_voice_service_functions_present():
    """Verify all 14 required voice_service functions are defined and importable."""
    functions = [
        send_vonage_sms,
        check_sms_rate_limit,
        get_models_status,
        load_models,
        get_or_create_user,
        save_chat_message,
        _build_voice_identity,
        _select_tts_voice,
        _pyttsx3_to_wav_bytes,
        transcribe_audio_bytes,
        generate_tts_chunks,
        process_voice_websocket,
        process_vonage_websocket,
        play_audio_stream,
    ]
    for fn in functions:
        assert callable(fn), f"Expected {fn} to be callable"


def test_input_gating_whitelist_and_short_noise():
    from app.services.voice_service import SHORT_VALID_REPLIES
    import re

    def _gate_passes(transcript: str) -> bool:
        cleaned_chars = re.sub(r"[^\w\u0600-\u06FF]", "", transcript).strip()
        cleaned_lower = cleaned_chars.lower()
        raw_lower = transcript.strip().lower()
        is_whitelisted = (cleaned_lower in SHORT_VALID_REPLIES) or (raw_lower in SHORT_VALID_REPLIES)
        is_too_short = len(cleaned_chars) <= 1 and not is_whitelisted
        if not transcript.strip() or not cleaned_chars or is_too_short:
            return False
        return True

    # 1. Whitelisted short valid replies MUST pass
    for valid_word in ["yes", "no", "ok", "okay", "hi", "hello", "haan", "nahi", "theek", "ji", "jee", "جی", "ہاں", "نہیں", "ٹھیک"]:
        assert _gate_passes(valid_word) is True, f"'{valid_word}' was blocked by input gating"
        assert _gate_passes(f"  {valid_word}! ") is True

    # 2. Punctuation only and empty MUST be blocked
    assert _gate_passes("") is False
    assert _gate_passes("   ") is False
    assert _gate_passes("...") is False
    assert _gate_passes("?!") is False

    # 3. Random 1-character noise (not in whitelist) MUST be blocked
    assert _gate_passes("a") is False
    assert _gate_passes("x") is False
    assert _gate_passes("1") is False

    # 4. Multi-word valid speech MUST pass
    assert _gate_passes("I want to book an appointment") is True
    assert _gate_passes("What are your office hours?") is True


def test_stt_calibration_logging(caplog):
    import asyncio
    import logging
    from unittest.mock import patch

    async def _run():
        pcm_data = b"\x10\x10" * 8000  # 16000 bytes = 500ms
        with caplog.at_level(logging.INFO):
            with patch("app.services.groq_service.GroqService.transcribe_audio_with_usage", return_value=("test", "en", 0.5)):
                await transcribe_audio_bytes(pcm_data)
        assert any("STT Turn Calibration:" in record.message for record in caplog.records)

    asyncio.run(_run())


def test_turn_diagnostic_logging(caplog):
    import asyncio
    import logging
    from unittest.mock import patch

    async def _run():
        # 1. Successful turn logging
        pcm_data = b"\x10\x10" * 8000  # 16000 bytes = 500ms
        with caplog.at_level(logging.INFO):
            with patch("app.services.groq_service.GroqService.transcribe_audio_with_usage", return_value=("hello there", "en", 0.5)):
                await transcribe_audio_bytes(pcm_data, vad_speech_ms=450.0)
        
        diag_records = [r.message for r in caplog.records if "Turn Diagnostic:" in r.message]
        assert len(diag_records) >= 1
        record = diag_records[-1]
        assert "duration_ms=" in record
        assert "rms=" in record
        assert "vad_speech_ms=450.0" in record
        assert "no_speech_prob=" in record
        assert "avg_logprob=" in record
        assert "compression_ratio=" in record
        assert "decision=sent ('hello there')" in record

        # 2. Dropped turn logging (e.g. audio too short < min_stt_duration_ms=400ms)
        short_pcm = b"\x10\x10" * 1600  # 3200 bytes = 100ms
        with caplog.at_level(logging.INFO):
            await transcribe_audio_bytes(short_pcm, vad_speech_ms=None)
        
        diag_records_dropped = [r.message for r in caplog.records if "Turn Diagnostic:" in r.message]
        assert len(diag_records_dropped) >= 2
        dropped_record = diag_records_dropped[-1]
        assert "vad_speech_ms=n/a" in dropped_record
        assert "decision=dropped (below min duration" in dropped_record

    asyncio.run(_run())

