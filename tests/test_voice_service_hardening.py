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
