import asyncio
import io
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.groq_service import (
    GroqService,
    KNOWN_HALLUCINATIONS,
    normalize_whisper_language,
)
from app.services.voice_service import (
    _calculate_pcm_rms,
    _call_language_cache,
    _web_tts_cache,
    _vonage_pcm_cache,
    TTS_DISK_CACHE_DIR,
    load_tts_disk_cache,
    save_tts_to_disk_cache,
    transcribe_audio_bytes,
)


def test_normalize_whisper_language():
    # 1. English variations -> "en"
    assert normalize_whisper_language("English") == "en"
    assert normalize_whisper_language("english") == "en"
    assert normalize_whisper_language("en") == "en"
    assert normalize_whisper_language(" EN ") == "en"

    # 2. Urdu & Hindi variations -> "ur"
    assert normalize_whisper_language("Urdu") == "ur"
    assert normalize_whisper_language("urdu") == "ur"
    assert normalize_whisper_language("ur") == "ur"
    assert normalize_whisper_language("Hindi") == "ur"
    assert normalize_whisper_language("hindi") == "ur"
    assert normalize_whisper_language("hi") == "ur"

    # 3. Unmapped / Unknown / None -> None (omit param)
    assert normalize_whisper_language("unknown") is None
    assert normalize_whisper_language("Turkish") is None
    assert normalize_whisper_language("Spanish") is None
    assert normalize_whisper_language("") is None
    assert normalize_whisper_language(None) is None


def test_groq_whisper_400_retry_without_language():
    async def _run():
        svc = GroqService()
        
        mock_400_resp = MagicMock()
        mock_400_resp.status_code = 400
        mock_400_resp.json.return_value = {"error": {"message": "unsupported language"}}

        mock_200_resp = MagicMock()
        mock_200_resp.status_code = 200
        mock_200_resp.json.return_value = {
            "text": "Hello world",
            "language": "English",
            "duration": 1.2,
            "segments": [{"no_speech_prob": 0.0, "avg_logprob": -0.2}],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_400_resp, mock_200_resp])

        with patch("app.services.groq_service.get_shared_client", return_value=mock_client):
            text, lang, dur = await svc.transcribe_audio_with_usage(
                b"fake_wav", language="English"
            )
            assert text == "Hello world"
            assert lang == "English"
            assert mock_client.post.call_count == 2

    asyncio.run(_run())


def test_whisper_hallucination_filtering():
    async def _run():
        svc = GroqService()

        # 1. High no_speech_prob
        mock_hallucination_resp = MagicMock()
        mock_hallucination_resp.status_code = 200
        mock_hallucination_resp.json.return_value = {
            "text": "Thank you for watching!",
            "language": "English",
            "duration": 1.0,
            "segments": [{"no_speech_prob": 0.85, "avg_logprob": -1.4}],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_hallucination_resp)

        with patch("app.services.groq_service.get_shared_client", return_value=mock_client):
            text, _, _ = await svc.transcribe_audio_with_usage(b"fake_wav")
            assert text == ""

        # 2. Known hallucination phrase
        mock_feyyaz_resp = MagicMock()
        mock_feyyaz_resp.status_code = 200
        mock_feyyaz_resp.json.return_value = {
            "text": "www.feyyaz.tv",
            "language": "English",
            "duration": 1.0,
            "segments": [{"no_speech_prob": 0.05, "avg_logprob": -0.1}],
        }
        mock_client.post = AsyncMock(return_value=mock_feyyaz_resp)

        with patch("app.services.groq_service.get_shared_client", return_value=mock_client):
            text, _, _ = await svc.transcribe_audio_with_usage(b"fake_wav")
            assert text == ""

    asyncio.run(_run())


def test_calculate_pcm_rms():
    # Pure zero silence
    silence = b"\x00\x00" * 800
    assert _calculate_pcm_rms(silence) == 0.0

    # Non-zero audio
    import struct
    samples = [1000, -1000, 500, -500] * 200
    audio_pcm = struct.pack(f"<{len(samples)}h", *samples)
    rms = _calculate_pcm_rms(audio_pcm)
    assert rms > 0.0


def test_transcribe_audio_rms_skip():
    async def _run():
        # Test audio below RMS threshold is dropped without calling Groq API
        silence_pcm = b"\x00\x00" * 3200
        with patch("app.core.config.settings.min_stt_rms", 50.0), \
             patch("app.core.config.settings.groq_api_key", "fake-key"), \
             patch("app.services.groq_service.GroqService.transcribe_audio_with_usage") as mock_groq:
            transcript = await transcribe_audio_bytes(silence_pcm)
            assert transcript == ""
            mock_groq.assert_not_called()

    asyncio.run(_run())


def test_tts_disk_cache_persistence(tmp_path):
    with patch("app.services.voice_service.TTS_DISK_CACHE_DIR", tmp_path):
        voice = "en-US-JennyNeural"
        text = "Test persistent audio phrase."
        mp3_bytes = b"FAKE_MP3_DATA"
        pcm_bytes = b"FAKE_PCM_DATA"

        save_tts_to_disk_cache(voice, text, mp3_bytes, pcm_bytes)

        # Clear in-memory caches
        _web_tts_cache.clear()
        _vonage_pcm_cache.clear()

        # Load from disk
        load_tts_disk_cache()

        cache_key = f"{voice}:{text}"
        assert cache_key in _web_tts_cache
        assert _vonage_pcm_cache[cache_key] == pcm_bytes


def test_language_cache_never_stores_unmapped():
    async def _run():
        call_id = "test-call-123"
        _call_language_cache.clear()

        # When Groq returns "English", cache maps to "en"
        pcm_audio = b"\x10\x10" * 8000  # 500ms audio
        with patch("app.services.groq_service.GroqService.transcribe_audio_with_usage", return_value=("Hello", "English", 1.0)):
            await transcribe_audio_bytes(pcm_audio, call_id=call_id)
            assert _call_language_cache.get(call_id) == "en"

        # When Groq returns "unknown", cache does not store it
        _call_language_cache.clear()
        with patch("app.services.groq_service.GroqService.transcribe_audio_with_usage", return_value=("Hello", "unknown", 1.0)):
            await transcribe_audio_bytes(pcm_audio, call_id=call_id)
            assert call_id not in _call_language_cache

    asyncio.run(_run())


def test_closing_and_gratitude_router_rules():
    from app.services.receptionist_router import ReceptionistRouter

    # 1. Plain "thank you" -> gratitude acknowledgement, does NOT end the call
    reply_thanks = ReceptionistRouter.get_hardcoded_response("thank you")
    assert reply_thanks == "You're welcome! Is there anything else I can help you with?"

    # 2. "thank you, what are your hours?" -> answers hours, does NOT trigger closing
    reply_hours = ReceptionistRouter.get_hardcoded_response("thank you, what are your hours?")
    assert "9 AM to 5 PM" in (reply_hours or "")

    # 3. "what time do you end" -> does NOT trigger closing
    reply_end = ReceptionistRouter.get_hardcoded_response("what time do you end")
    assert reply_end != "Goodbye, and thank you for contacting Logic Pulse!"

    # 4. Explicit "bye" -> closing goodbye
    reply_bye = ReceptionistRouter.get_hardcoded_response("bye")
    assert reply_bye == "Goodbye, and thank you for contacting Logic Pulse!"

    # 5. "thank you bye" -> explicit closing
    reply_thank_bye = ReceptionistRouter.get_hardcoded_response("thank you bye")
    assert reply_thank_bye == "Goodbye, and thank you for contacting Logic Pulse!"


def test_real_question_with_logic_pulse_not_dropped():
    async def _run():
        svc = GroqService()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "text": "Tell me about the services provided by Logic Pulse",
            "language": "English",
            "duration": 3.2,
            "segments": [{"no_speech_prob": 0.01, "avg_logprob": -0.15}],
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("app.services.groq_service.get_shared_client", return_value=mock_client):
            text, lang, dur = await svc.transcribe_audio_with_usage(b"fake_wav_3s")
            assert text == "Tell me about the services provided by Logic Pulse"
            assert dur == 3.2

    asyncio.run(_run())


def test_empty_buffer_on_speech_ended_skips_stt(client):
    """Test that if no audio was buffered on speech_ended, STT is skipped and no transcript is returned."""
    with client.websocket_connect("/api/v1/voice/stream") as ws:
        # Client sends speech_ended without sending audio bytes first
        import json
        ws.send_text(json.dumps({"type": "speech_ended"}))
        
        # Send a subsequent real audio chunk to confirm connection remains alive and healthy
        ws.send_bytes(b"\x10\x10" * 8000)
        msg = ws.receive_json()
        assert msg["type"] == "listening"


def test_real_sentences_with_hallucination_words_not_dropped():
    async def _run():
        svc = GroqService()
        for phrase in ["I want to subscribe to your service", "what time is the end of the day"]:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "text": phrase,
                "language": "English",
                "duration": 2.5,
                "segments": [{"no_speech_prob": 0.05, "avg_logprob": -0.2}],
            }
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch("app.services.groq_service.get_shared_client", return_value=mock_client):
                text, lang, dur = await svc.transcribe_audio_with_usage(b"fake_wav")
                assert text == phrase

    asyncio.run(_run())


def test_generic_hallucinations_dropped_on_suspicious_audio():
    async def _run():
        svc = GroqService()
        for phrase in ["the end", "watching", "subscribe to", "thank you for watching"]:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "text": phrase,
                "language": "English",
                "duration": 0.5,
                "segments": [{"no_speech_prob": 0.65, "avg_logprob": -1.2}],
            }
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            with patch("app.services.groq_service.get_shared_client", return_value=mock_client):
                text, lang, dur = await svc.transcribe_audio_with_usage(b"fake_wav")
                assert text == ""

    asyncio.run(_run())


def test_prompt_echo_guard():
    from app.services.groq_service import is_prompt_echo, WHISPER_PROMPT_HINT
    hint = (
        "Logic Pulse AI Receptionist: Software development, AI automation, chatbots, "
        "web apps, dashboards, appointments, office timings, Muhammad Ramzan. English and Urdu."
    )

    # 1. Short real utterances using words from prompt MUST NOT be dropped
    short_utterances = [
        "web applications",
        "appointment booking",
        "do you build mobile apps",
        "what are your office timings",
        "software development",
        "chatbots",
    ]
    for utterance in short_utterances:
        assert not is_prompt_echo(utterance, hint), f"Short utterance '{utterance}' was falsely flagged as echo"

    # 2. Long echo of the prompt hint MUST be dropped
    long_echo = "Logic Pulse AI Receptionist Software development AI automation chatbots web apps dashboards appointments"
    assert is_prompt_echo(long_echo, hint) is True

    # 3. Exact hint text MUST be dropped
    assert is_prompt_echo(hint, hint) is True


def test_former_prompts_and_prefixes_dropped_without_runtime_prompt():
    """Verify former prompt strings and prefixes are dropped even when prompt=None."""
    from app.services.groq_service import is_prompt_echo

    # Exact former prompt strings
    assert is_prompt_echo("Logic Pulse AI Receptionist.", prompt=None) is True
    assert is_prompt_echo("English and Urdu.", prompt=None) is True
    assert is_prompt_echo("english and urdu", prompt=None) is True

    # Prefixes of former prompt strings (3+ words)
    assert is_prompt_echo("Logic Pulse AI Receptionist", prompt=None) is True
    assert is_prompt_echo("Logic Pulse AI", prompt=None) is True
    assert is_prompt_echo("Logic Pulse AI Receptionist Software", prompt=None) is True

    # Real user speech and bare company name must NOT be falsely dropped
    assert is_prompt_echo("Logic Pulse", prompt=None) is False
    assert is_prompt_echo("tell me about Logic Pulse", prompt=None) is False
    assert is_prompt_echo("Do you speak English?", prompt=None) is False
    assert is_prompt_echo("Kya aap Urdu bolte hain?", prompt=None) is False
    assert is_prompt_echo("I want an appointment", prompt=None) is False


def test_stt_calls_do_not_send_prompt_to_groq(monkeypatch):
    """Verify transcribe_audio_bytes never sends WHISPER_PROMPT_HINT or prompt text to Groq."""
    import asyncio
    from unittest.mock import AsyncMock, patch
    from app.core.config import settings
    from app.services.voice_service import transcribe_audio_bytes

    monkeypatch.setattr(settings, "stt_provider", "groq_whisper")
    monkeypatch.setattr(settings, "groq_api_key", "mock_key")

    captured_prompt = "NOT_NONE"

    async def mock_transcribe(wav_bytes, model=None, language=None, prompt=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        return ("hello", "en", 1.0)

    import struct
    sample_rate = 16000
    duration_samples = 8000  # 500ms
    audio_bytes = struct.pack(f"<{duration_samples}h", *([1000] * duration_samples))

    with patch("app.services.groq_service.GroqService.transcribe_audio_with_usage", new=mock_transcribe):
        result = asyncio.run(transcribe_audio_bytes(audio_bytes))
        assert result == "hello"
        assert captured_prompt is None


