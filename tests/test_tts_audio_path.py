"""
tests/test_tts_audio_path.py — Phase E: TTS and Audio Path Hardening Tests

Covers:
1. Edge-tts pipelined sentence-by-sentence streaming with strict index ordering
   even under artificially reversed completion times.
2. Cache keying (voice + text), length boundaries, and zero edge_tts calls for router turns.
3. 700ms filler clip behavior (triggers on slow LLM turns, skipped on cached/router turns).
"""

import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.voice_service import (
    _tts_cache_key,
    _select_tts_voice,
    _web_tts_cache,
    _vonage_pcm_cache,
    prewarm_canned_text,
    generate_tts_chunks,
    process_voice_websocket,
)
from app.core.config import settings


def test_cache_key_and_storage_structure():
    """Verify cache keys contain voice and text, and only short canned replies are cached."""
    key_en = _tts_cache_key("en-US-JennyNeural", "Hello, how can I assist you?")
    key_ur = _tts_cache_key("ur-PK-UzmaNeural", "Ji bilkul, batayein.")

    assert key_en == "en-US-JennyNeural:Hello, how can I assist you?"
    assert key_ur == "ur-PK-UzmaNeural:Ji bilkul, batayein."

    async def _run():
        # Prewarm short text
        short_text = "Good morning, thank you for calling."
        with patch("app.services.voice_service.synthesize_full_tts", new_callable=AsyncMock) as mock_synth:
            mock_synth.return_value = b"ID3\x03\x00\x00\x00dummy_mp3_bytes"
            await prewarm_canned_text(short_text, voice="en-US-JennyNeural")

        expected_key = _tts_cache_key("en-US-JennyNeural", short_text)
        assert expected_key in _web_tts_cache
        assert short_text in _web_tts_cache

        # Long text (>450 chars) must NOT be cached
        long_text = "A" * 500
        with patch("app.services.voice_service.synthesize_full_tts", new_callable=AsyncMock) as mock_synth:
            await prewarm_canned_text(long_text)
            mock_synth.assert_not_called()

    asyncio.run(_run())


def test_router_reply_hits_cache_with_zero_edge_tts_calls():
    """Verify router responses use prewarmed cache without calling edge_tts."""
    async def _run():
        router_reply = "Our office is located at Logic Pulse Headquarters, Tech Park, Suite 400."
        voice = _select_tts_voice(router_reply)
        cache_key = _tts_cache_key(voice, router_reply)

        dummy_audio = b"ID3_fake_audio_payload"
        b64_dummy = base64.b64encode(dummy_audio).decode("utf-8")
        _web_tts_cache[cache_key] = [b64_dummy]
        _web_tts_cache[router_reply] = [b64_dummy]

        sent_messages = []
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=lambda msg: sent_messages.append(msg))

        step = 0
        async def mock_receive():
            nonlocal step
            if step == 0:
                step += 1
                return {"text": '{"type": "speech_ended"}'}
            while not any(m.get("type") == "response_end" for m in sent_messages):
                await asyncio.sleep(0.01)
            return {"type": "websocket.disconnect"}

        mock_ws.receive = mock_receive

        with patch("app.services.voice_service.transcribe_audio_bytes", new_callable=AsyncMock) as mock_stt, \
             patch("app.services.voice_service.edge_tts") as mock_edge_tts, \
             patch("app.services.voice_service.get_or_create_user", new_callable=AsyncMock) as mock_user:

            user_obj = MagicMock()
            user_obj.id = 999
            user_obj.name = "Test User"
            user_obj.email = "test@example.com"
            mock_user.return_value = user_obj

            mock_stt.return_value = "where is your office"

            # Prewarm exact router response for "where is your office"
            from app.services.receptionist_router import LOCATION_RESPONSES
            for loc in LOCATION_RESPONSES:
                v = _select_tts_voice(loc)
                _web_tts_cache[_tts_cache_key(v, loc)] = [b64_dummy]
                _web_tts_cache[loc] = [b64_dummy]

            # Run websocket session
            await process_voice_websocket(mock_ws, "+923001234567")

            # Assert edge_tts.Communicate was NEVER instantiated because cache hit
            mock_edge_tts.Communicate.assert_not_called()

            # Verify audio_chunk was sent from cache
            audio_chunks = [m for m in sent_messages if m.get("type") == "audio_chunk"]
            assert len(audio_chunks) >= 1
            assert base64.b64decode(audio_chunks[0]["data"]) == dummy_audio

    asyncio.run(_run())


def test_pipelined_sentence_ordering_reversed_completion():
    """
    Verify producer/consumer yields chunks strictly in ascending index order
    even if sentence 1 completes synthesis before sentence 0.
    """
    async def _run():
        sent_messages = []
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=lambda msg: sent_messages.append(msg))

        step = 0
        async def mock_receive():
            nonlocal step
            if step == 0:
                step += 1
                return {"text": '{"type": "speech_ended"}'}
            while not any(m.get("type") == "response_end" for m in sent_messages):
                await asyncio.sleep(0.01)
            return {"type": "websocket.disconnect"}

        mock_ws.receive = mock_receive

        # We simulate 2 sentences from LLM stream
        async def fake_llm_stream(*args, **kwargs):
            yield "First slow sentence."
            yield "Second fast sentence."

        async def fake_generate_tts_chunks(sentence, voice=None, telemetry_recorder=None):
            if "First" in sentence:
                # Artificial delay on sentence 0
                await asyncio.sleep(0.08)
                yield b"AUDIO_CHUNK_0"
            else:
                # Sentence 1 finishes much faster
                await asyncio.sleep(0.01)
                yield b"AUDIO_CHUNK_1"

        with patch("app.services.voice_service.transcribe_audio_bytes", new_callable=AsyncMock) as mock_stt, \
             patch("app.services.voice_service.get_ai_response_stream", side_effect=fake_llm_stream), \
             patch("app.services.voice_service.generate_tts_chunks", side_effect=fake_generate_tts_chunks), \
             patch("app.services.voice_service.get_or_create_user", new_callable=AsyncMock) as mock_user:

            user_obj = MagicMock()
            user_obj.id = 998
            user_obj.name = "Order Test"
            user_obj.email = "order@example.com"
            mock_user.return_value = user_obj
            mock_stt.return_value = "Tell me two things please."

            # Clear cache for these test sentences
            for s in ("First slow sentence.", "Second fast sentence."):
                _web_tts_cache.pop(s, None)
                _web_tts_cache.pop(_tts_cache_key(settings.tts_voice, s), None)

            await process_voice_websocket(mock_ws, "+923001234567")

            audio_chunks = [m for m in sent_messages if m.get("type") == "audio_chunk" and not m.get("is_filler")]
            assert len(audio_chunks) == 2
            # Verify strict index order
            assert audio_chunks[0]["index"] == 0
            assert base64.b64decode(audio_chunks[0]["data"]) == b"AUDIO_CHUNK_0"
            assert audio_chunks[1]["index"] == 1
            assert base64.b64decode(audio_chunks[1]["data"]) == b"AUDIO_CHUNK_1"

    asyncio.run(_run())


def test_filler_clip_fires_on_slow_llm_and_skips_on_fast_turn():
    """
    Verify 700ms filler clip:
    1. Fires when LLM takes > 700ms to produce audio.
    2. Is skipped / cancelled when audio is ready before 700ms.
    """
    async def _run():
        # Prewarm filler
        filler_audio = b"FILLER_AUDIO_JI"
        b64_filler = base64.b64encode(filler_audio).decode("utf-8")
        _web_tts_cache["Ji,"] = [b64_filler]
        _web_tts_cache[_tts_cache_key(settings.tts_voice, "Ji,")] = [b64_filler]

        # Case 1: Slow LLM turn (>700ms delay)
        sent_messages_slow = []
        mock_ws_slow = AsyncMock()
        mock_ws_slow.send_json = AsyncMock(side_effect=lambda msg: sent_messages_slow.append(msg))

        step_slow = 0
        async def mock_receive_slow():
            nonlocal step_slow
            if step_slow == 0:
                step_slow += 1
                return {"text": '{"type": "speech_ended"}'}
            while not any(m.get("type") == "response_end" for m in sent_messages_slow):
                await asyncio.sleep(0.01)
            return {"type": "websocket.disconnect"}

        mock_ws_slow.receive = mock_receive_slow

        async def slow_llm_stream(*args, **kwargs):
            await asyncio.sleep(0.80)  # Exceeds 700ms
            yield "Finally here is the response."

        async def fast_tts(sentence, voice=None, telemetry_recorder=None):
            yield b"REAL_AUDIO"

        with patch("app.services.voice_service.transcribe_audio_bytes", new_callable=AsyncMock) as mock_stt, \
             patch("app.services.voice_service.get_ai_response_stream", side_effect=slow_llm_stream), \
             patch("app.services.voice_service.generate_tts_chunks", side_effect=fast_tts), \
             patch("app.services.voice_service.get_or_create_user", new_callable=AsyncMock) as mock_user:

            user_obj = MagicMock()
            user_obj.id = 997
            user_obj.name = "Filler Test"
            user_obj.email = "filler@example.com"
            mock_user.return_value = user_obj
            mock_stt.return_value = "Complex query needing thinking time."

            _web_tts_cache.pop("Finally here is the response.", None)
            _web_tts_cache.pop(_tts_cache_key(settings.tts_voice, "Finally here is the response."), None)

            await process_voice_websocket(mock_ws_slow, "+923001234567")

            filler_msgs = [m for m in sent_messages_slow if m.get("type") == "audio_chunk" and m.get("is_filler")]
            assert len(filler_msgs) == 1
            assert filler_msgs[0]["index"] == -1
            assert base64.b64decode(filler_msgs[0]["data"]) == filler_audio

            real_msgs = [m for m in sent_messages_slow if m.get("type") == "audio_chunk" and not m.get("is_filler")]
            assert len(real_msgs) == 1
            assert real_msgs[0]["index"] == 0

        # Case 2: Fast turn (<700ms) - filler must NOT fire
        sent_messages_fast = []
        mock_ws_fast = AsyncMock()
        mock_ws_fast.send_json = AsyncMock(side_effect=lambda msg: sent_messages_fast.append(msg))

        step_fast = 0
        async def mock_receive_fast():
            nonlocal step_fast
            if step_fast == 0:
                step_fast += 1
                return {"text": '{"type": "speech_ended"}'}
            while not any(m.get("type") == "response_end" for m in sent_messages_fast):
                await asyncio.sleep(0.01)
            return {"type": "websocket.disconnect"}

        mock_ws_fast.receive = mock_receive_fast

        async def fast_llm_stream(*args, **kwargs):
            await asyncio.sleep(0.05)
            yield "Quick answer."

        with patch("app.services.voice_service.transcribe_audio_bytes", new_callable=AsyncMock) as mock_stt, \
             patch("app.services.voice_service.get_ai_response_stream", side_effect=fast_llm_stream), \
             patch("app.services.voice_service.generate_tts_chunks", side_effect=fast_tts), \
             patch("app.services.voice_service.get_or_create_user", new_callable=AsyncMock) as mock_user:

            user_obj = MagicMock()
            user_obj.id = 996
            user_obj.name = "Fast Test"
            user_obj.email = "fast@example.com"
            mock_user.return_value = user_obj
            mock_stt.return_value = "Quick query."

            _web_tts_cache.pop("Quick answer.", None)
            _web_tts_cache.pop(_tts_cache_key(settings.tts_voice, "Quick answer."), None)

            await process_voice_websocket(mock_ws_fast, "+923001234567")

            filler_msgs = [m for m in sent_messages_fast if m.get("type") == "audio_chunk" and m.get("is_filler")]
            assert len(filler_msgs) == 0

    asyncio.run(_run())
