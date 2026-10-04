from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from typing import AsyncGenerator
from uuid import uuid4

import httpx
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

from app.core.config import settings
from app.core.websocket_manager import connection_manager
from app.crud.chat import chat_crud
from app.crud.user import user_crud
from app.db.session import async_session_maker
from app.services.ai_service import get_ai_response, get_ai_response_stream
from app.services.groq_service import GroqService, GroqServiceError, get_shared_client
from app.services.receptionist_router import (
    GREETING_RESPONSES,
    TIMING_RESPONSES,
    LOCATION_RESPONSES,
    SERVICES_RESPONSES,
    PRICING_RESPONSES,
    APPOINTMENT_RESPONSES,
    CLARIFICATION_RESPONSES,
    FALLBACK_RESPONSES,
    CLOSING_RESPONSES,
)
from app.services.telemetry import VoiceTelemetryRecorder
from app.utils.audio_buffer import VADProcessor, TurnDetector, pcm_to_wav, mp3_to_pcm, wav_to_pcm
from app.utils.prompts import voice_receptionist_prompt

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency
    WhisperModel = None

try:
    import edge_tts
except Exception:  # pragma: no cover - optional dependency
    edge_tts = None

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None

logger = logging.getLogger(__name__)

_whisper_model = None
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
_vad_processor: VADProcessor | None = None

# Pre-rendered and cached audio clips
_web_tts_cache: dict[str, list[str]] = {}
_vonage_pcm_cache: dict[str, bytes] = {}
_call_language_cache: dict[str, str] = {}
_prewarm_done: bool = False

GREETING_TEXT = "Hello, thank you for calling Logic Pulse. This is Pulse Concierge AI. How can I assist you today?"
FILLER_CLIPS = ["Ji,", "Ek second,", "Sure,"]


def is_exit_phrase(text: str) -> bool:
    """
    Evaluates whether an utterance is an intentional exit/goodbye command.
    Uses word boundaries and ensures the call isn't ended on substrings (e.g. 'maybe')
    or when 'bye' is embedded in a longer conversational query (e.g. 'can you say bye later').
    """
    if not text:
        return False
    cleaned = re.sub(r"[^\w\s]", " ", text.strip().lower())
    words = cleaned.split()
    if not words or len(words) > 5:
        return False

    # Conversational or question indicators
    if any(q in words for q in ("can", "could", "why", "what", "how", "when", "if", "say", "saying", "mean", "world", "tell", "ask")):
        return False

    exact_phrases = {
        "bye", "goodbye", "good bye", "bye bye", "exit", "quit",
        "ok bye", "okay bye", "bye now", "goodbye now", "thanks bye",
        "thank you bye", "allah hafiz", "khuda hafiz", "alvida",
        "goodbye have a nice day", "goodbye and thank you", "goodbye sir", "bye sir", "bye madam"
    }
    cleaned_phrase = " ".join(words)
    if cleaned_phrase in exact_phrases:
        return True

    # 1 or 2 words with explicit exit token
    if len(words) <= 2:
        return bool(re.search(r"\b(goodbye|bye|exit|quit|allah\s*hafiz|khuda\s*hafiz|alvida)\b", cleaned_phrase))

    # 3 words with conversational polite wrapper
    if len(words) == 3 and any(w in ("ok", "okay", "thank", "thanks", "sir", "madam", "now", "see") for w in words):
        return bool(re.search(r"\b(goodbye|bye|exit|quit)\b", cleaned_phrase))

    return False


async def _safe_send_json(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_json(payload)
    except Exception:
        logger.debug("WebSocket send failed; client may have disconnected.", exc_info=True)


def _build_voice_identity(
    phone_number: str | None,
    anonymous_session_id: str | None = None,
) -> tuple[str, str]:
    """Return a non-PII persistence key; anonymous calls never share history."""
    raw_identity = phone_number.strip() if phone_number and phone_number.strip() else anonymous_session_id
    raw_identity = raw_identity or str(uuid4())
    identity_digest = hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()[:24]
    return "Caller", f"voice-{identity_digest}@voice.logicpulse.com"


async def get_or_create_user(phone_number: str | None, anonymous_session_id: str | None = None):
    async with async_session_maker() as db:
        session_key, email = _build_voice_identity(phone_number, anonymous_session_id)
        user = await user_crud.get_by_email(db, email)
        if user:
            return user

        return await user_crud.create(
            db,
            name=session_key,
            email=email,
        )


async def save_chat_message(user_id: int, user_message: str, bot_response: str) -> None:
    try:
        async with async_session_maker() as db:
            await chat_crud.create_message(
                db,
                user_id=user_id,
                role="user",
                content=user_message,
            )
            await chat_crud.create_message(
                db,
                user_id=user_id,
                role="assistant",
                content=bot_response,
            )
    except Exception:
        logger.exception("Failed to save voice chat messages")


_history_cache: dict[int, list[dict[str, str]]] = {}


async def get_recent_voice_history(user_id: int) -> list[dict[str, str]]:
    if user_id in _history_cache:
        return _history_cache[user_id]
    try:
        async with async_session_maker() as db:
            recent_messages = await chat_crud.get_recent_messages(
                db,
                user_id=user_id,
                limit=settings.short_memory_messages,
            )
            history = [
                {"role": message.role, "content": message.content}
                for message in recent_messages
            ]
            _history_cache[user_id] = history
            return history
    except Exception:
        logger.exception("Failed to load voice history")
        return []


def load_models() -> None:
    global _whisper_model, _vad_processor

    if "pytest" in sys.modules or os.getenv("TESTING", "").lower() in ("true", "1"):
        return

    started_at = time.perf_counter()
    try:
        _vad_processor = VADProcessor(
            aggressiveness=settings.vad_aggressiveness,
            sample_rate=16000,
            min_rms=settings.vad_min_rms,
        )

        if WhisperModel is None:
            logger.warning("faster-whisper is unavailable; voice transcription is disabled.")
            return

        _whisper_model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )
        elapsed = time.perf_counter() - started_at
        logger.info("Voice models loaded in %.2fs", elapsed)
    except Exception:
        logger.exception("Failed to load voice models; server will keep running.")


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None and WhisperModel is not None:
        try:
            logger.info("Initializing WhisperModel on demand...")
            _whisper_model = WhisperModel(
                settings.whisper_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
            logger.info("WhisperModel initialized successfully.")
        except Exception:
            logger.exception("Failed to lazy-load WhisperModel")
    return _whisper_model


def get_models_status() -> dict:
    stt_provider_cfg = getattr(settings, "stt_provider", "groq_whisper")
    groq_whisper_available = stt_provider_cfg == "groq_whisper" and bool(getattr(settings, "groq_api_key", None))
    return {
        "whisper_loaded": _whisper_model is not None or groq_whisper_available,
        "vad_loaded": _vad_processor is not None and _vad_processor.vad is not None,
        "tts_available": edge_tts is not None or pyttsx3 is not None,
    }


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    telemetry_recorder: VoiceTelemetryRecorder | None = None,
    call_id: str | None = None,
) -> str:
    if telemetry_recorder:
        telemetry_recorder.mark_stt_start()
        audio_dur = len(audio_bytes) / 32000.0 if audio_bytes else 0.0
        telemetry_recorder.record_stt_duration(audio_dur)

    if not audio_bytes:
        logger.warning("Cannot transcribe empty audio bytes.")
        if telemetry_recorder:
            telemetry_recorder.mark_stt_end()
        return ""

    cached_lang = _call_language_cache.get(call_id) if call_id else None

    # 1. Groq Cloud Whisper path (when enabled & API key configured)
    stt_provider_cfg = getattr(settings, "stt_provider", "groq_whisper")
    if stt_provider_cfg == "groq_whisper" and getattr(settings, "groq_api_key", None):
        try:
            wav_bytes = pcm_to_wav(audio_bytes, sample_rate=16000, channels=1, sampwidth=2)
            groq_svc = GroqService()
            result, detected_lang, dur = await groq_svc.transcribe_audio_with_usage(
                wav_bytes, language=cached_lang
            )
            if call_id and detected_lang and detected_lang != "unknown":
                _call_language_cache[call_id] = detected_lang

            logger.info("Groq transcribed %d bytes | length=%d | lang=%s", len(audio_bytes), len(result), detected_lang)
            if telemetry_recorder:
                telemetry_recorder.record_stt_result(
                    transcript=result,
                    language=detected_lang,
                    language_probability=1.0,
                    provider="groq_whisper",
                    model=getattr(settings, "groq_whisper_model", "whisper-large-v3-turbo"),
                )
                telemetry_recorder.mark_stt_end()
            return result
        except Exception:
            logger.exception("Groq STT failed; falling back to local Faster-Whisper.")

    # 2. Local Faster-Whisper path (direct float32 numpy array, zero disk I/O)
    whisper = _whisper_model
    if whisper is None:
        whisper = _get_whisper_model()

    if whisper is None:
        logger.warning("Local faster-whisper model unavailable; voice transcription disabled.")
        if telemetry_recorder:
            telemetry_recorder.mark_stt_end()
        return ""

    try:
        def _transcribe() -> tuple[str, str | None, float | None]:
            if np is not None:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                segments, info = whisper.transcribe(
                    audio_array,
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=False,
                    language=cached_lang,
                    initial_prompt="Logic Pulse AI Receptionist: Appointments, timings, services in English and Urdu.",
                )
            else:
                wav_bytes = pcm_to_wav(audio_bytes, sample_rate=16000, channels=1, sampwidth=2)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(wav_bytes)
                    tmp_name = tmp.name
                try:
                    segments, info = whisper.transcribe(
                        tmp_name,
                        beam_size=1,
                        temperature=0.0,
                        vad_filter=False,
                        language=cached_lang,
                        initial_prompt="Logic Pulse AI Receptionist: Appointments, timings, services in English and Urdu.",
                    )
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)

            text = " ".join(
                segment.text.strip() for segment in segments if segment.text.strip()
            ).strip()
            lang = getattr(info, "language", None)
            lang_prob = getattr(info, "language_probability", None)
            return text, lang, lang_prob

        result, detected_lang, lang_prob = await asyncio.to_thread(_transcribe)
        if call_id and detected_lang:
            _call_language_cache[call_id] = detected_lang

        logger.info("Transcribed %d bytes | length=%d | lang=%s | prob=%s", len(audio_bytes), len(result), detected_lang, lang_prob)
        if telemetry_recorder:
            telemetry_recorder.record_stt_result(
                transcript=result,
                language=detected_lang,
                language_probability=lang_prob,
                provider="faster_whisper",
                model=getattr(settings, "whisper_model", "base"),
            )
            telemetry_recorder.mark_stt_end()
        return result
    except Exception:
        logger.exception("Audio transcription failed")
        if telemetry_recorder:
            telemetry_recorder.mark_stt_end()
        return ""


def _pyttsx3_to_wav_bytes(text: str) -> bytes:
    if pyttsx3 is None:
        return b""

    temp_path = None
    try:
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name

        engine.setProperty("voice", settings.tts_voice)
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        engine.stop()

        with open(temp_path, "rb") as audio_file:
            return audio_file.read()
    except Exception:
        logger.exception("pyttsx3 fallback synthesis failed")
        return b""
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                logger.debug("Failed to delete temporary TTS file", exc_info=True)


def _select_tts_voice(text: str) -> str:
    # 1. Urdu Nastaliq characters
    if re.search(r"[\u0600-\u06FF]", text):
        return "ur-PK-UzmaNeural"

    # 2. Extract word tokens
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", text)]
    if not words:
        return settings.tts_voice

    # English functional/stop words
    english_markers = {
        "the", "is", "with", "our", "for", "at", "to", "and", "please", "your",
        "we", "are", "you", "can", "what", "which", "would", "like", "schedule",
        "appointment", "meeting", "office", "hours", "contact", "thank", "goodbye",
        "scheduled", "confirmed", "receive", "details", "sms", "services", "system"
    }

    # Roman Urdu grammatical particles and pronouns
    roman_urdu_markers = {
        "aap", "aapki", "aapka", "hamari", "humari", "hain", "hai", "kya", "madad",
        "shukriya", "khushamdeed", "rabta", "salam", "assalam", "karen", "karte",
        "kar", "batao", "kon", "kaun", "kahan", "hum", "mein", "boht", "acha", "theek",
        "walaikum", "subah", "sham", "baje", "karna", "chahte", "sakta", "sakte"
    }

    eng_count = sum(1 for w in words if w in english_markers)
    urdu_count = sum(1 for w in words if w in roman_urdu_markers)

    if urdu_count > eng_count and urdu_count >= 1:
        return "ur-PK-UzmaNeural"
    return settings.tts_voice


def _tts_cache_key(voice: str, text: str) -> str:
    return f"{voice}:{text.strip()}"


async def generate_tts_chunks(
    text: str,
    voice: str | None = None,
    telemetry_recorder: VoiceTelemetryRecorder | None = None,
) -> AsyncGenerator[bytes, None]:
    if not text.strip():
        return

    if telemetry_recorder:
        telemetry_recorder.mark_tts_start()

    yielded_any = False
    try:
        if edge_tts is None:
            raise RuntimeError("edge_tts is unavailable")

        voice_to_use = voice or _select_tts_voice(text)
        communicate = edge_tts.Communicate(text, voice=voice_to_use)
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                data = chunk.get("data")
                if isinstance(data, bytes) and data:
                    if not yielded_any and telemetry_recorder:
                        telemetry_recorder.mark_tts_first_audio()
                    yielded_any = True
                    yield data
    except Exception:
        logger.exception("edge-tts failed; falling back to pyttsx3")
        try:
            audio_bytes = await asyncio.to_thread(_pyttsx3_to_wav_bytes, text)
            if audio_bytes:
                if not yielded_any and telemetry_recorder:
                    telemetry_recorder.mark_tts_first_audio()
                yielded_any = True
                yield audio_bytes
        except Exception:
            logger.exception("pyttsx3 fallback failed")
    finally:
        if telemetry_recorder:
            telemetry_recorder.mark_tts_end()


async def synthesize_full_tts(text: str, voice: str | None = None) -> bytes:
    chunks = []
    async for chunk in generate_tts_chunks(text, voice=voice):
        chunks.append(chunk)
    return b"".join(chunks)


async def prewarm_canned_text(text: str, voice: str | None = None) -> None:
    if not text or len(text.strip()) > 450:
        return
    voice_to_use = voice or _select_tts_voice(text)
    cache_key = _tts_cache_key(voice_to_use, text)
    if cache_key in _web_tts_cache:
        return
    try:
        audio = await synthesize_full_tts(text, voice=voice_to_use)
        if audio:
            b64_val = base64.b64encode(audio).decode("utf-8")
            _web_tts_cache[cache_key] = [b64_val]
            _web_tts_cache[text.strip()] = [b64_val]
            pcm = await asyncio.to_thread(mp3_to_pcm, audio, 16000)
            if pcm:
                _vonage_pcm_cache[cache_key] = pcm
                _vonage_pcm_cache[text.strip()] = pcm
    except Exception as exc:
        logger.debug("Failed prewarming TTS for %s: %s", text[:20], exc)


async def groq_keepalive_worker() -> None:
    """Keeps connection alive to Groq API to eliminate TLS handshake overhead."""
    while True:
        try:
            await asyncio.sleep(45.0)
            if settings.groq_api_key:
                client = get_shared_client()
                url = f"{settings.groq_base_url.rstrip('/')}/models"
                await client.get(url, headers={"Authorization": f"Bearer {settings.groq_api_key}"}, timeout=5.0)
        except Exception:
            pass


async def prewarm_voice_cache() -> None:
    global _prewarm_done
    if _prewarm_done or "pytest" in sys.modules or os.getenv("TESTING", "").lower() in ("true", "1"):
        return
    _prewarm_done = True
    logger.info("Prewarming Voice Cache and AI engines...")

    all_texts = [
        GREETING_TEXT,
        *GREETING_RESPONSES,
        *TIMING_RESPONSES,
        *LOCATION_RESPONSES,
        *SERVICES_RESPONSES,
        *PRICING_RESPONSES,
        *APPOINTMENT_RESPONSES,
        *CLARIFICATION_RESPONSES,
        *FALLBACK_RESPONSES,
        *CLOSING_RESPONSES,
        *FILLER_CLIPS,
        "ہمارے دفتری اوقات پیر سے جمعہ صبح 9 بجے سے شام 5 بجے تک ہیں۔",
        "ہم کسٹم سافٹ ویئر ڈویلپمنٹ، AI اور آٹومیشن سسٹمز، چیٹ بوٹس، بزنس ڈیش بورڈز اور ویب و موبائل ایپلیکیشنز فراہم کرتے ہیں۔",
        "Goodbye! Have a nice day.",
    ]

    # Prewarm Edge-TTS in parallel tasks
    tts_tasks = [prewarm_canned_text(txt) for txt in all_texts]
    for filler in FILLER_CLIPS:
        tts_tasks.append(prewarm_canned_text(filler, voice="ur-PK-UzmaNeural"))
        tts_tasks.append(prewarm_canned_text(filler, voice=settings.tts_voice))

    await asyncio.gather(*tts_tasks, return_exceptions=True)
    logger.info("Prewarmed %d TTS response entries into in-memory cache.", len(_web_tts_cache))

    # Prewarm Groq LLM
    if settings.groq_api_key:
        try:
            groq_svc = GroqService()
            await groq_svc.chat_completion([{"role": "user", "content": "ping"}])
            logger.info("Groq LLM connection prewarmed successfully.")
        except Exception as exc:
            logger.debug("Groq prewarm ping failed: %s", exc)

    # Start Groq background keepalive task (runs every 45s)
    asyncio.create_task(groq_keepalive_worker())


async def process_voice_websocket(websocket: WebSocket, phone_number: str | None) -> None:
    call_id = str(uuid4())
    user = await get_or_create_user(phone_number, anonymous_session_id=call_id)
    # Load history ONCE at call start into memory to eliminate DB reads on the critical path
    session_history = await get_recent_voice_history(user.id)

    turn_detector = TurnDetector()
    current_generation_task: asyncio.Task | None = None
    assistant_is_speaking = False
    assistant_speaking_started_at: float = 0.0
    active_recorder: VoiceTelemetryRecorder | None = None
    client_listening_sent = False
    loop = asyncio.get_running_loop()

    async def _run_generation_flow(pcm_to_process: bytes, recorder: VoiceTelemetryRecorder):
        nonlocal assistant_is_speaking, assistant_speaking_started_at
        try:
            transcript = await transcribe_audio_bytes(pcm_to_process, telemetry_recorder=recorder, call_id=call_id)
            if not transcript.strip():
                logger.info("Empty transcript from %d bytes audio.", len(pcm_to_process))
                await _safe_send_json(websocket, {"type": "no_speech", "data": "No clear speech detected. Please speak again."})
                recorder.finalize()
                recorder.log_latency_summary()
                return

            assistant_is_speaking = True
            assistant_speaking_started_at = loop.time()
            await _safe_send_json(websocket, {"type": "transcript", "data": transcript})

            if is_exit_phrase(transcript):
                await _safe_send_json(websocket, {"type": "goodbye", "data": "Goodbye!"})
                assistant_is_speaking = False
                recorder.finalize()
                recorder.log_latency_summary()
                raise WebSocketDisconnect()

            system_prompt = voice_receptionist_prompt()
            await _safe_send_json(websocket, {"type": "response_start", "data": ""})

            # Pipelined LLM and TTS Producer/Consumer
            task_queue: asyncio.Queue[asyncio.Task[tuple[int, str, bytes] | None] | None] = asyncio.Queue()
            first_audio_sent = False
            cancel_filler = asyncio.Event()

            # 700ms filler task: if LLM-bound turn has no audio within 700ms, play one filler
            async def _filler_timer():
                try:
                    await asyncio.sleep(0.700)
                    if not cancel_filler.is_set() and not first_audio_sent:
                        voice_pref = settings.tts_voice
                        filler_clip = (
                            _web_tts_cache.get(_tts_cache_key(voice_pref, "Ji,"))
                            or _web_tts_cache.get(_tts_cache_key("ur-PK-UzmaNeural", "Ji,"))
                            or _web_tts_cache.get("Ji,")
                            or _web_tts_cache.get("Sure,")
                            or _web_tts_cache.get(_tts_cache_key(voice_pref, "Sure,"))
                        )
                        if filler_clip:
                            recorder.mark_send_start()
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "audio_chunk",
                                    "data": filler_clip[0],
                                    "index": -1,
                                    "is_filler": True,
                                },
                            )
                            recorder.mark_audio_sent()
                except asyncio.CancelledError:
                    pass

            filler_task = asyncio.create_task(_filler_timer())

            producer_error: Exception | None = None
            full_response_parts: list[str] = []

            async def _synthesize_single_sentence(chunk_idx: int, sent_text: str) -> tuple[int, str, bytes] | None:
                voice = _select_tts_voice(sent_text)
                cache_key = _tts_cache_key(voice, sent_text)
                cached = _web_tts_cache.get(cache_key) or _web_tts_cache.get(sent_text.strip())
                if cached:
                    full_b64 = "".join(cached)
                    return (chunk_idx, sent_text, base64.b64decode(full_b64))

                sentence_audio_parts = []
                async for audio_chunk in generate_tts_chunks(sent_text, voice=voice, telemetry_recorder=recorder):
                    sentence_audio_parts.append(audio_chunk)

                if sentence_audio_parts:
                    return (chunk_idx, sent_text, b"".join(sentence_audio_parts))
                return None

            async def _llm_tts_producer():
                nonlocal producer_error
                try:
                    chunk_index = 0
                    async for sentence in get_ai_response_stream(
                        user_name=getattr(user, "name", None) or (phone_number or "guest"),
                        user_email=getattr(user, "email", None) or f"{getattr(user, 'name', None) or 'guest'}@example.com",
                        message=transcript,
                        history=session_history,
                        system_prompt=system_prompt,
                        telemetry_recorder=recorder,
                        user_id=getattr(user, "id", None),
                    ):
                        if not sentence.strip():
                            continue
                        full_response_parts.append(sentence)

                        t = asyncio.create_task(_synthesize_single_sentence(chunk_index, sentence))
                        await task_queue.put(t)
                        chunk_index += 1

                    await task_queue.put(None)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    producer_error = exc
                    await task_queue.put(None)

            producer_task = asyncio.create_task(_llm_tts_producer())
            pending_synth_tasks: list[asyncio.Task] = []

            try:
                while True:
                    task = await task_queue.get()
                    if task is None:
                        break
                    pending_synth_tasks.append(task)
                    item = await task
                    if item is None:
                        continue

                    chunk_idx, sent_text, audio_bytes = item
                    encoded = base64.b64encode(audio_bytes).decode("utf-8")
                    recorder.mark_send_start()
                    await _safe_send_json(
                        websocket,
                        {
                            "type": "audio_chunk",
                            "data": encoded,
                            "index": chunk_idx,
                        },
                    )
                    if not first_audio_sent:
                        cancel_filler.set()
                        if not filler_task.done():
                            filler_task.cancel()
                        recorder.mark_audio_sent()
                        first_audio_sent = True

                await producer_task
                if producer_error:
                    raise producer_error

            finally:
                cancel_filler.set()
                if not filler_task.done():
                    filler_task.cancel()
                if not producer_task.done():
                    producer_task.cancel()
                for t in pending_synth_tasks:
                    if not t.done():
                        t.cancel()

            full_response = " ".join(full_response_parts).strip()
            cost_obj = getattr(recorder, "cost", None)
            provider_val = getattr(cost_obj, "provider", "")
            is_local = (provider_val in ("local_router", "tool_service"))
            source_type = "tool_service" if provider_val == "tool_service" else ("local_rule" if is_local else "ai_llm")
            provider_name = "Deterministic Appointment Tool" if provider_val == "tool_service" else ("Local Rule Engine ($0 Cost)" if is_local else f"Groq AI ({getattr(cost_obj, 'model', settings.groq_model)})")
            await _safe_send_json(
                websocket,
                {
                    "type": "response_end",
                    "data": full_response,
                    "source": source_type,
                    "provider": provider_name,
                },
            )
            assistant_is_speaking = False
            recorder.finalize()
            recorder.log_latency_summary()

            if full_response:
                session_history.append({"role": "user", "content": transcript})
                session_history.append({"role": "assistant", "content": full_response})
                max_history = getattr(settings, "short_memory_messages", 12)
                if len(session_history) > max_history:
                    del session_history[:-max_history]
                asyncio.create_task(
                    save_chat_message(
                        user_id=user.id,
                        user_message=transcript,
                        bot_response=full_response,
                    )
                )
        except asyncio.CancelledError:
            logger.info("Generation flow cancelled due to interruption.")
            assistant_is_speaking = False
            recorder.finalize()
        except WebSocketDisconnect:
            assistant_is_speaking = False
            raise
        except Exception:
            logger.exception("Error in voice generation flow")
            assistant_is_speaking = False
            recorder.finalize()
            await _safe_send_json(websocket, {"type": "error", "data": "Failed to process audio"})

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=35)
            except asyncio.TimeoutError:
                await _safe_send_json(websocket, {"type": "timeout", "data": "Session ended"})
                break
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception:
                logger.exception("WebSocket receive failed")
                break

            if msg.get("type") == "websocket.disconnect":
                logger.info("WebSocket disconnect frame received from client.")
                break

            if "bytes" in msg:
                data = msg["bytes"]
                if not isinstance(data, bytes) or len(data) > settings.voice_max_packet_bytes:
                    logger.warning("Rejected oversized or malformed voice packet")
                    await _safe_send_json(websocket, {"type": "error", "data": "Audio packet rejected"})
                    continue

                if not client_listening_sent:
                    await _safe_send_json(websocket, {"type": "listening", "data": ""})
                    client_listening_sent = True

                turn_result = turn_detector.process_frame(data)

                if turn_result.turn_started:
                    # Audio-driven Barge-in: only interrupt if assistant is actively speaking/streaming audio
                    if assistant_is_speaking and current_generation_task and not current_generation_task.done():
                        guard_ms = getattr(settings, "barge_in_guard_ms", 300)
                        time_since_speech_ms = (loop.time() - assistant_speaking_started_at) * 1000.0
                        # Ignore barge-in for first BARGE_IN_GUARD_MS (speaker echo protection)
                        if time_since_speech_ms >= guard_ms:
                            current_generation_task.cancel()
                            assistant_is_speaking = False
                            logger.info("Barge-in: Interrupted ongoing voice generation task via audio VAD.")
                            await _safe_send_json(websocket, {"type": "voice_interrupted"})

                    if active_recorder is None:
                        active_recorder = VoiceTelemetryRecorder(call_id=call_id, channel="voice_ws")
                        active_recorder.mark_audio_received()
                        active_recorder.mark_vad_start()

                # Check if natural turn completion occurred
                if turn_result.turn_completed and turn_result.audio_bytes:
                    client_listening_sent = False
                    pcm_to_process = turn_result.audio_bytes

                    if active_recorder is None:
                        active_recorder = VoiceTelemetryRecorder(call_id=call_id, channel="voice_ws")
                        active_recorder.mark_audio_received()
                        active_recorder.mark_vad_start()
                    active_recorder.mark_vad_end(
                        endpointing_delay_ms=turn_result.endpointing_delay_ms
                    )

                    turn_recorder = active_recorder
                    active_recorder = None

                    if assistant_is_speaking and current_generation_task and not current_generation_task.done():
                        current_generation_task.cancel()
                        assistant_is_speaking = False

                    current_generation_task = asyncio.create_task(
                        _run_generation_flow(pcm_to_process, turn_recorder)
                    )

            elif "text" in msg:
                try:
                    event_data = json.loads(msg["text"])
                except Exception:
                    logger.warning("Failed to parse JSON text from WebSocket")
                    continue

                event_type = event_data.get("type")
                if event_type == "interrupt":
                    logger.info("Interrupt received from client.")
                    turn_detector.reset()
                    if current_generation_task and not current_generation_task.done():
                        current_generation_task.cancel()
                        assistant_is_speaking = False
                        logger.info("Cancelled ongoing LLM/TTS generation task.")
                    active_recorder = None
                    client_listening_sent = False
                    await _safe_send_json(websocket, {"type": "voice_interrupted"})

                elif event_type == "goodbye":
                    logger.info("Goodbye received from client. Closing session.")
                    await _safe_send_json(websocket, {"type": "goodbye", "data": "Goodbye!"})
                    break

                elif event_type == "speech_ended":
                    logger.info("Speech ended received from client.")
                    # If TurnDetector has no active utterance, the turn was already completed by server VAD
                    if not turn_detector._active_utterance:
                        logger.debug("Ignored speech_ended event: turn already processed by server VAD.")
                        continue

                    pcm_to_process = bytes(turn_detector._active_utterance)
                    turn_detector.reset()
                    client_listening_sent = False

                    if active_recorder is None:
                        active_recorder = VoiceTelemetryRecorder(call_id=call_id, channel="voice_ws")
                        active_recorder.mark_audio_received()
                        active_recorder.mark_vad_start()
                    active_recorder.mark_vad_end()

                    turn_recorder = active_recorder
                    active_recorder = None

                    if current_generation_task and not current_generation_task.done():
                        current_generation_task.cancel()
                        assistant_is_speaking = False

                    current_generation_task = asyncio.create_task(
                        _run_generation_flow(pcm_to_process, turn_recorder)
                    )

    finally:
        if current_generation_task and not current_generation_task.done():
            current_generation_task.cancel()
            assistant_is_speaking = False


_sms_history = defaultdict(list)


def check_sms_rate_limit(phone_number: str) -> bool:
    now = time.time()
    window = settings.sms_rate_limit_window_seconds
    limit = settings.sms_rate_limit_count
    
    _sms_history[phone_number] = [t for t in _sms_history[phone_number] if now - t < window]
    if len(_sms_history[phone_number]) >= limit:
        return False
        
    _sms_history[phone_number].append(now)
    return True


async def send_vonage_sms(to_phone: str, body: str) -> str:
    """
    Sends an SMS using Vonage SMS REST API.
    Returns status: 'sent', 'failed', 'rate_limited', or 'disabled'.
    """
    if not settings.vonage_api_key or not settings.vonage_api_secret or not settings.vonage_from_number:
        logger.warning("Vonage SMS credentials are not fully configured.")
        return "disabled"

    if not check_sms_rate_limit(to_phone):
        logger.warning("SMS rate limit exceeded")
        return "rate_limited"

    max_chars = 450
    if len(body) > max_chars:
        body = body[:max_chars - 3] + "..."

    try:
        url = "https://rest.nexmo.com/sms/json"
        data = {
            "api_key": settings.vonage_api_key,
            "api_secret": settings.vonage_api_secret,
            "from": settings.vonage_from_number,
            "to": to_phone,
            "text": body
        }
        client = get_shared_client()
        resp = await client.post(url, data=data)
        resp.raise_for_status()
        resp_data = resp.json()
        
        messages = resp_data.get("messages", [])
        if messages and messages[0].get("status") == "0":
            logger.info("Vonage SMS sent successfully")
            return "sent"
        else:
            status_code = messages[0].get("status") if messages else "unknown"
            logger.error("Vonage SMS failed with provider status %s", status_code)
            return "failed"
    except Exception:
        logger.exception("Failed to send Vonage SMS")
        return "failed"


async def play_audio_stream(
    connection_id: str,
    text_or_stream: str | AsyncGenerator[str, None],
    cancel_event: asyncio.Event,
    telemetry_recorder: VoiceTelemetryRecorder | None = None,
):
    """
    Streams audio responses to Vonage by converting generated TTS chunks to 16kHz PCM
    and sending them as 20ms binary frames with deadline clock pacing.
    """
    try:
        is_first_frame = True
        loop = asyncio.get_running_loop()

        # 1. If text is pre-rendered in cache, stream immediately
        if isinstance(text_or_stream, str):
            voice = _select_tts_voice(text_or_stream)
            cache_key = _tts_cache_key(voice, text_or_stream)
            cached_pcm = _vonage_pcm_cache.get(cache_key) or _vonage_pcm_cache.get(text_or_stream.strip())
            if cached_pcm:
                pcm_bytes = cached_pcm
                chunk_size = 640
                next_t = loop.time()
                for i in range(0, len(pcm_bytes), chunk_size):
                    if cancel_event.is_set():
                        break
                    packet = pcm_bytes[i : i + chunk_size]
                    if len(packet) < chunk_size:
                        packet = packet + b"\x00" * (chunk_size - len(packet))

                    await connection_manager.send_bytes(connection_id, packet)
                    if is_first_frame and telemetry_recorder:
                        telemetry_recorder.mark_audio_sent()
                        is_first_frame = False

                    next_t += 0.02
                    sleep_dur = next_t - loop.time()
                    if sleep_dur > 0:
                        await asyncio.sleep(sleep_dur)
                return

        async def _sentence_iter():
            if isinstance(text_or_stream, str):
                yield text_or_stream
            else:
                async for item in text_or_stream:
                    yield item

        # Sentence-level producer/consumer for audio synthesis and playback
        pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=10)

        async def _synth_producer():
            try:
                async for sentence in _sentence_iter():
                    if cancel_event.is_set():
                        break
                    if not sentence.strip():
                        continue

                    # Check cached sentence PCM
                    voice = _select_tts_voice(sentence)
                    cache_key = _tts_cache_key(voice, sentence)
                    cached_pcm = _vonage_pcm_cache.get(cache_key) or _vonage_pcm_cache.get(sentence.strip())
                    if cached_pcm:
                        await pcm_queue.put(cached_pcm)
                        continue

                    sentence_parts = []
                    async for audio_chunk in generate_tts_chunks(sentence, voice=voice, telemetry_recorder=telemetry_recorder):
                        if cancel_event.is_set():
                            break
                        sentence_parts.append(audio_chunk)

                    if sentence_parts:
                        full_audio = b"".join(sentence_parts)
                        # Decode mp3 to pcm in worker thread
                        pcm_bytes = await asyncio.to_thread(mp3_to_pcm, full_audio, 16000)
                        if pcm_bytes:
                            await pcm_queue.put(pcm_bytes)

                await pcm_queue.put(None)
            except Exception as exc:
                logger.debug("Vonage synth producer exception: %s", exc)
                await pcm_queue.put(None)

        synth_task = asyncio.create_task(_synth_producer())

        try:
            next_t = loop.time()
            chunk_size = 640
            while True:
                pcm_sentence = await pcm_queue.get()
                if pcm_sentence is None or cancel_event.is_set():
                    break

                for i in range(0, len(pcm_sentence), chunk_size):
                    if cancel_event.is_set():
                        break
                    packet = pcm_sentence[i : i + chunk_size]
                    if len(packet) < chunk_size:
                        packet = packet + b"\x00" * (chunk_size - len(packet))

                    await connection_manager.send_bytes(connection_id, packet)
                    if is_first_frame and telemetry_recorder:
                        telemetry_recorder.mark_audio_sent()
                        is_first_frame = False

                    next_t += 0.02
                    sleep_dur = next_t - loop.time()
                    if sleep_dur > 0:
                        await asyncio.sleep(sleep_dur)

            await synth_task
        finally:
            if not synth_task.done():
                synth_task.cancel()

    except asyncio.CancelledError:
        logger.info("Vonage playback stream task cancelled.")
    except Exception:
        logger.exception("Error in play_audio_stream")
    finally:
        if telemetry_recorder:
            telemetry_recorder.finalize()


async def process_vonage_websocket(websocket: WebSocket) -> None:
    """
    Manages Vonage WebSocket VoIP session with VAD, Barge-in, and 1.0s latency thresholds.
    """
    connection_id = None
    phone_number = None
    try:
        data = await asyncio.wait_for(websocket.receive(), timeout=10.0)
        if "text" in data:
            meta = json.loads(data["text"])
            connection_id = meta.get("uuid") or meta.get("conversation_uuid")
            phone_number = meta.get("from")
            logger.info("Vonage WebSocket connected: %s", connection_id)
    except Exception as exc:
        logger.warning("Failed to parse initial Vonage metadata. Auto-generating session ID.", exc_info=exc)

    if not connection_id:
        connection_id = str(uuid4())

    await connection_manager.connect(connection_id, websocket)

    cancel_event = asyncio.Event()
    playback_task: asyncio.Task | None = None
    current_generation_task: asyncio.Task | None = None
    loop = asyncio.get_running_loop()

    async def play_vonage_audio(text_or_stream: str | AsyncGenerator[str, None], recorder: VoiceTelemetryRecorder | None = None):
        nonlocal playback_task, cancel_event
        if playback_task and not playback_task.done():
            cancel_event.set()
            playback_task.cancel()
            await connection_manager.send_json(connection_id, {"action": "clear"})
            logger.info("Barge-in: Vonage playback interrupted.")

        cancel_event = asyncio.Event()
        playback_task = asyncio.create_task(
            play_audio_stream(connection_id, text_or_stream, cancel_event, telemetry_recorder=recorder)
        )

    # 1. Play greeting IMMEDIATELY from pre-rendered cache
    await play_vonage_audio(GREETING_TEXT)

    # 2. Parallel user retrieval and history initialization
    user_task = asyncio.create_task(get_or_create_user(phone_number, anonymous_session_id=connection_id))
    user = await user_task
    session_history = await get_recent_voice_history(user.id)

    turn_detector = TurnDetector()
    last_activity_time = loop.time()
    active_recorder: VoiceTelemetryRecorder | None = None
    silence_timeout = float(settings.silence_timeout_sec)

    async def _run_vonage_turn(pcm_data: bytes, recorder: VoiceTelemetryRecorder):
        try:
            transcript = await transcribe_audio_bytes(pcm_data, telemetry_recorder=recorder, call_id=connection_id)
            if not transcript.strip():
                recorder.finalize()
                recorder.log_latency_summary()
                return

            if is_exit_phrase(transcript):
                farewell = "Goodbye! Have a nice day."
                await play_vonage_audio(farewell, recorder)
                recorder.finalize()
                recorder.log_latency_summary()
                await asyncio.sleep(2.0)
                return

            system_prompt = voice_receptionist_prompt()
            response_stream = get_ai_response_stream(
                user_name=getattr(user, "name", None) or (phone_number or "guest"),
                user_email=getattr(user, "email", None) or f"{getattr(user, 'name', None) or 'guest'}@example.com",
                message=transcript,
                history=session_history,
                system_prompt=system_prompt,
                telemetry_recorder=recorder,
                user_id=getattr(user, "id", None),
            )

            full_sentences = []
            async def _collect_stream():
                async for sentence in response_stream:
                    if sentence.strip():
                        full_sentences.append(sentence)
                        yield sentence

            await play_vonage_audio(_collect_stream(), recorder)

            # Wait for playback to finish and save history
            if playback_task:
                await playback_task
            full_bot_text = " ".join(full_sentences).strip()
            if full_bot_text:
                session_history.append({"role": "user", "content": transcript})
                session_history.append({"role": "assistant", "content": full_bot_text})
                max_history = getattr(settings, "short_memory_messages", 12)
                if len(session_history) > max_history:
                    del session_history[:-max_history]
                asyncio.create_task(
                    save_chat_message(
                        user_id=user.id,
                        user_message=transcript,
                        bot_response=full_bot_text,
                    )
                )
            recorder.log_latency_summary()
        except asyncio.CancelledError:
            logger.info("Vonage turn cancelled due to barge-in.")
        except Exception as exc:
            logger.exception("Error in Vonage turn processing: %s", exc)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=35.0)
            except asyncio.TimeoutError:
                logger.info("Session timeout for Vonage connection: %s", connection_id)
                break

            if "bytes" in msg:
                data = msg["bytes"]
                if not isinstance(data, bytes) or len(data) > settings.voice_max_packet_bytes:
                    continue
            elif "text" in msg:
                continue
            else:
                break

            turn_result = turn_detector.process_frame(data)
            current_time = loop.time()

            if turn_result.turn_started:
                last_activity_time = current_time
                # Barge-in: If caller starts speaking, cancel playback and any running turn immediately
                if playback_task and not playback_task.done():
                    cancel_event.set()
                    playback_task.cancel()
                    await connection_manager.send_json(connection_id, {"action": "clear"})
                    logger.info("Barge-in: Cancelled playback on Vonage.")
                if current_generation_task and not current_generation_task.done():
                    current_generation_task.cancel()

                if active_recorder is None:
                    active_recorder = VoiceTelemetryRecorder(call_id=connection_id, channel="vonage_voice")
                    active_recorder.mark_audio_received()
                    active_recorder.mark_vad_start()

            elif turn_result.is_speech:
                last_activity_time = current_time

            if turn_result.turn_completed and turn_result.audio_bytes:
                last_activity_time = current_time
                if active_recorder is None:
                    active_recorder = VoiceTelemetryRecorder(call_id=connection_id, channel="vonage_voice")
                    active_recorder.mark_audio_received()
                    active_recorder.mark_vad_start()
                active_recorder.mark_vad_end(
                    endpointing_delay_ms=turn_result.endpointing_delay_ms
                )
                turn_recorder = active_recorder
                active_recorder = None

                if current_generation_task and not current_generation_task.done():
                    current_generation_task.cancel()

                current_generation_task = asyncio.create_task(_run_vonage_turn(turn_result.audio_bytes, turn_recorder))

            elif current_time - last_activity_time > silence_timeout:
                logger.info("Silence timeout exceeded on Vonage connection %s", connection_id)
                break

    except WebSocketDisconnect:
        logger.info("Vonage websocket disconnected normally: %s", connection_id)
    except Exception as exc:
        logger.exception("Error in Vonage WebSocket loop for %s: %s", connection_id, exc)
    finally:
        if playback_task and not playback_task.done():
            playback_task.cancel()
        if current_generation_task and not current_generation_task.done():
            current_generation_task.cancel()
        connection_manager.disconnect(connection_id)
        logger.info("Vonage WebSocket session ended: %s", connection_id)
