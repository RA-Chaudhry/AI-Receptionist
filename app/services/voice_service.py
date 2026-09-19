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
from uuid import uuid4
from typing import AsyncGenerator

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.core.config import settings
from app.core.websocket_manager import connection_manager
from app.crud.chat import chat_crud
from app.crud.user import user_crud
from app.db.session import async_session_maker
from app.services.ai_service import get_ai_response, get_ai_response_stream
from app.services.groq_service import GroqService, GroqServiceError
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
    return "Caller", f"voice-{identity_digest}@local.invalid"


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


async def get_recent_voice_history(user_id: int) -> list[dict[str, str]]:
    try:
        async with async_session_maker() as db:
            recent_messages = await chat_crud.get_recent_messages(
                db,
                user_id=user_id,
                limit=settings.short_memory_messages,
            )
            return [
                {"role": message.role, "content": message.content}
                for message in recent_messages
            ]
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
    whisper = _get_whisper_model()
    return {
        "whisper_loaded": whisper is not None,
        "vad_loaded": _vad_processor is not None and _vad_processor.vad is not None,
        "tts_available": edge_tts is not None or pyttsx3 is not None,
    }


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    telemetry_recorder: VoiceTelemetryRecorder | None = None,
) -> str:
    if telemetry_recorder:
        telemetry_recorder.mark_stt_start()
        # 16000 Hz, 1 channel, 16-bit (2 bytes per sample) = 32000 bytes/sec
        audio_dur = len(audio_bytes) / 32000.0 if audio_bytes else 0.0
        telemetry_recorder.record_stt_duration(audio_dur)

    if not audio_bytes:
        logger.warning("Cannot transcribe empty audio bytes.")
        if telemetry_recorder:
            telemetry_recorder.mark_stt_end()
        return ""

    wav_bytes = pcm_to_wav(audio_bytes, sample_rate=16000, channels=1, sampwidth=2)

    # Optional Groq Cloud Whisper path (when enabled & API key configured)
    stt_provider_cfg = getattr(settings, "stt_provider", "faster_whisper")
    if stt_provider_cfg == "groq_whisper" and getattr(settings, "groq_api_key", None):
        try:
            groq_svc = GroqService()
            result, detected_lang, dur = await groq_svc.transcribe_audio_with_usage(wav_bytes)
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

    whisper = _get_whisper_model()
    if whisper is None:
        logger.warning("Local faster-whisper model unavailable; voice transcription disabled.")
        if telemetry_recorder:
            telemetry_recorder.mark_stt_end()
        return ""

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name
            tmp.write(wav_bytes)

        def _transcribe() -> tuple[str, str | None, float | None]:
            segments, info = whisper.transcribe(  # type: ignore[union-attr]
                temp_path,
                beam_size=1,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                condition_on_previous_text=False,
                vad_filter=True,
                language=None,
                initial_prompt="Logic Pulse AI Receptionist: Appointments, timings, services in English and Urdu.",
            )
            text = " ".join(
                segment.text.strip() for segment in segments if segment.text.strip()
            ).strip()
            lang = getattr(info, "language", None)
            lang_prob = getattr(info, "language_probability", None)
            return text, lang, lang_prob

        result, detected_lang, lang_prob = await asyncio.to_thread(_transcribe)
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
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                logger.debug("Failed to delete temporary audio file", exc_info=True)


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

    # Roman Urdu grammatical particles and pronouns (excluding proper nouns like Muhammad, Ramzan, Logic Pulse)
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


async def generate_tts_chunks(
    text: str,
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

        voice_to_use = _select_tts_voice(text)
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



async def process_voice_websocket(websocket: WebSocket, phone_number: str | None) -> None:
    call_id = str(uuid4())
    user = await get_or_create_user(phone_number, anonymous_session_id=call_id)
    turn_detector = TurnDetector()
    audio_accumulator = bytearray()
    loop = asyncio.get_running_loop()
    current_generation_task: asyncio.Task | None = None
    assistant_is_speaking = False
    active_recorder: VoiceTelemetryRecorder | None = None

    async def _build_history() -> list[dict[str, str]]:
        return await get_recent_voice_history(user.id)

    async def _run_generation_flow(pcm_to_process: bytes, recorder: VoiceTelemetryRecorder):
        nonlocal assistant_is_speaking
        try:
            transcript = await transcribe_audio_bytes(pcm_to_process, telemetry_recorder=recorder)
            if not transcript.strip():
                logger.info("Empty transcript from %d bytes audio.", len(pcm_to_process))
                await _safe_send_json(websocket, {"type": "no_speech", "data": "No clear speech detected. Please speak again."})
                recorder.finalize()
                return

            assistant_is_speaking = True
            await _safe_send_json(websocket, {"type": "transcript", "data": transcript})

            transcript_lower = transcript.lower()
            if any(word in transcript_lower for word in ["goodbye", "bye", "exit"]):
                await _safe_send_json(websocket, {"type": "goodbye", "data": "Goodbye!"})
                assistant_is_speaking = False
                recorder.finalize()
                raise WebSocketDisconnect()

            system_prompt = voice_receptionist_prompt()
            await _safe_send_json(websocket, {"type": "response_start", "data": ""})

            chunk_index = 0
            full_response_parts = []
            first_audio_sent = False

            async for sentence in get_ai_response_stream(
                user_name=getattr(user, "name", None) or (phone_number or "guest"),
                user_email=getattr(user, "email", None) or f"{getattr(user, 'name', None) or 'guest'}@example.com",
                message=transcript,
                history=await _build_history(),
                system_prompt=system_prompt,
                telemetry_recorder=recorder,
            ):
                if not sentence.strip():
                    continue
                full_response_parts.append(sentence)

                # Stream TTS audio for this sentence immediately
                sentence_audio_parts = []
                async for audio_chunk in generate_tts_chunks(sentence, telemetry_recorder=recorder):
                    if not first_audio_sent:
                        recorder.mark_audio_sent()
                        first_audio_sent = True
                    sentence_audio_parts.append(audio_chunk)

                if sentence_audio_parts:
                    complete_sentence_audio = b"".join(sentence_audio_parts)
                    encoded = base64.b64encode(complete_sentence_audio).decode()
                    await _safe_send_json(
                        websocket,
                        {
                            "type": "audio_chunk",
                            "data": encoded,
                            "index": chunk_index,
                        },
                    )
                    chunk_index += 1

            full_response = " ".join(full_response_parts).strip()
            cost_obj = getattr(recorder, "cost", None)
            provider_val = getattr(cost_obj, "provider", "")
            is_local = (provider_val == "local_router")
            source_type = "local_rule" if is_local else "ai_llm"
            provider_name = "Local Rule Engine ($0 Cost)" if is_local else f"Groq AI ({getattr(cost_obj, 'model', settings.groq_model)})"
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

            if full_response:
                await save_chat_message(
                    user_id=user.id,
                    user_message=transcript,
                    bot_response=full_response,
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
                turn_result = turn_detector.process_frame(data)

                if turn_result.turn_started:
                    # Audio-driven Barge-in: only interrupt if assistant is actively speaking/streaming audio
                    if assistant_is_speaking and current_generation_task and not current_generation_task.done():
                        current_generation_task.cancel()
                        assistant_is_speaking = False
                        logger.info("Barge-in: Interrupted ongoing voice generation task via audio VAD.")
                        await _safe_send_json(websocket, {"type": "voice_interrupted"})

                    if active_recorder is None:
                        active_recorder = VoiceTelemetryRecorder(call_id=call_id, channel="voice_ws")
                        active_recorder.mark_audio_received()
                        active_recorder.mark_vad_start()

                audio_accumulator.extend(data)
                if len(audio_accumulator) > settings.voice_max_buffer_bytes:
                    logger.warning("Voice buffer limit reached; resetting partial turn")
                    audio_accumulator.clear()
                    turn_detector.reset()
                    active_recorder = None
                    await _safe_send_json(websocket, {"type": "error", "data": "Audio turn exceeded size limit"})
                    continue
                await _safe_send_json(websocket, {"type": "listening", "data": ""})

                # Check if natural turn completion occurred
                if turn_result.turn_completed and turn_result.audio_bytes:
                    pcm_to_process = turn_result.audio_bytes
                    audio_accumulator.clear()

                    if active_recorder is None:
                        active_recorder = VoiceTelemetryRecorder(call_id=call_id, channel="voice_ws")
                        active_recorder.mark_audio_received()
                        active_recorder.mark_vad_start()
                    active_recorder.mark_vad_end()

                    turn_recorder = active_recorder
                    active_recorder = None

                    if assistant_is_speaking and current_generation_task and not current_generation_task.done():
                        current_generation_task.cancel()
                        assistant_is_speaking = False

                    current_generation_task = asyncio.create_task(_run_generation_flow(pcm_to_process, turn_recorder))

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
                    await _safe_send_json(websocket, {"type": "voice_interrupted"})

                elif event_type == "goodbye":
                    logger.info("Goodbye received from client. Closing session.")
                    await _safe_send_json(websocket, {"type": "goodbye", "data": "Goodbye!"})
                    break

                elif event_type == "speech_ended":
                    logger.info("Speech ended received from client.")
                    
                    # If a generation task is already actively running for this turn, do not cancel it
                    # unless we have a large new buffer of speech that was not included.
                    if current_generation_task and not current_generation_task.done():
                        if len(audio_accumulator) < 9600:
                            logger.info("Generation task already processing speech. Keeping active task.")
                            audio_accumulator.clear()
                            turn_detector.reset()
                            continue

                    pcm_to_process = bytes(audio_accumulator) if audio_accumulator else (turn_detector._active_utterance or b"")
                    audio_accumulator.clear()
                    turn_detector.reset()
                    if not pcm_to_process:
                        logger.info("Speech ended received with empty audio buffer.")
                        continue

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

                    current_generation_task = asyncio.create_task(_run_generation_flow(pcm_to_process, turn_recorder))


    finally:
        if current_generation_task and not current_generation_task.done():
            current_generation_task.cancel()
            assistant_is_speaking = False


from collections import defaultdict
import httpx

_sms_history = defaultdict(list)

def check_sms_rate_limit(phone_number: str) -> bool:
    now = time.time()
    window = settings.sms_rate_limit_window_seconds
    limit = settings.sms_rate_limit_count
    
    # Clean up old timestamps
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

    # Keep SMS concise (max 2-3 messages = max 450 chars)
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, data=data)
            resp.raise_for_status()
            resp_data = resp.json()
            
            messages = resp_data.get("messages", [])
            if messages and messages[0].get("status") == "0":
                logger.info("Vonage SMS sent successfully")
                return "sent"
            else:
                status_code = messages[0].get("status") if messages else "unknown"
                error_text = messages[0].get("error-text") if messages else "No response messages"
                logger.error("Vonage SMS failed with provider status %s", status_code)
                return "failed"
    except Exception:
        logger.exception("Failed to send Vonage SMS")
        # System should never crash due to external SMS failure
        return "failed"


async def play_audio_stream(
    connection_id: str,
    text_or_stream: str | AsyncGenerator[str, None],
    cancel_event: asyncio.Event,
    telemetry_recorder: VoiceTelemetryRecorder | None = None,
):
    """
    Streams audio responses to Vonage by converting generated TTS chunks to 16kHz PCM
    and sending them as 20ms binary frames. Stops immediately if cancel_event is set.
    Supports either a static text string or an incremental sentence-level stream.
    """
    try:
        is_first_frame = True

        async def _sentence_iter():
            if isinstance(text_or_stream, str):
                yield text_or_stream
            else:
                async for item in text_or_stream:
                    yield item

        async for sentence in _sentence_iter():
            if cancel_event.is_set():
                break
            if not sentence.strip():
                continue

            async for audio_chunk in generate_tts_chunks(sentence, telemetry_recorder=telemetry_recorder):
                if cancel_event.is_set():
                    break

                # Convert generated TTS bytes (WAV/MP3) to raw PCM (16kHz, 16-bit mono)
                pcm_bytes = mp3_to_pcm(audio_chunk, target_rate=16000)
                if not pcm_bytes:
                    continue

                # Send raw PCM in 20ms chunks (640 bytes = 20ms of 16kHz 16-bit mono PCM)
                chunk_size = 640
                delay = 0.02
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
                    await asyncio.sleep(delay)
    except asyncio.CancelledError:
        logger.info("Playback stream task cancelled.")
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
        # Wait for the initial text message containing metadata from Vonage
        data = await asyncio.wait_for(websocket.receive(), timeout=10.0)
        if "text" in data:
            meta = json.loads(data["text"])
            connection_id = meta.get("uuid") or meta.get("conversation_uuid")
            phone_number = meta.get("from")
            logger.info("Vonage WebSocket connected")
    except Exception as exc:
        logger.warning("Failed to parse initial Vonage metadata. Auto-generating session ID.", exc_info=exc)

    if not connection_id:
        connection_id = str(uuid4())

    await connection_manager.connect(connection_id, websocket)

    user = await get_or_create_user(phone_number, anonymous_session_id=connection_id)
    turn_detector = TurnDetector()
    loop = asyncio.get_running_loop()
    last_activity_time = loop.time()
    active_recorder: VoiceTelemetryRecorder | None = None
    silence_timeout = float(settings.silence_timeout_sec)

    playback_task = None
    cancel_event = asyncio.Event()

    async def _build_history() -> list[dict[str, str]]:
        return await get_recent_voice_history(user.id)

    async def play_reply(text: str | AsyncGenerator[str, None], recorder: VoiceTelemetryRecorder | None = None):
        nonlocal playback_task, cancel_event
        # 1. Cancel previous playback if running
        if playback_task and not playback_task.done():
            cancel_event.set()
            playback_task.cancel()
            await connection_manager.send_json(connection_id, {"action": "clear"})
            logger.info("Barge-in: playback interrupted.")

        # 2. Reset cancellation event
        cancel_event = asyncio.Event()
        # 3. Stream the new response
        playback_task = asyncio.create_task(
            play_audio_stream(connection_id, text, cancel_event, telemetry_recorder=recorder)
        )

    # Trigger welcome greeting on connect
    GREETING_TEXT = "Hello, thank you for calling Logic Pulse. This is Pulse Concierge AI. How can I assist you today?"
    await play_reply(GREETING_TEXT)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=35.0)
            except asyncio.TimeoutError:
                logger.info(f"Session timeout for connection: {connection_id}")
                break

            if "bytes" in msg:
                data = msg["bytes"]
                if not isinstance(data, bytes) or len(data) > settings.voice_max_packet_bytes:
                    logger.warning("Rejected oversized or malformed Vonage audio packet")
                    continue
            elif "text" in msg:
                # Skip event status updates
                continue
            else:
                break

            turn_result = turn_detector.process_frame(data)
            current_time = loop.time()
            turn_recorder: VoiceTelemetryRecorder | None = None

            if turn_result.turn_started:
                last_activity_time = current_time
                # Barge-in: If caller starts speaking during assistant playback, cancel immediately
                if playback_task and not playback_task.done():
                    cancel_event.set()
                    playback_task.cancel()
                    await connection_manager.send_json(connection_id, {"action": "clear"})
                    logger.info("Barge-in: Interrupted current playing audio stream on Vonage.")

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
                active_recorder.mark_vad_end()
                turn_recorder = active_recorder
                active_recorder = None

                transcript = await transcribe_audio_bytes(turn_result.audio_bytes, telemetry_recorder=turn_recorder)
            elif current_time - last_activity_time > silence_timeout:
                logger.info("Silence timeout exceeded")
                break
            else:
                continue

            if not transcript.strip():
                if turn_recorder:
                    turn_recorder.finalize()
                continue

            logger.info("Voice transcript received | length=%d", len(transcript))

            # Check for goodbye exit phrases
            transcript_lower = transcript.lower()
            if any(word in transcript_lower for word in ["goodbye", "bye", "exit"]):
                farewell = "Goodbye! Have a nice day."
                await play_reply(farewell, turn_recorder)
                await asyncio.sleep(2.0)
                break

            system_prompt = voice_receptionist_prompt()
            response_stream = get_ai_response_stream(
                user_name=getattr(user, "name", None) or (phone_number or "guest"),
                user_email=getattr(user, "email", None) or f"{getattr(user, 'name', None) or 'guest'}@example.com",
                message=transcript,
                history=await _build_history(),
                system_prompt=system_prompt,
                telemetry_recorder=turn_recorder,
            )

            full_sentences = []
            async def _stream_and_collect():
                async for s in response_stream:
                    full_sentences.append(s)
                    yield s

            await play_reply(_stream_and_collect(), turn_recorder)

            # Save chat history once playback completes
            async def _save_after_playback():
                if playback_task:
                    try:
                        await playback_task
                    except (asyncio.CancelledError, Exception) as exc:
                        logger.debug("Playback task finished or cancelled: %s", exc)
                full_bot_text = " ".join(full_sentences).strip()
                if full_bot_text:
                    await save_chat_message(
                        user_id=user.id,
                        user_message=transcript,
                        bot_response=full_bot_text,
                    )

            asyncio.create_task(_save_after_playback())


    except WebSocketDisconnect:
        logger.info(f"Vonage websocket disconnected normally: {connection_id}")
    except Exception as exc:
        logger.exception(f"Error in Vonage WebSocket loop for connection {connection_id}")
    finally:
        if playback_task and not playback_task.done():
            playback_task.cancel()
        connection_manager.disconnect(connection_id)
        logger.info(f"Vonage WebSocket session ended: {connection_id}")
