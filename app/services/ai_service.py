import logging
import re
from typing import TYPE_CHECKING, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.user import user_crud
from app.db.session import async_session_maker
from app.schemas.tools import CreateAppointmentRequest
from app.services.groq_service import GroqService
from app.services.receptionist_router import (
    ReceptionistRouter,
    _extract_context_from_history,
    hour_to_display,
    normalise_input,
)
from app.services.tools_service import AppointmentToolService
from app.utils.prompts import receptionist_prompt
from app.utils.stream_utils import stream_sentence_chunks

if TYPE_CHECKING:
    from app.services.telemetry import VoiceTelemetryRecorder

logger = logging.getLogger(__name__)


async def _execute_confirmed_appointment(
    query: str,
    history: list[dict[str, str]],
    user_name: str,
    user_email: str,
    user_id: int | None = None,
    db: AsyncSession | None = None,
) -> str | None:
    """
    Checks if the conversation is awaiting confirmation and the user provided explicit confirmation.
    If complete appointment details (day, hour, phone) exist, executes AppointmentToolService.create_appointment
    and returns the authoritative user-facing result.
    """
    ctx = _extract_context_from_history(history)
    if ctx.get("appointment_booked"):
        return None
    if ctx.get("awaiting_confirmation") != "appointment" and ctx.get("awaiting_entity") != "confirmation":
        return None

    norm_query = normalise_input(query).lower()
    is_cancel = bool(re.search(r"\b(no|cancel|different|change|another|stop|nahi|نہیں|نا|منسوخ|کینسل)\b|نہیں|منسوخ", norm_query))
    if is_cancel:
        return None

    is_confirm = bool(
        re.search(
            r"\b(yes|confirm|i confirm|go ahead|proceed|ok|okay|all right|alright|sure|please submit|submit it|submit|please book|book it|correct|good|sounds good|haan|ji haan|theek hai|kardein|ہاں|جی ہاں|جی|ٹھیک ہے|درست|بک کریں|جمع کرائیں)\b|ہاں|جی ہاں|ٹھیک ہے|درست|بک کریں",
            norm_query,
        )
    )
    if not is_confirm:
        return None

    day = ctx.get("selected_day")
    hour = ctx.get("selected_time_hour")
    phone = ctx.get("phone_number")

    if not day or hour is None or not phone:
        return None

    time_disp = hour_to_display(hour)
    req = CreateAppointmentRequest(
        customer_name=user_name or "Valued Client",
        customer_phone=str(phone),
        customer_email=user_email,
        day=str(day),
        time_str=time_disp,
        service="General Consultation",
    )

    if db is not None:
        actual_user_id = user_id
        if actual_user_id is None:
            db_user = await user_crud.get_by_email(db, user_email)
            if not db_user:
                db_user = await user_crud.create(db, name=user_name or "Valued Client", email=user_email)
            actual_user_id = db_user.id
        tool_res = await AppointmentToolService.create_appointment(req=req, user_id=actual_user_id, db=db)
        return tool_res.user_facing_message

    async with async_session_maker() as session:
        actual_user_id = user_id
        if actual_user_id is None:
            db_user = await user_crud.get_by_email(session, user_email)
            if not db_user:
                db_user = await user_crud.create(session, name=user_name or "Valued Client", email=user_email)
            actual_user_id = db_user.id
        tool_res = await AppointmentToolService.create_appointment(req=req, user_id=actual_user_id, db=session)
        return tool_res.user_facing_message


async def get_ai_response_stream(
    user_name: str,
    user_email: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    telemetry_recorder: "VoiceTelemetryRecorder | None" = None,
    user_id: int | None = None,
    db: AsyncSession | None = None,
) -> AsyncGenerator[str, None]:
    """
    Yields sentence-level response chunks incrementally for streaming voice TTS.
    Records TTFT (Time-To-First-Token) on the exact arrival of the first token chunk.
    """
    if telemetry_recorder:
        telemetry_recorder.mark_llm_start()

    history = history or []

    # ── Check appointment confirmation execution ─────────────────────────────
    tool_reply = await _execute_confirmed_appointment(
        query=message,
        history=history,
        user_name=user_name,
        user_email=user_email,
        user_id=user_id,
        db=db,
    )
    if tool_reply is not None:
        logger.info("ai_service_stream: appointment tool executed on user confirmation.")
        if telemetry_recorder:
            telemetry_recorder.record_llm_usage(
                provider="tool_service",
                model="create_appointment",
                input_tokens=0,
                output_tokens=len(tool_reply.split()),
            )
            telemetry_recorder.mark_llm_first_token()
            telemetry_recorder.mark_llm_end()

        async def _tool_tokens():
            yield tool_reply

        async for sentence in stream_sentence_chunks(_tool_tokens()):
            yield sentence
        return

    # ── Layer 1: local router (zero API cost) ─────────────────────────────────
    local_reply = ReceptionistRouter.get_hardcoded_response(
        message=message,
        history=history,
    )
    if local_reply is not None:
        logger.info("ai_service_stream: local router handled query — Groq skipped.")
        if telemetry_recorder:
            telemetry_recorder.record_llm_usage(
                provider="local_router",
                model="regex_rules",
                input_tokens=0,
                output_tokens=0,
            )
            telemetry_recorder.mark_llm_first_token()
            telemetry_recorder.mark_llm_end()

        # If local reply is already cached as a unit, yield directly for instant single-chunk playback
        from app.services.voice_service import _web_tts_cache, _tts_cache_key, _select_tts_voice
        voice = _select_tts_voice(local_reply)
        cache_key = _tts_cache_key(voice, local_reply)
        if cache_key in _web_tts_cache or local_reply.strip() in _web_tts_cache:
            yield local_reply.strip()
            return

        # Otherwise split into natural sentences for TTS
        async def _local_tokens():
            yield local_reply

        async for sentence in stream_sentence_chunks(_local_tokens()):
            yield sentence
        return

    if system_prompt is None:
        system_prompt = receptionist_prompt()
        logger.debug("Using receptionist_prompt for AI response stream")
    else:
        logger.debug("Using provided system prompt for AI response stream")

    # ── Layer 2: LLM Streaming (Groq) ─────────────────────────────────────────
    system_prompt_text = (
        f"{system_prompt} "
        f"The current caller is {user_name} and their email is {user_email}."
    )

    messages = [{"role": "system", "content": system_prompt_text}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    groq = GroqService()
    accumulated_output = []

    async def _token_generator() -> AsyncGenerator[str, None]:
        first_token = True
        try:
            async for token in groq.chat_completion_stream(messages):
                if first_token and telemetry_recorder:
                    telemetry_recorder.mark_llm_first_token()
                    first_token = False
                accumulated_output.append(token)
                yield token
        except GeneratorExit:
            return
        except Exception:
            logger.exception("Failed to stream tokens from Groq API")
            if first_token:
                if telemetry_recorder:
                    telemetry_recorder.mark_llm_first_token()
                yield "Our assistant is currently unavailable. A human representative will contact you shortly."

    try:
        async for sentence in stream_sentence_chunks(_token_generator()):
            yield sentence
    finally:
        if telemetry_recorder:
            full_text = "".join(accumulated_output)
            out_tokens = max(1, len(full_text.split())) if full_text else 0
            in_tokens = sum(len(m.get("content", "").split()) for m in messages)
            telemetry_recorder.record_llm_usage(
                provider="groq_stream",
                model=getattr(groq, "model", "llama-3.1-8b-instant"),
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
            telemetry_recorder.mark_llm_end()


async def get_ai_response(
    user_name: str,
    user_email: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    telemetry_recorder: "VoiceTelemetryRecorder | None" = None,
    user_id: int | None = None,
    db: AsyncSession | None = None,
) -> str:
    if telemetry_recorder:
        telemetry_recorder.mark_llm_start()

    history = history or []

    # ── Check appointment confirmation execution ─────────────────────────────
    tool_reply = await _execute_confirmed_appointment(
        query=message,
        history=history,
        user_name=user_name,
        user_email=user_email,
        user_id=user_id,
        db=db,
    )
    if tool_reply is not None:
        logger.info("ai_service: appointment tool executed on user confirmation.")
        if telemetry_recorder:
            telemetry_recorder.record_llm_usage(
                provider="tool_service",
                model="create_appointment",
                input_tokens=0,
                output_tokens=len(tool_reply.split()),
            )
            telemetry_recorder.mark_llm_end()
        return tool_reply

    # ── Layer 1: local router (zero API cost) ─────────────────────────────────
    local_reply = ReceptionistRouter.get_hardcoded_response(
        message=message,
        history=history,
    )
    if local_reply is not None:
        logger.info("ai_service: local router handled query — Groq skipped.")
        if telemetry_recorder:
            telemetry_recorder.record_llm_usage(
                provider="local_router",
                model="regex_rules",
                input_tokens=0,
                output_tokens=0,
            )
            telemetry_recorder.mark_llm_end()
        return local_reply

    if system_prompt is None:
        system_prompt = receptionist_prompt()
        logger.debug("Using receptionist_prompt for AI response")
    else:
        logger.debug("Using provided system prompt for AI response")

    # ── Layer 2: LLM (Groq) ───────────────────────────────────────────────────
    system_prompt_text = (
        f"{system_prompt} "
        f"The current caller is {user_name} and their email is {user_email}."
    )

    messages = [{"role": "system", "content": system_prompt_text}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    groq = GroqService()
    try:
        completion_result = await groq.chat_completion_with_usage(messages)
        if telemetry_recorder:
            telemetry_recorder.record_llm_usage(
                provider="groq",
                model=completion_result.model,
                input_tokens=completion_result.input_tokens,
                output_tokens=completion_result.output_tokens,
            )
            telemetry_recorder.mark_llm_end()
        return completion_result.content
    except Exception:
        logger.exception("Failed to generate AI response")
        if telemetry_recorder:
            telemetry_recorder.record_llm_usage(
                provider="groq_fallback",
                model="fallback_message",
                input_tokens=0,
                output_tokens=0,
            )
            telemetry_recorder.mark_llm_end()
        return "Our assistant is currently unavailable. A human representative will contact you shortly."

