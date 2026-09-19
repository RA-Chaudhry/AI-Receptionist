import logging
import re
from typing import TYPE_CHECKING, AsyncGenerator

from app.services.groq_service import GroqService
from app.utils.prompts import receptionist_prompt
from app.services.receptionist_router import ReceptionistRouter
from app.utils.stream_utils import stream_sentence_chunks

if TYPE_CHECKING:
    from app.services.telemetry import VoiceTelemetryRecorder

logger = logging.getLogger(__name__)


async def get_ai_response_stream(
    user_name: str,
    user_email: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    telemetry_recorder: "VoiceTelemetryRecorder | None" = None,
) -> AsyncGenerator[str, None]:
    """
    Yields sentence-level response chunks incrementally for streaming voice TTS.
    Records TTFT (Time-To-First-Token) on the exact arrival of the first token chunk.
    """
    if telemetry_recorder:
        telemetry_recorder.mark_llm_start()

    history = history or []

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

        # Split local reply into natural sentences for TTS
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
) -> str:
    if telemetry_recorder:
        telemetry_recorder.mark_llm_start()

    history = history or []

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

