from sqlalchemy.ext.asyncio import AsyncSession
import logging

from fastapi import BackgroundTasks
from app.core.config import settings
from app.crud.chat import chat_crud
from app.crud.user import user_crud
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.groq_service import GroqService
from app.utils.prompts import receptionist_prompt
from app.services.email_service import EmailService
from app.services.voice_service import send_vonage_sms
from app.services.logger import log_interaction
import re
from app.schemas.tools import CreateAppointmentRequest
from app.services.receptionist_router import (
    ReceptionistRouter,
    _extract_context_from_history,
    normalise_input,
    hour_to_display,
)
from app.services.tools_service import AppointmentToolService

logger = logging.getLogger(__name__)


async def _send_notifications_and_log(
    email: str | None,
    phone_number: str | None,
    query: str,
    response: str,
):
    delivery_status = {
        "email": "not_applicable",
        "sms": "not_applicable",
        "call": "not_applicable"
    }

    # Asynchronously send email
    if email:
        email_service = EmailService()
        success = await email_service.send_email(
            to_email=email,
            subject="Pulse Concierge Response",
            body=response,
        )
        delivery_status["email"] = "sent" if success else "failed"

    # Asynchronously send SMS
    if phone_number:
        sms_status = await send_vonage_sms(phone_number, response)
        delivery_status["sms"] = sms_status

    # Log structured interaction
    log_interaction(
        identifier=email or phone_number or "anonymous",
        query=query,
        response=response,
        delivery_status=delivery_status,
    )


class ChatService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        # GroqService init is deferred to avoid raising on missing key when
        # the local router handles the request without needing the LLM.
        self._groq: GroqService | None = None

    def _get_groq(self) -> GroqService:
        if self._groq is None:
            self._groq = GroqService()
        return self._groq

    async def handle_chat(self, payload: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
        user = await user_crud.get_or_create(
            self.db,
            user_id=payload.user_id,
            name=payload.user_name,
            email=payload.user_email,
        )

        await chat_crud.create_message(
            self.db,
            user_id=user.id,
            role="user",
            content=payload.message,
        )

        # Fetch recent history (used both by the router and, if needed, by Groq)
        recent_messages = await chat_crud.get_recent_messages(
            self.db,
            user_id=user.id,
            limit=settings.short_memory_messages,
        )

        history: list[dict[str, str]] = [
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ]

        assistant_reply: str | None = None
        source = "local_rule"
        provider = "Local Rule Engine ($0 Cost)"

        # ── Check if user is confirming an appointment ready for backend tool execution ──
        ctx = _extract_context_from_history(history)
        norm_query = normalise_input(payload.message).lower()
        is_cancel = bool(re.search(r"\b(no|cancel|different|change|another|stop|nahi|نہیں|نا|منسوخ|کینسل)\b|نہیں|منسوخ", norm_query))
        is_confirm = not is_cancel and bool(
            re.search(
                r"\b(yes|confirm|i confirm|go ahead|proceed|ok|okay|sure|please submit|submit it|submit|please book|book it|correct|good|sounds good|haan|ji haan|theek hai|kardein|ہاں|جی ہاں|جی|ٹھیک ہے|درست|بک کریں|جمع کرائیں)\b|ہاں|جی ہاں|ٹھیک ہے|درست|بک کریں",
                norm_query,
            )
        )

        if not ctx.get("appointment_booked") and (ctx.get("awaiting_confirmation") == "appointment" or ctx.get("awaiting_entity") == "confirmation") and is_confirm:
            day = ctx.get("selected_day")
            hour = ctx.get("selected_time_hour")
            phone = ctx.get("phone_number") or payload.phone_number
            if day and hour is not None and phone:
                time_disp = hour_to_display(hour)
                req = CreateAppointmentRequest(
                    customer_name=user.name or "Valued Client",
                    customer_phone=str(phone),
                    customer_email=user.email,
                    day=str(day),
                    time_str=time_disp,
                    service="General Consultation",
                )
                tool_res = await AppointmentToolService.create_appointment(
                    req=req,
                    user_id=user.id,
                    db=self.db,
                )
                assistant_reply = tool_res.user_facing_message
                source = "tool_service"
                provider = "Deterministic Appointment Tool"

        if assistant_reply is None:
            # ── Layer 1: local rule-based router (zero API cost) ──────────────────
            local_reply = ReceptionistRouter.get_hardcoded_response(
                message=payload.message,
                history=history,
            )

            if local_reply is not None:
                logger.info("ChatService: local router handled the query — Groq skipped.")
                assistant_reply = local_reply
                source = "local_rule"
                provider = "Local Rule Engine ($0 Cost)"
            else:
                # ── Layer 2: LLM fallback for complex / multi-intent queries ──────
                messages: list[dict[str, str]] = [{"role": "system", "content": receptionist_prompt()}]
                messages.extend(history)

                try:
                    groq_svc = self._get_groq()
                    assistant_reply = await groq_svc.chat_completion(messages)
                    source = "ai_llm"
                    provider = f"Groq AI ({groq_svc.model})"
                except Exception:
                    logger.exception("AI generation failed in ChatService")
                    assistant_reply = (
                        "Our assistant is currently unavailable. "
                        "A human representative will contact you shortly."
                    )
                    source = "fallback"
                    provider = "System Fallback"

        assistant_message = await chat_crud.create_message(
            self.db,
            user_id=user.id,
            role="assistant",
            content=assistant_reply,
        )

        # Enqueue notification sending and structured interaction logging
        background_tasks.add_task(
            _send_notifications_and_log,
            email=payload.user_email,
            phone_number=payload.phone_number,
            query=payload.message,
            response=assistant_reply,
        )

        return ChatResponse(
            user=user,
            user_message=payload.message,
            assistant_reply=assistant_reply,
            created_at=assistant_message.created_at,
            source=source,
            provider=provider,
        )
