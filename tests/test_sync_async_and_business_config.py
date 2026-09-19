import asyncio
import json
import pytest
import tiktoken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.business_config import (
    BusinessConfig,
    get_business_config,
    reset_business_config,
    set_business_config,
)
from app.crud.appointment import appointment_crud
from app.crud.outbox import outbox_crud
from app.crud.user import user_crud
from app.models.appointment import Appointment
from app.models.outbox import OutboxEvent
from app.schemas.tools import (
    CheckAvailabilityRequest,
    CreateAppointmentRequest,
    FailureMode,
)
from app.services.outbox_service import OutboxWorker, outbox_worker
from app.services.tools_service import AppointmentToolService
from app.utils.prompts import (
    crm_intent_summary_prompt,
    receptionist_prompt,
    review_reply_prompt,
    voice_receptionist_prompt,
)


# ====================================================
# PHASE 11: SYNC VS ASYNC + TRANSACTIONAL OUTBOX TESTS
# ====================================================


def test_outbox_event_created_atomically_with_appointment(isolated_db):
    """Test 1: Appointment mutation and outbox event commit together atomically."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Outbox User", email="outbox@example.com")
            req = CreateAppointmentRequest(
                customer_name="Outbox User",
                customer_phone="03001234567",
                day="Monday",
                time_str="10 AM",
                service="Custom Software Development",
            )
            result = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)
            assert result.success is True
            app_id = result.data["appointment_id"]

            # Query outbox_events table
            stmt = select(OutboxEvent).where(OutboxEvent.event_type == "appointment.created")
            events = (await db.execute(stmt)).scalars().all()
            assert len(events) == 1
            event = events[0]
            assert event.status == "PENDING"
            payload = json.loads(event.payload)
            assert payload["appointment_id"] == app_id
            assert payload["customer_name"] == "Outbox User"

    asyncio.run(_run())


def test_outbox_event_rollback_safety(isolated_db, monkeypatch):
    """Test 2: Appointment rollback -> no outbox event is persisted in DB."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Rollback User", email="rollback@example.com")

            # Force commit failure
            async def failing_commit():
                raise RuntimeError("DB failure during transaction commit")

            monkeypatch.setattr(db, "commit", failing_commit)

            req = CreateAppointmentRequest(
                customer_name="Rollback User",
                customer_phone="03001234567",
                day="Monday",
                time_str="11 AM",
            )
            result = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)
            assert result.success is False

        # Clean session: ensure 0 outbox events exist
        async with session_maker() as db2:
            stmt = select(OutboxEvent)
            events = (await db2.execute(stmt)).scalars().all()
            assert len(events) == 0

    asyncio.run(_run())


def test_outbox_worker_processes_pending_event(isolated_db):
    """Test 3: OutboxWorker processes pending events and updates status to COMPLETED."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        dispatched_sms = []
        dispatched_crm = []

        async def mock_sms(payload):
            dispatched_sms.append(payload)

        async def mock_crm(payload):
            dispatched_crm.append(payload)

        worker = OutboxWorker()
        worker.set_handlers_for_testing("appointment.created", [mock_sms, mock_crm])

        async with session_maker() as db:
            user = await user_crud.create(db, name="Worker User", email="worker@example.com")
            req = CreateAppointmentRequest(
                customer_name="Worker User",
                customer_phone="03001234567",
                day="Tuesday",
                time_str="2 PM",
            )
            await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)

            # Process pending batch
            stats = await worker.process_batch(db)
            assert stats["total"] >= 1
            assert stats["succeeded"] >= 1
            assert stats["failed"] == 0

            # Verify mock handlers were invoked
            assert len(dispatched_sms) == 1
            assert dispatched_sms[0]["customer_phone"] == "03001234567"
            assert len(dispatched_crm) == 1

            # Verify outbox record status updated to COMPLETED in DB
            stmt = select(OutboxEvent).where(OutboxEvent.status == "COMPLETED")
            completed_events = (await db.execute(stmt)).scalars().all()
            assert len(completed_events) >= 1
            assert completed_events[0].processed_at is not None

    asyncio.run(_run())


def test_outbox_worker_failure_and_retry_tracking(isolated_db):
    """Test 4: Failed notification handler triggers retry tracking with incremented retry_count."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async def failing_handler(payload):
            raise ConnectionError("SMS Gateway 503 Service Unavailable")

        worker = OutboxWorker()
        worker.set_handlers_for_testing("appointment.created", [failing_handler])

        async with session_maker() as db:
            user = await user_crud.create(db, name="Retry User", email="retry@example.com")
            req = CreateAppointmentRequest(
                customer_name="Retry User",
                customer_phone="03001234567",
                day="Wednesday",
                time_str="3 PM",
            )
            await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)

            # Process batch with failing handler
            stats = await worker.process_batch(db)
            assert stats["failed"] >= 1

            # Verify event marked as FAILED with retry_count=1 and error_message stored
            stmt = select(OutboxEvent).where(OutboxEvent.status == "FAILED")
            failed_events = (await db.execute(stmt)).scalars().all()
            assert len(failed_events) >= 1
            failed_event = failed_events[0]
            assert failed_event.retry_count == 1
            assert "SMS Gateway 503" in failed_event.error_message

    asyncio.run(_run())


# ===============================================
# PHASE 12: PROMPT & TOKEN OPTIMIZATION TESTS
# ===============================================


def test_prompt_token_optimization_measurement():
    """Test prompt token measurement using tiktoken cl100k_base tokenizer."""
    enc = tiktoken.get_encoding("cl100k_base")

    rec_prompt = receptionist_prompt()
    voice_prompt = voice_receptionist_prompt()
    review_prompt = review_reply_prompt()
    crm_prompt = crm_intent_summary_prompt("Sample call transcript.")

    rec_tokens = len(enc.encode(rec_prompt))
    voice_tokens = len(enc.encode(voice_prompt))
    review_tokens = len(enc.encode(review_prompt))
    crm_tokens = len(enc.encode(crm_prompt))

    # Assert prompts are concise and token-efficient (well within optimized bounds)
    assert rec_tokens < 300, f"receptionist_prompt token count {rec_tokens} exceeds 300"
    assert voice_tokens < 200, f"voice_receptionist_prompt token count {voice_tokens} exceeds 200"
    assert review_tokens < 100, f"review_reply_prompt token count {review_tokens} exceeds 100"
    assert crm_tokens < 100, f"crm_intent_summary_prompt token count {crm_tokens} exceeds 100"


def test_prompt_behavioral_integrity():
    """Verify optimized prompts preserve critical business rules, constraints, and instructions."""
    rec = receptionist_prompt()
    assert "Logic Pulse" in rec
    assert "Muhammad Ramzan" in rec
    assert "Max 2 sentences" in rec
    assert "Never hallucinate" in rec
    assert "Monday, Tuesday, Wednesday, Thursday, Friday" in rec

    voice = voice_receptionist_prompt()
    assert "Logic Pulse" in voice
    assert "Muhammad Ramzan" in voice
    assert "Max 2 sentences" in voice


# ====================================================
# PHASE 13: DECOUPLED BUSINESS CONFIGURATION TESTS
# ====================================================


def test_business_config_default_logic_pulse():
    """Test default BusinessConfig accurately reflects Logic Pulse specifications."""
    reset_business_config()
    cfg = get_business_config()
    assert cfg.business_name == "Logic Pulse"
    assert cfg.ceo_name == "Muhammad Ramzan"
    assert cfg.working_hours_start == 9
    assert cfg.working_hours_end == 17
    assert "Friday" in cfg.working_days


def test_business_config_custom_business_validation_and_prompts(isolated_db):
    """Test switching to a completely different business config dynamically alters engine rules & prompts."""
    session_maker = isolated_db["session_maker"]

    # Configure a custom clinic: Apex Healthcare (Sun-Thu, 8 AM - 4 PM)
    custom_cfg = BusinessConfig(
        business_name="Apex Healthcare",
        ceo_name="Dr. Sarah Jenkins",
        assistant_name="Apex Concierge",
        working_days=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"],
        working_hours_start=8,
        working_hours_end=16,
        timezone="America/New_York",
        services=["General Practice", "Cardiology Consultation", "Pediatric Care"],
        contact_phone="+1 555 123 4567",
        contact_email="care@apexhealth.com",
    )
    set_business_config(custom_cfg)

    try:
        # Verify prompt dynamically uses custom config
        rec_prompt = receptionist_prompt()
        assert "Apex Healthcare" in rec_prompt
        assert "Dr. Sarah Jenkins" in rec_prompt
        assert "Sunday, Monday, Tuesday, Wednesday, Thursday" in rec_prompt
        assert "8 AM to 4 PM" in rec_prompt
        assert "Cardiology Consultation" in rec_prompt

        async def _run():
            async with session_maker() as db:
                user = await user_crud.create(db, name="Patient John", email="john@example.com")

                # 1. Friday is a weekend for Apex -> should be REJECTED
                req_friday = CreateAppointmentRequest(
                    customer_name="Patient John",
                    customer_phone="03001234567",
                    day="Friday",
                    time_str="10 AM",
                )
                res_fri = await AppointmentToolService.create_appointment(req_friday, user_id=user.id, db=db)
                assert res_fri.success is False
                assert res_fri.error_code == FailureMode.ERR_WEEKEND_RESTRICTION

                # 2. Sunday is a working day for Apex -> should be ACCEPTED
                req_sunday = CreateAppointmentRequest(
                    customer_name="Patient John",
                    customer_phone="03001234567",
                    day="Sunday",
                    time_str="9 AM",
                    service="Cardiology Consultation",
                )
                res_sun = await AppointmentToolService.create_appointment(req_sunday, user_id=user.id, db=db)
                assert res_sun.success is True
                assert res_sun.data["appointment_id"] > 0
                assert "Dr. Sarah Jenkins" in res_sun.user_facing_message
                assert "Apex Healthcare" in res_sun.user_facing_message

                # 3. 5 PM is outside Apex working hours (8 AM - 4 PM) -> should be REJECTED
                req_5pm = CheckAvailabilityRequest(day="Monday", time_str="5 PM")
                res_5pm = await AppointmentToolService.check_availability(req_5pm, db=db)
                assert res_5pm.success is False
                assert res_5pm.error_code == FailureMode.ERR_OUT_OF_HOURS

        asyncio.run(_run())

    finally:
        # Reset back to default Logic Pulse configuration
        reset_business_config()
