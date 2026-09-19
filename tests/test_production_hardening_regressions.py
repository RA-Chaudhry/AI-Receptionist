"""Focused regressions for production hardening invariants."""

import asyncio
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
from app.crud.appointment import appointment_crud
from app.crud.outbox import outbox_crud
from app.crud.user import user_crud
from app.models.appointment import Appointment
from app.models.outbox import OutboxEvent
from app.schemas.tools import CancelAppointmentRequest, CheckAvailabilityRequest, CreateAppointmentRequest
from app.services.outbox_service import OutboxWorker
from app.services.receptionist_router import ReceptionistRouter
from app.services.tools_service import AppointmentToolService
from app.services.voice_service import _build_voice_identity


def test_tenant_scopes_appointments_and_slot_availability(isolated_db):
    """A tenant cannot mutate or reserve another tenant's appointment slot."""
    session_maker = isolated_db["session_maker"]

    async def run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Tenant Isolation", email="tenant-isolation@example.com")
            created = await AppointmentToolService.create_appointment(
                CreateAppointmentRequest(
                    customer_name="Tenant A Caller",
                    customer_phone="03001234567",
                    day="Monday",
                    time_str="10 AM",
                    tenant_id="tenant-a",
                ),
                user_id=user.id,
                db=db,
            )
            assert created.success

            other_tenant_slot = await AppointmentToolService.check_availability(
                CheckAvailabilityRequest(day="Monday", time_str="10 AM", tenant_id="tenant-b"), db
            )
            assert other_tenant_slot.success

            cross_tenant_cancel = await AppointmentToolService.cancel_appointment(
                CancelAppointmentRequest(
                    appointment_id=created.data["appointment_id"],
                    caller_phone="03001234567",
                    tenant_id="tenant-b",
                ),
                db,
            )
            assert not cross_tenant_cancel.success
            assert cross_tenant_cancel.status_code == 404

            tenant_a_appointment = await appointment_crud.get_by_id(
                db, created.data["appointment_id"], tenant_id="tenant-a"
            )
            assert tenant_a_appointment is not None
            assert tenant_a_appointment.status == "confirmed"

    asyncio.run(run())


def test_database_active_slot_index_prevents_racy_duplicate_rows(isolated_db):
    """The unique partial index, not only a read-then-write check, is the guard."""
    session_maker = isolated_db["session_maker"]

    async def run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Slot Guard", email="slot-guard@example.com")
            first = Appointment(
                tenant_id="tenant-a",
                user_id=user.id,
                customer_name="First",
                customer_phone="03001234567",
                service="General Consultation",
                day="Monday",
                time_str="10 AM",
                status="confirmed",
            )
            db.add(first)
            await db.commit()

        async with session_maker() as db:
            duplicate = Appointment(
                tenant_id="tenant-a",
                user_id=user.id,
                customer_name="Second",
                customer_phone="03009876543",
                service="General Consultation",
                day="Monday",
                time_str="10 AM",
                status="rescheduled",
            )
            db.add(duplicate)
            try:
                await db.commit()
                raised = False
            except Exception:
                raised = True
                await db.rollback()
            assert raised

        async with session_maker() as db:
            # A cancelled record intentionally releases the slot.
            first = (await db.execute(select(Appointment).where(Appointment.tenant_id == "tenant-a"))).scalar_one()
            first.status = "cancelled"
            await db.commit()
            replacement = Appointment(
                tenant_id="tenant-a",
                user_id=user.id,
                customer_name="Replacement",
                customer_phone="03001112222",
                service="General Consultation",
                day="Monday",
                time_str="10 AM",
                status="confirmed",
            )
            db.add(replacement)
            await db.commit()

    asyncio.run(run())


def test_router_never_claims_booking_or_sms_without_tool_result():
    history = [
        {"role": "user", "content": "Book Monday at 3 PM"},
        {
            "role": "assistant",
            "content": "I can request an appointment for Monday at 3 PM. Please confirm and share your phone number.",
        },
    ]
    reply = ReceptionistRouter.get_hardcoded_response("03001234567", history)
    lowered = reply.lower()
    assert "confirmed" not in lowered
    assert "sms has been sent" not in lowered
    assert "booking request" in lowered


def test_voice_identity_hashes_phone_and_isolates_anonymous_sessions():
    caller_name, caller_email = _build_voice_identity("03001234567")
    _, anonymous_one = _build_voice_identity(None, anonymous_session_id="call-one")
    _, anonymous_two = _build_voice_identity(None, anonymous_session_id="call-two")

    assert caller_name == "Caller"
    assert "03001234567" not in caller_email
    assert anonymous_one != anonymous_two


def test_outbox_backoff_and_claim_token_protect_worker_ownership(isolated_db):
    session_maker = isolated_db["session_maker"]

    async def run():
        async with session_maker() as db:
            event = await outbox_crud.create_event(db, "appointment.created", "{}")
            await db.commit()
            event_id = event.id

        async with session_maker() as db:
            claimed = await outbox_crud.claim_events(db)
            assert len(claimed) == 1
            stale_token = claimed[0].claim_token
            before_failure = datetime.now(timezone.utc)
            failed = await outbox_crud.mark_failed(
                db,
                event_id,
                "simulated timeout",
                claim_token=stale_token,
                retry_base_seconds=10,
                retry_max_seconds=60,
            )
            assert failed is not None
            assert failed.next_attempt_at is not None
            assert failed.next_attempt_at.replace(tzinfo=timezone.utc) > before_failure
            assert await outbox_crud.claim_events(db) == []

        async with session_maker() as db:
            event = await outbox_crud.create_event(db, "appointment.created", "{}")
            await db.commit()
            event_id = event.id
            claim = (await outbox_crud.claim_events(db))[0]
            assert await outbox_crud.mark_completed(db, event_id, claim_token="wrong-token") is None
            current = await outbox_crud.get_by_id(db, event_id)
            assert current.status == "PROCESSING"
            assert current.claim_token == claim.claim_token

    asyncio.run(run())


def test_crm_auth_fails_closed_in_production(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "admin_api_key", "operator-secret")

    assert client.get("/api/v1/users").status_code == 401
    assert client.get("/api/v1/users", headers={"X-API-Key": "operator-secret"}).status_code == 200
