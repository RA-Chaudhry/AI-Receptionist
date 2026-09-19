import asyncio
from unittest.mock import AsyncMock, patch
import pytest

from app.crud.user import user_crud
from app.schemas.tools import (
    CancelAppointmentRequest,
    CheckAvailabilityRequest,
    CreateAppointmentRequest,
    FailureMode,
    RescheduleAppointmentRequest,
    ToolAction,
)
from app.services.tools_service import AppointmentToolService, FAILURE_MESSAGES
from app.utils.prompts import format_tool_grounding_prompt


def test_tool_check_availability_success(isolated_db):
    """Verify check_availability succeeds for valid weekday and hours."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            req = CheckAvailabilityRequest(day="Friday", time_str="3 PM", service="Software Consultation")
            result = await AppointmentToolService.check_availability(req, db)

            assert result.success is True
            assert result.status_code == 200
            assert result.data is not None
            assert result.data["available"] is True
            assert "available" in result.user_facing_message.lower()

    asyncio.run(_run())


def test_tool_create_appointment_success(isolated_db):
    """Verify create_appointment successfully writes to DB and returns confirmation."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Usman Khan", email="usman@example.com")

            req = CreateAppointmentRequest(
                customer_name="Usman Khan",
                customer_phone="03001234567",
                customer_email="usman@example.com",
                day="Friday",
                time_str="3 PM",
                service="AI Automation",
            )
            result = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)

            assert result.success is True
            assert result.status_code == 201
            assert result.data is not None
            assert result.data["status"] == "confirmed"
            assert "confirmed" in result.user_facing_message.lower()
            assert "03001234567" in result.user_facing_message

    asyncio.run(_run())


def test_tool_double_booking_slot_conflict(isolated_db):
    """Verify double-booking attempt on same day and time is rejected with ERR_SLOT_CONFLICT."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="User A", email="usera@example.com")

            # 1. First booking succeeds
            req1 = CreateAppointmentRequest(
                customer_name="User A",
                customer_phone="03001111111",
                day="Monday",
                time_str="10 AM",
            )
            res1 = await AppointmentToolService.create_appointment(req1, user_id=user.id, db=db)
            assert res1.success is True

            # 2. Second booking on exact same slot fails
            req2 = CreateAppointmentRequest(
                customer_name="User B",
                customer_phone="03002222222",
                day="Monday",
                time_str="10 AM",
            )
            res2 = await AppointmentToolService.create_appointment(req2, user_id=user.id, db=db)
            assert res2.success is False
            assert res2.status_code == 409
            assert res2.error_code == FailureMode.ERR_SLOT_CONFLICT
            assert "already booked" in res2.user_facing_message.lower()

    asyncio.run(_run())


def test_tool_weekend_rejection_failure_mode(isolated_db):
    """Verify weekend appointments (Saturday, Sunday) are rejected with ERR_WEEKEND_RESTRICTION."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="User Sat", email="sat@example.com")

            # Saturday
            req_sat = CreateAppointmentRequest(
                customer_name="User Sat",
                customer_phone="03001234567",
                day="Saturday",
                time_str="2 PM",
            )
            res_sat = await AppointmentToolService.create_appointment(req_sat, user_id=user.id, db=db)
            assert res_sat.success is False
            assert res_sat.error_code == FailureMode.ERR_WEEKEND_RESTRICTION
            assert "weekend" in res_sat.user_facing_message.lower()

            # Sunday
            req_sun = CheckAvailabilityRequest(day="Sunday", time_str="11 AM")
            res_sun = await AppointmentToolService.check_availability(req_sun, db)
            assert res_sun.success is False
            assert res_sun.error_code == FailureMode.ERR_WEEKEND_RESTRICTION

    asyncio.run(_run())


def test_tool_out_of_hours_failure_mode(isolated_db):
    """Verify booking outside 9 AM–5 PM is rejected with ERR_OUT_OF_HOURS."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Night Owl", email="night@example.com")

            # 12 AM Midnight
            req_mid = CreateAppointmentRequest(
                customer_name="Night Owl",
                customer_phone="03001234567",
                day="Tuesday",
                time_str="12 AM",
            )
            res_mid = await AppointmentToolService.create_appointment(req_mid, user_id=user.id, db=db)
            assert res_mid.success is False
            assert res_mid.error_code == FailureMode.ERR_OUT_OF_HOURS
            assert "working hours" in res_mid.user_facing_message.lower()

            # 7 PM Evening
            req_eve = CheckAvailabilityRequest(day="Wednesday", time_str="7 PM")
            res_eve = await AppointmentToolService.check_availability(req_eve, db)
            assert res_eve.success is False
            assert res_eve.error_code == FailureMode.ERR_OUT_OF_HOURS

    asyncio.run(_run())


def test_tool_invalid_pakistani_phone_failure_mode(isolated_db):
    """Verify non-Pakistani or short phone numbers are rejected with ERR_INVALID_PHONE."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Bad Phone", email="phone@example.com")

            req = CreateAppointmentRequest(
                customer_name="Bad Phone",
                customer_phone="12345",
                day="Thursday",
                time_str="11 AM",
            )
            res = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)
            assert res.success is False
            assert res.error_code == FailureMode.ERR_INVALID_PHONE
            assert "11-digit pakistani phone number" in res.user_facing_message.lower()

    asyncio.run(_run())


def test_tool_cancel_appointment_authorized_and_unauthorized(isolated_db):
    """Verify cancellation requires phone authorization and updates status."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Cancel User", email="cancel@example.com")

            # Create appointment
            create_req = CreateAppointmentRequest(
                customer_name="Cancel User",
                customer_phone="03001234567",
                day="Friday",
                time_str="11 AM",
            )
            created = await AppointmentToolService.create_appointment(create_req, user_id=user.id, db=db)
            appt_id = created.data["appointment_id"]

            # 1. Unauthorized attempt (different phone) -> 403 ERR_UNAUTHORIZED
            unauth_req = CancelAppointmentRequest(appointment_id=appt_id, caller_phone="03119999999")
            unauth_res = await AppointmentToolService.cancel_appointment(unauth_req, db)
            assert unauth_res.success is False
            assert unauth_res.status_code == 403
            assert unauth_res.error_code == FailureMode.ERR_UNAUTHORIZED

            # 2. Authorized attempt (matching phone) -> 200 Success
            auth_req = CancelAppointmentRequest(appointment_id=appt_id, caller_phone="03001234567")
            auth_res = await AppointmentToolService.cancel_appointment(auth_req, db)
            assert auth_res.success is True
            assert auth_res.status_code == 200
            assert "cancelled" in auth_res.user_facing_message.lower()

    asyncio.run(_run())


def test_tool_reschedule_appointment(isolated_db):
    """Verify rescheduling checks authorization, slot availability, and updates DB."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Resched User", email="resched@example.com")

            create_req = CreateAppointmentRequest(
                customer_name="Resched User",
                customer_phone="03007778899",
                day="Tuesday",
                time_str="2 PM",
            )
            created = await AppointmentToolService.create_appointment(create_req, user_id=user.id, db=db)
            appt_id = created.data["appointment_id"]

            resched_req = RescheduleAppointmentRequest(
                appointment_id=appt_id,
                caller_phone="03007778899",
                new_day="Thursday",
                new_time_str="4 PM",
            )
            res = await AppointmentToolService.reschedule_appointment(resched_req, db)
            assert res.success is True
            assert res.status_code == 200
            assert res.data["new_day"] == "Thursday"
            assert res.data["new_time_str"] == "4 PM"

    asyncio.run(_run())


def test_tool_db_write_failure_mode(isolated_db):
    """Verify database write exceptions degrade gracefully to ERR_DB_FAILURE."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="DB Error User", email="dberr@example.com")

            req = CreateAppointmentRequest(
                customer_name="DB Error User",
                customer_phone="03001234567",
                day="Friday",
                time_str="3 PM",
            )

            # Force database commit to fail
            with patch.object(db, "commit", side_effect=RuntimeError("Simulated DB connection failure")):
                res = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)
                assert res.success is False
                assert res.status_code == 500
                assert res.error_code == FailureMode.ERR_DB_FAILURE
                assert "temporary system issue" in res.user_facing_message.lower()

    asyncio.run(_run())


def test_tool_result_grounding_prompt_generation():
    """Verify format_tool_grounding_prompt properly formats success and failure injection."""
    # 1. Success Grounding
    success_dict = {
        "success": True,
        "action": "create_appointment",
        "user_facing_message": "Your appointment is confirmed for Friday at 3 PM.",
    }
    prompt_success = format_tool_grounding_prompt(success_dict)
    assert "[BACKEND TOOL RESULT - SUCCESS]" in prompt_success
    assert "Your appointment is confirmed for Friday at 3 PM." in prompt_success

    # 2. Failure Grounding
    failure_dict = {
        "success": False,
        "action": "create_appointment",
        "error_code": "ERR_SLOT_CONFLICT",
        "user_facing_message": "I apologize, but that time slot is already booked.",
    }
    prompt_fail = format_tool_grounding_prompt(failure_dict)
    assert "[BACKEND TOOL RESULT - FAILED]" in prompt_fail
    assert "ERR_SLOT_CONFLICT" in prompt_fail
    assert "Do NOT claim the action succeeded" in prompt_fail
