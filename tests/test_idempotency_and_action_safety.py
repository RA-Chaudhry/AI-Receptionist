import asyncio
import json
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.appointment import appointment_crud
from app.crud.idempotency import idempotency_crud
from app.crud.user import user_crud
from app.models.appointment import Appointment
from app.models.idempotency import IdempotencyRecord
from app.schemas.tools import (
    CancelAppointmentRequest,
    CreateAppointmentRequest,
    FailureMode,
    RescheduleAppointmentRequest,
    ToolAction,
    ToolExecutionResult,
)
from app.services.tools_service import AppointmentToolService, compute_request_hash
from app.utils.prompts import format_tool_grounding_prompt


# ==========================================
# PHASE 9: DATABASE-BACKED IDEMPOTENCY TESTS
# ==========================================


def test_idempotency_first_create_creates_mutation(isolated_db):
    """Test 1: First create with idempotency key -> actual database mutation occurs."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Zara Khan", email="zara@example.com")
            req = CreateAppointmentRequest(
                customer_name="Zara Khan",
                customer_phone="03001234567",
                day="Monday",
                time_str="10 AM",
                service="AI Solutions",
                idempotency_key="idemp-key-001",
                tenant_id="tenant-alpha",
            )
            result = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)
            assert result.success is True
            assert result.status_code == 201
            assert result.data is not None
            app_id = result.data["appointment_id"]
            assert app_id > 0

            # Verify persisted idempotency record in DB
            record = await idempotency_crud.get_record(
                db,
                tenant_id="tenant-alpha",
                key="idemp-key-001",
                operation=ToolAction.CREATE_APPOINTMENT,
            )
            assert record is not None
            assert record.status == "COMPLETED"
            assert record.request_hash == compute_request_hash(req.model_dump())
            assert record.result_data is not None

    asyncio.run(_run())


def test_idempotency_exact_retry_returns_cached_result(isolated_db):
    """Test 2: Exact retry with same key & payload -> no second mutation, returns cached result."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Bilal Ahmed", email="bilal@example.com")
            req = CreateAppointmentRequest(
                customer_name="Bilal Ahmed",
                customer_phone="03001234567",
                day="Tuesday",
                time_str="11 AM",
                idempotency_key="idemp-key-002",
                tenant_id="default",
            )
            # Call 1: First mutation
            res1 = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)
            assert res1.success is True
            app_id1 = res1.data["appointment_id"]

            # Call 2: Exact retry
            res2 = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)
            assert res2.success is True
            assert res2.data["appointment_id"] == app_id1
            assert res2.message == res1.message

            # Verify only 1 appointment row exists in DB
            stmt = select(Appointment).where(Appointment.customer_name == "Bilal Ahmed")
            rows = (await db.execute(stmt)).scalars().all()
            assert len(rows) == 1

    asyncio.run(_run())


def test_idempotency_conflict_on_different_payload(isolated_db):
    """Test 3: Same key + different payload -> deterministic ERR_IDEMPOTENCY_CONFLICT."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Sara Ali", email="sara@example.com")
            req1 = CreateAppointmentRequest(
                customer_name="Sara Ali",
                customer_phone="03001234567",
                day="Wednesday",
                time_str="2 PM",
                idempotency_key="idemp-key-003",
            )
            res1 = await AppointmentToolService.create_appointment(req1, user_id=user.id, db=db)
            assert res1.success is True

            # Call 2: Same key, different time payload
            req2 = CreateAppointmentRequest(
                customer_name="Sara Ali",
                customer_phone="03001234567",
                day="Wednesday",
                time_str="4 PM",
                idempotency_key="idemp-key-003",
            )
            res2 = await AppointmentToolService.create_appointment(req2, user_id=user.id, db=db)
            assert res2.success is False
            assert res2.status_code == 409
            assert res2.error_code == FailureMode.ERR_IDEMPOTENCY_CONFLICT

    asyncio.run(_run())


def test_idempotency_concurrent_duplicate_requests(isolated_db):
    """Test 4: Concurrent duplicate requests -> exactly ONE actual mutation created."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Hamza Khan", email="hamza@example.com")

        # Simulate two concurrent tasks executing against the DB
        async def call_create():
            async with session_maker() as db:
                req = CreateAppointmentRequest(
                    customer_name="Hamza Khan",
                    customer_phone="03001234567",
                    day="Thursday",
                    time_str="3 PM",
                    idempotency_key="idemp-key-004-concurrent",
                )
                return await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)

        # Run both tasks concurrently
        results = await asyncio.gather(call_create(), call_create(), return_exceptions=False)
        assert all(r.success for r in results)

        # Both return the exact same appointment ID
        app_id_1 = results[0].data["appointment_id"]
        app_id_2 = results[1].data["appointment_id"]
        assert app_id_1 == app_id_2

        # Verify only 1 appointment row was persisted in DB
        async with session_maker() as db:
            stmt = select(Appointment).where(Appointment.customer_name == "Hamza Khan")
            rows = (await db.execute(stmt)).scalars().all()
            assert len(rows) == 1

    asyncio.run(_run())


def test_idempotency_rollback_leaves_no_successful_record(isolated_db, monkeypatch):
    """Test 5: DB failure/rollback -> no falsely successful idempotency result left behind."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Faulty User", email="faulty@example.com")

            # Force database commit to fail
            async def failing_commit():
                raise RuntimeError("Forced DB connection failure during commit")

            monkeypatch.setattr(db, "commit", failing_commit)

            req = CreateAppointmentRequest(
                customer_name="Faulty User",
                customer_phone="03001234567",
                day="Friday",
                time_str="10 AM",
                idempotency_key="idemp-key-rollback-005",
            )
            result = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)
            assert result.success is False
            assert result.error_code == FailureMode.ERR_DB_FAILURE

        # Verify no successful idempotency record in clean session
        async with session_maker() as db:
            record = await idempotency_crud.get_record(
                db,
                tenant_id="default",
                key="idemp-key-rollback-005",
                operation=ToolAction.CREATE_APPOINTMENT,
            )
            assert record is None or record.status != "COMPLETED"

    asyncio.run(_run())


def test_idempotency_process_restart_simulation(isolated_db):
    """Test 6: Simulated process restart -> persisted DB idempotency state works authoritatively."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        # Session 1: Create appointment
        async with session_maker() as db1:
            user = await user_crud.create(db1, name="Restart User", email="restart@example.com")
            req = CreateAppointmentRequest(
                customer_name="Restart User",
                customer_phone="03001234567",
                day="Monday",
                time_str="2 PM",
                idempotency_key="idemp-key-restart-006",
            )
            res1 = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db1)
            assert res1.success is True
            orig_app_id = res1.data["appointment_id"]

        # Session 2 (simulating fresh process / new worker):
        async with session_maker() as db2:
            res2 = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db2)
            assert res2.success is True
            assert res2.data["appointment_id"] == orig_app_id

    asyncio.run(_run())


def test_idempotency_different_tenants_same_key(isolated_db):
    """Test 7: Different tenants using same key -> independent operations succeed."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Tenant User", email="tenant@example.com")

            req_tenant_a = CreateAppointmentRequest(
                customer_name="Tenant A User",
                customer_phone="03001234567",
                day="Tuesday",
                time_str="10 AM",
                idempotency_key="shared-key-100",
                tenant_id="tenant-A",
            )
            res_a = await AppointmentToolService.create_appointment(req_tenant_a, user_id=user.id, db=db)
            assert res_a.success is True

            req_tenant_b = CreateAppointmentRequest(
                customer_name="Tenant B User",
                customer_phone="03009876543",
                day="Tuesday",
                time_str="2 PM",
                idempotency_key="shared-key-100",
                tenant_id="tenant-B",
            )
            res_b = await AppointmentToolService.create_appointment(req_tenant_b, user_id=user.id, db=db)
            assert res_b.success is True
            assert res_a.data["appointment_id"] != res_b.data["appointment_id"]

    asyncio.run(_run())


def test_idempotency_different_operations_same_key(isolated_db):
    """Test 8: Different operations using same key -> independent operations succeed."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Multi Op User", email="multiop@example.com")

            # Create operation
            req_create = CreateAppointmentRequest(
                customer_name="Multi Op User",
                customer_phone="03001234567",
                day="Wednesday",
                time_str="11 AM",
                idempotency_key="op-key-200",
            )
            res_create = await AppointmentToolService.create_appointment(req_create, user_id=user.id, db=db)
            assert res_create.success is True
            app_id = res_create.data["appointment_id"]

            # Cancel operation with same idempotency key string
            req_cancel = CancelAppointmentRequest(
                appointment_id=app_id,
                caller_phone="03001234567",
                idempotency_key="op-key-200",
            )
            res_cancel = await AppointmentToolService.cancel_appointment(req_cancel, db=db)
            assert res_cancel.success is True
            assert res_cancel.data["status"] == "cancelled"

    asyncio.run(_run())


# ===============================================
# PHASE 10: ACTION SAFETY / ANTI-HALLUCINATION
# ===============================================


def test_action_safety_backend_truth_grounding():
    """Test action safety prompt grounding ensures assistant only confirms upon backend truth."""
    # 1. Successful committed result
    success_result = {
        "success": True,
        "action": "create_appointment",
        "user_facing_message": "Your appointment with Muhammad Ramzan has been confirmed for Friday at 3 PM.",
    }
    prompt_success = format_tool_grounding_prompt(success_result)
    assert "[BACKEND TOOL RESULT - SUCCESS]" in prompt_success
    assert "Confirm the details accurately" in prompt_success
    assert "Muhammad Ramzan" in prompt_success

    # 2. Failed result (e.g. slot conflict or weekend rejection)
    failed_result = {
        "success": False,
        "action": "create_appointment",
        "error_code": "ERR_SLOT_CONFLICT",
        "user_facing_message": "That slot is already booked. Would another time work?",
    }
    prompt_failed = format_tool_grounding_prompt(failed_result)
    assert "[BACKEND TOOL RESULT - FAILED]" in prompt_failed
    assert "Do NOT claim the action succeeded" in prompt_failed
    assert "ERR_SLOT_CONFLICT" in prompt_failed


def test_action_safety_failure_modes_do_not_produce_success(isolated_db):
    """Test adversarial failure conditions prevent false success confirmations."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Safety User", email="safety@example.com")

            # A. Weekend rejection -> success is False
            req_weekend = CreateAppointmentRequest(
                customer_name="Safety User",
                customer_phone="03001234567",
                day="Sunday",
                time_str="3 PM",
            )
            res_weekend = await AppointmentToolService.create_appointment(req_weekend, user_id=user.id, db=db)
            assert res_weekend.success is False
            assert res_weekend.data is None

            # B. Out of hours -> success is False
            req_ooh = CreateAppointmentRequest(
                customer_name="Safety User",
                customer_phone="03001234567",
                day="Monday",
                time_str="8 PM",
            )
            res_ooh = await AppointmentToolService.create_appointment(req_ooh, user_id=user.id, db=db)
            assert res_ooh.success is False
            assert res_ooh.data is None

            # C. Unauthorized cancellation -> success is False
            req_create = CreateAppointmentRequest(
                customer_name="Safety User",
                customer_phone="03001234567",
                day="Friday",
                time_str="11 AM",
            )
            res_create = await AppointmentToolService.create_appointment(req_create, user_id=user.id, db=db)
            assert res_create.success is True
            app_id = res_create.data["appointment_id"]

            req_unauth_cancel = CancelAppointmentRequest(
                appointment_id=app_id,
                caller_phone="03339999999",  # Wrong phone
            )
            res_cancel = await AppointmentToolService.cancel_appointment(req_unauth_cancel, db=db)
            assert res_cancel.success is False
            assert res_cancel.error_code == FailureMode.ERR_UNAUTHORIZED

    asyncio.run(_run())
