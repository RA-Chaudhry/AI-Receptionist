import hashlib
import json
import logging
import re
from typing import Any
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.business_config import BusinessConfig, get_business_config
from app.crud.appointment import appointment_crud
from app.crud.idempotency import idempotency_crud
from app.crud.outbox import outbox_crud
from app.models.idempotency import IdempotencyRecord
from app.models.outbox import OutboxEvent
from app.schemas.tools import (
    CancelAppointmentRequest,
    CheckAvailabilityRequest,
    CreateAppointmentRequest,
    FailureMode,
    RescheduleAppointmentRequest,
    ToolAction,
    ToolExecutionResult,
)

logger = logging.getLogger(__name__)


def compute_request_hash(payload: dict[str, Any]) -> str:
    """
    Deterministically computes a SHA256 hex digest for a request payload.
    Excludes non-payload metadata like idempotency_key or tenant_id to hash the core business intent.
    """
    norm_dict = {
        k: (v.strip().lower() if isinstance(v, str) else v)
        for k, v in sorted(payload.items())
        if k not in {"idempotency_key", "tenant_id"} and v is not None
    }
    raw_str = json.dumps(norm_dict, sort_keys=True)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


def get_failure_messages(config: BusinessConfig | None = None) -> dict[FailureMode, str]:
    cfg = config or get_business_config()
    days_str = ", ".join(cfg.working_days)
    start_fmt = f"{cfg.working_hours_start % 12 or 12} {'AM' if cfg.working_hours_start < 12 else 'PM'}"
    end_fmt = f"{cfg.working_hours_end % 12 or 12} {'AM' if cfg.working_hours_end < 12 else 'PM'}"

    return {
        FailureMode.ERR_WEEKEND_RESTRICTION: (
            f"We are closed on weekends (Saturday and Sunday). "
            f"Our office operates on {days_str}, {start_fmt} to {end_fmt}."
        ),
        FailureMode.ERR_OUT_OF_HOURS: (
            f"That time is outside our working hours. "
            f"We are available between {start_fmt} and {end_fmt}, {days_str}."
        ),
        FailureMode.ERR_INVALID_PHONE: (
            "Could you please provide a valid 11-digit Pakistani phone number (e.g., 03001234567)?"
        ),
        FailureMode.ERR_SLOT_CONFLICT: (
            "I apologize, but that time slot is already booked. Would another time work for you?"
        ),
        FailureMode.ERR_APPOINTMENT_NOT_FOUND: (
            "I couldn't find an active appointment matching those details. "
            "Could you please double-check your appointment ID?"
        ),
        FailureMode.ERR_UNAUTHORIZED: (
            "For security and privacy, appointments can only be cancelled or rescheduled "
            "from the phone number used to book them."
        ),
        FailureMode.ERR_DB_FAILURE: (
            "We are experiencing a temporary system issue processing your appointment request. "
            "Please hold while I connect you to a representative."
        ),
        FailureMode.ERR_MISSING_ENTITIES: (
            "I would be happy to help with your appointment. Could you please specify your preferred day and time?"
        ),
        FailureMode.ERR_UNKNOWN_TOOL: (
            "I am unable to perform that specific operation. Let me know if you would like to book an appointment."
        ),
        FailureMode.ERR_IDEMPOTENCY_CONFLICT: (
            "This request conflicts with an existing operation with the same idempotency key."
        ),
    }


FAILURE_MESSAGES = get_failure_messages()
WORKING_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday"}
WEEKEND_DAYS = {"saturday", "sunday"}



def validate_working_day(day_str: str, config: BusinessConfig | None = None) -> FailureMode | None:
    cfg = config or get_business_config()
    canonical = day_str.strip().lower()
    valid_days = {d.strip().lower() for d in cfg.working_days}
    if canonical not in valid_days:
        return FailureMode.ERR_WEEKEND_RESTRICTION
    return None


def validate_working_hours(time_str: str, config: BusinessConfig | None = None) -> FailureMode | None:
    cfg = config or get_business_config()
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", time_str, re.IGNORECASE)
    if not match:
        return FailureMode.ERR_OUT_OF_HOURS

    hour = int(match.group(1))
    meridiem = (match.group(3) or "").lower().replace(".", "")

    if "pm" in meridiem and hour != 12:
        hour_24 = hour + 12
    elif "am" in meridiem and hour == 12:
        hour_24 = 0
    elif "am" in meridiem:
        hour_24 = hour
    else:
        # Infer reasonable office hours if no meridiem given (e.g. 3 -> 15:00, 10 -> 10:00)
        hour_24 = hour if hour >= cfg.working_hours_start and hour <= 12 else hour + 12

    if hour_24 < cfg.working_hours_start or hour_24 > cfg.working_hours_end:
        return FailureMode.ERR_OUT_OF_HOURS
    return None


def validate_pakistani_phone(phone_str: str) -> FailureMode | None:
    cleaned = re.sub(r"[\s\-()]", "", phone_str)
    # Valid Pakistani formats: 03001234567 (11 digits), +923001234567, 923001234567
    if re.match(r"^(?:03\d{9}|\+923\d{9}|923\d{9})$", cleaned):
        return None
    return FailureMode.ERR_INVALID_PHONE


class AppointmentToolService:
    """
    Deterministic backend tool execution service for business operations.
    Enforces business rules, authorises actions, guarantees database-backed idempotency,
    and commits transactional outbox events atomically.
    """

    @staticmethod
    async def check_availability(
        req: CheckAvailabilityRequest,
        db: AsyncSession,
        config: BusinessConfig | None = None,
    ) -> ToolExecutionResult:
        cfg = config or get_business_config()
        failure_msgs = get_failure_messages(cfg)

        # 1. Day validation
        day_err = validate_working_day(req.day, config=cfg)
        if day_err:
            return ToolExecutionResult(
                success=False,
                action=ToolAction.CHECK_AVAILABILITY,
                status_code=400,
                error_code=day_err,
                message=f"Day {req.day} is not a valid working day for {cfg.business_name}.",
                user_facing_message=failure_msgs[day_err],
            )

        # 2. Hours validation
        hours_err = validate_working_hours(req.time_str, config=cfg)
        if hours_err:
            return ToolExecutionResult(
                success=False,
                action=ToolAction.CHECK_AVAILABILITY,
                status_code=400,
                error_code=hours_err,
                message=f"Time {req.time_str} is outside working hours ({cfg.working_hours_start}:00 - {cfg.working_hours_end}:00).",
                user_facing_message=failure_msgs[hours_err],
            )

        # 3. Database slot conflict check
        try:
            conflict = await appointment_crud.get_confirmed_by_slot(
                db, day=req.day, time_str=req.time_str, tenant_id=req.tenant_id
            )
            if conflict:
                return ToolExecutionResult(
                    success=False,
                    action=ToolAction.CHECK_AVAILABILITY,
                    status_code=409,
                    error_code=FailureMode.ERR_SLOT_CONFLICT,
                    message=f"Slot on {req.day} at {req.time_str} is already booked.",
                    user_facing_message=failure_msgs[FailureMode.ERR_SLOT_CONFLICT],
                )

            return ToolExecutionResult(
                success=True,
                action=ToolAction.CHECK_AVAILABILITY,
                status_code=200,
                data={"available": True, "day": req.day.capitalize(), "time_str": req.time_str.upper()},
                message=f"Slot on {req.day} at {req.time_str} is available.",
                user_facing_message=f"Good news! The {req.day.capitalize()} at {req.time_str.upper()} slot is available. Would you like me to book it for you?",
            )
        except Exception:
            logger.exception("Database error checking availability")
            return ToolExecutionResult(
                success=False,
                action=ToolAction.CHECK_AVAILABILITY,
                status_code=500,
                error_code=FailureMode.ERR_DB_FAILURE,
                message="Database query failed while checking availability.",
                user_facing_message=failure_msgs[FailureMode.ERR_DB_FAILURE],
            )

    @staticmethod
    async def create_appointment(
        req: CreateAppointmentRequest,
        user_id: int,
        db: AsyncSession,
        config: BusinessConfig | None = None,
    ) -> ToolExecutionResult:
        cfg = config or get_business_config()
        failure_msgs = get_failure_messages(cfg)
        tenant_id = req.tenant_id
        req_hash = compute_request_hash(req.model_dump())

        # Check DB-backed idempotency record before performing mutation
        if req.idempotency_key:
            existing_record = await idempotency_crud.get_record(
                db,
                tenant_id=tenant_id,
                key=req.idempotency_key,
                operation=ToolAction.CREATE_APPOINTMENT,
            )
            if existing_record:
                if existing_record.request_hash == req_hash:
                    if existing_record.status == "COMPLETED" and existing_record.result_data:
                        cached_dict = json.loads(existing_record.result_data)
                        return ToolExecutionResult(**cached_dict)
                else:
                    return ToolExecutionResult(
                        success=False,
                        action=ToolAction.CREATE_APPOINTMENT,
                        status_code=409,
                        error_code=FailureMode.ERR_IDEMPOTENCY_CONFLICT,
                        message=f"Idempotency conflict: key '{req.idempotency_key}' was used with different payload.",
                        user_facing_message=failure_msgs[FailureMode.ERR_IDEMPOTENCY_CONFLICT],
                    )

        # 1. Day validation
        day_err = validate_working_day(req.day, config=cfg)
        if day_err:
            return ToolExecutionResult(
                success=False,
                action=ToolAction.CREATE_APPOINTMENT,
                status_code=400,
                error_code=day_err,
                message=f"Cannot book appointment outside working days ({req.day}).",
                user_facing_message=failure_msgs[day_err],
            )

        # 2. Hours validation
        hours_err = validate_working_hours(req.time_str, config=cfg)
        if hours_err:
            return ToolExecutionResult(
                success=False,
                action=ToolAction.CREATE_APPOINTMENT,
                status_code=400,
                error_code=hours_err,
                message=f"Cannot book appointment outside working hours ({req.time_str}).",
                user_facing_message=failure_msgs[hours_err],
            )

        # 3. Phone validation
        phone_err = validate_pakistani_phone(req.customer_phone)
        if phone_err:
            return ToolExecutionResult(
                success=False,
                action=ToolAction.CREATE_APPOINTMENT,
                status_code=400,
                error_code=phone_err,
                message=f"Invalid Pakistani phone number: {req.customer_phone}",
                user_facing_message=failure_msgs[phone_err],
            )

        # 4. Conflict check & database creation with atomic outbox event & idempotency record
        try:
            conflict = await appointment_crud.get_confirmed_by_slot(
                db, day=req.day, time_str=req.time_str, tenant_id=tenant_id
            )
            if conflict:
                return ToolExecutionResult(
                    success=False,
                    action=ToolAction.CREATE_APPOINTMENT,
                    status_code=409,
                    error_code=FailureMode.ERR_SLOT_CONFLICT,
                    message=f"Double booking prevented: {req.day} at {req.time_str} is already occupied.",
                    user_facing_message=failure_msgs[FailureMode.ERR_SLOT_CONFLICT],
                )

            # Create appointment entity (staged in session without intermediate commit)
            appointment = await appointment_crud.create(db, req, user_id=user_id, commit=False)

            exec_result = ToolExecutionResult(
                success=True,
                action=ToolAction.CREATE_APPOINTMENT,
                status_code=201,
                data={
                    "appointment_id": appointment.id,
                    "customer_name": appointment.customer_name,
                    "customer_phone": appointment.customer_phone,
                    "day": appointment.day,
                    "time_str": appointment.time_str,
                    "service": appointment.service,
                    "status": appointment.status,
                },
                message=f"Appointment #{appointment.id} created successfully.",
                user_facing_message=(
                    f"Your appointment with {cfg.ceo_name} at {cfg.business_name} has been confirmed for "
                    f"{appointment.day} at {appointment.time_str}. A confirmation SMS has been sent to {appointment.customer_phone}."
                ),
            )

            # Atomic Transactional Outbox Event
            outbox_payload = json.dumps({
                "appointment_id": appointment.id,
                "customer_name": appointment.customer_name,
                "customer_phone": appointment.customer_phone,
                "customer_email": appointment.customer_email,
                "service": appointment.service,
                "day": appointment.day,
                "time_str": appointment.time_str,
                "business_name": cfg.business_name,
            })
            await outbox_crud.create_event(
                db,
                event_type="appointment.created",
                payload=outbox_payload,
            )

            # Durable DB Idempotency Record
            if req.idempotency_key:
                await idempotency_crud.create_record(
                    db,
                    tenant_id=tenant_id,
                    key=req.idempotency_key,
                    operation=ToolAction.CREATE_APPOINTMENT,
                    request_hash=req_hash,
                    status="COMPLETED",
                    result_data=exec_result.model_dump_json(),
                )

            await db.commit()
            return exec_result

        except IntegrityError:
            await db.rollback()
            logger.warning("IntegrityError during create_appointment; checking for idempotency race")
            if req.idempotency_key:
                existing = await idempotency_crud.get_record(
                    db,
                    tenant_id=tenant_id,
                    key=req.idempotency_key,
                    operation=ToolAction.CREATE_APPOINTMENT,
                )
                if existing:
                    if existing.request_hash == req_hash and existing.result_data:
                        return ToolExecutionResult(**json.loads(existing.result_data))
                    return ToolExecutionResult(
                        success=False,
                        action=ToolAction.CREATE_APPOINTMENT,
                        status_code=409,
                        error_code=FailureMode.ERR_IDEMPOTENCY_CONFLICT,
                        message="Concurrent idempotency conflict.",
                        user_facing_message=failure_msgs[FailureMode.ERR_IDEMPOTENCY_CONFLICT],
                    )

            return ToolExecutionResult(
                success=False,
                action=ToolAction.CREATE_APPOINTMENT,
                status_code=500,
                error_code=FailureMode.ERR_DB_FAILURE,
                message="Integrity error writing appointment to database.",
                user_facing_message=failure_msgs[FailureMode.ERR_DB_FAILURE],
            )
        except Exception:
            await db.rollback()
            logger.exception("Database error creating appointment")
            return ToolExecutionResult(
                success=False,
                action=ToolAction.CREATE_APPOINTMENT,
                status_code=500,
                error_code=FailureMode.ERR_DB_FAILURE,
                message="Failed to write appointment to database.",
                user_facing_message=failure_msgs[FailureMode.ERR_DB_FAILURE],
            )

    @staticmethod
    async def cancel_appointment(
        req: CancelAppointmentRequest,
        db: AsyncSession,
        config: BusinessConfig | None = None,
    ) -> ToolExecutionResult:
        cfg = config or get_business_config()
        failure_msgs = get_failure_messages(cfg)
        tenant_id = req.tenant_id
        req_hash = compute_request_hash(req.model_dump())

        if req.idempotency_key:
            existing_record = await idempotency_crud.get_record(
                db,
                tenant_id=tenant_id,
                key=req.idempotency_key,
                operation=ToolAction.CANCEL_APPOINTMENT,
            )
            if existing_record:
                if existing_record.request_hash == req_hash:
                    if existing_record.status == "COMPLETED" and existing_record.result_data:
                        return ToolExecutionResult(**json.loads(existing_record.result_data))
                else:
                    return ToolExecutionResult(
                        success=False,
                        action=ToolAction.CANCEL_APPOINTMENT,
                        status_code=409,
                        error_code=FailureMode.ERR_IDEMPOTENCY_CONFLICT,
                        message=f"Idempotency conflict: key '{req.idempotency_key}' was used with different payload.",
                        user_facing_message=failure_msgs[FailureMode.ERR_IDEMPOTENCY_CONFLICT],
                    )

        try:
            appointment = await appointment_crud.get_by_id(
                db, req.appointment_id, tenant_id=tenant_id
            )
            if not appointment or appointment.status == "cancelled":
                return ToolExecutionResult(
                    success=False,
                    action=ToolAction.CANCEL_APPOINTMENT,
                    status_code=404,
                    error_code=FailureMode.ERR_APPOINTMENT_NOT_FOUND,
                    message=f"Appointment #{req.appointment_id} not found or already cancelled.",
                    user_facing_message=failure_msgs[FailureMode.ERR_APPOINTMENT_NOT_FOUND],
                )

            # Authorization Check: caller phone must match booking phone
            caller_norm = re.sub(r"[\s\-()]", "", req.caller_phone)
            booked_norm = re.sub(r"[\s\-()]", "", appointment.customer_phone)
            if not caller_norm.endswith(booked_norm[-9:]):
                return ToolExecutionResult(
                    success=False,
                    action=ToolAction.CANCEL_APPOINTMENT,
                    status_code=403,
                    error_code=FailureMode.ERR_UNAUTHORIZED,
                    message="Caller phone does not match appointment record.",
                    user_facing_message=failure_msgs[FailureMode.ERR_UNAUTHORIZED],
                )

            updated = await appointment_crud.update_status(
                db,
                req.appointment_id,
                status="cancelled",
                tenant_id=tenant_id,
                commit=False,
            )
            exec_result = ToolExecutionResult(
                success=True,
                action=ToolAction.CANCEL_APPOINTMENT,
                status_code=200,
                data={"appointment_id": req.appointment_id, "status": "cancelled"},
                message=f"Appointment #{req.appointment_id} cancelled successfully.",
                user_facing_message=f"Your appointment on {appointment.day} at {appointment.time_str} has been successfully cancelled.",
            )

            # Outbox event
            await outbox_crud.create_event(
                db,
                event_type="appointment.cancelled",
                payload=json.dumps({"appointment_id": req.appointment_id, "caller_phone": req.caller_phone}),
            )

            # Idempotency record
            if req.idempotency_key:
                await idempotency_crud.create_record(
                    db,
                    tenant_id=tenant_id,
                    key=req.idempotency_key,
                    operation=ToolAction.CANCEL_APPOINTMENT,
                    request_hash=req_hash,
                    status="COMPLETED",
                    result_data=exec_result.model_dump_json(),
                )

            await db.commit()
            return exec_result
        except IntegrityError:
            await db.rollback()
            if req.idempotency_key:
                existing = await idempotency_crud.get_record(
                    db,
                    tenant_id=tenant_id,
                    key=req.idempotency_key,
                    operation=ToolAction.CANCEL_APPOINTMENT,
                )
                if existing and existing.result_data:
                    return ToolExecutionResult(**json.loads(existing.result_data))
            return ToolExecutionResult(
                success=False,
                action=ToolAction.CANCEL_APPOINTMENT,
                status_code=500,
                error_code=FailureMode.ERR_DB_FAILURE,
                message="Database error cancelling appointment.",
                user_facing_message=failure_msgs[FailureMode.ERR_DB_FAILURE],
            )
        except Exception:
            await db.rollback()
            logger.exception("Database error cancelling appointment")
            return ToolExecutionResult(
                success=False,
                action=ToolAction.CANCEL_APPOINTMENT,
                status_code=500,
                error_code=FailureMode.ERR_DB_FAILURE,
                message="Database update failed while cancelling appointment.",
                user_facing_message=failure_msgs[FailureMode.ERR_DB_FAILURE],
            )

    @staticmethod
    async def reschedule_appointment(
        req: RescheduleAppointmentRequest,
        db: AsyncSession,
        config: BusinessConfig | None = None,
    ) -> ToolExecutionResult:
        cfg = config or get_business_config()
        failure_msgs = get_failure_messages(cfg)
        tenant_id = req.tenant_id
        req_hash = compute_request_hash(req.model_dump())

        if req.idempotency_key:
            existing_record = await idempotency_crud.get_record(
                db,
                tenant_id=tenant_id,
                key=req.idempotency_key,
                operation=ToolAction.RESCHEDULE_APPOINTMENT,
            )
            if existing_record:
                if existing_record.request_hash == req_hash:
                    if existing_record.status == "COMPLETED" and existing_record.result_data:
                        return ToolExecutionResult(**json.loads(existing_record.result_data))
                else:
                    return ToolExecutionResult(
                        success=False,
                        action=ToolAction.RESCHEDULE_APPOINTMENT,
                        status_code=409,
                        error_code=FailureMode.ERR_IDEMPOTENCY_CONFLICT,
                        message=f"Idempotency conflict: key '{req.idempotency_key}' was used with different payload.",
                        user_facing_message=failure_msgs[FailureMode.ERR_IDEMPOTENCY_CONFLICT],
                    )

        # 1. Day validation
        day_err = validate_working_day(req.new_day, config=cfg)
        if day_err:
            return ToolExecutionResult(
                success=False,
                action=ToolAction.RESCHEDULE_APPOINTMENT,
                status_code=400,
                error_code=day_err,
                message=f"Cannot reschedule to non-working day ({req.new_day}).",
                user_facing_message=failure_msgs[day_err],
            )

        # 2. Hours validation
        hours_err = validate_working_hours(req.new_time_str, config=cfg)
        if hours_err:
            return ToolExecutionResult(
                success=False,
                action=ToolAction.RESCHEDULE_APPOINTMENT,
                status_code=400,
                error_code=hours_err,
                message=f"Cannot reschedule outside working hours ({req.new_time_str}).",
                user_facing_message=failure_msgs[hours_err],
            )

        try:
            appointment = await appointment_crud.get_by_id(
                db, req.appointment_id, tenant_id=tenant_id
            )
            if not appointment or appointment.status == "cancelled":
                return ToolExecutionResult(
                    success=False,
                    action=ToolAction.RESCHEDULE_APPOINTMENT,
                    status_code=404,
                    error_code=FailureMode.ERR_APPOINTMENT_NOT_FOUND,
                    message=f"Appointment #{req.appointment_id} not found or is cancelled.",
                    user_facing_message=failure_msgs[FailureMode.ERR_APPOINTMENT_NOT_FOUND],
                )

            # Authorization Check
            caller_norm = re.sub(r"[\s\-()]", "", req.caller_phone)
            booked_norm = re.sub(r"[\s\-()]", "", appointment.customer_phone)
            if not caller_norm.endswith(booked_norm[-9:]):
                return ToolExecutionResult(
                    success=False,
                    action=ToolAction.RESCHEDULE_APPOINTMENT,
                    status_code=403,
                    error_code=FailureMode.ERR_UNAUTHORIZED,
                    message="Caller phone does not match appointment record.",
                    user_facing_message=failure_msgs[FailureMode.ERR_UNAUTHORIZED],
                )

            # Slot conflict check for new slot
            conflict = await appointment_crud.get_confirmed_by_slot(
                db,
                day=req.new_day,
                time_str=req.new_time_str,
                tenant_id=tenant_id,
            )
            if conflict and conflict.id != appointment.id:
                return ToolExecutionResult(
                    success=False,
                    action=ToolAction.RESCHEDULE_APPOINTMENT,
                    status_code=409,
                    error_code=FailureMode.ERR_SLOT_CONFLICT,
                    message=f"Slot {req.new_day} at {req.new_time_str} is already booked.",
                    user_facing_message=failure_msgs[FailureMode.ERR_SLOT_CONFLICT],
                )

            rescheduled = await appointment_crud.reschedule(
                db,
                appointment_id=req.appointment_id,
                new_day=req.new_day,
                new_time_str=req.new_time_str,
                tenant_id=tenant_id,
                commit=False,
            )
            exec_result = ToolExecutionResult(
                success=True,
                action=ToolAction.RESCHEDULE_APPOINTMENT,
                status_code=200,
                data={
                    "appointment_id": req.appointment_id,
                    "new_day": rescheduled.day if rescheduled else req.new_day.capitalize(),
                    "new_time_str": rescheduled.time_str if rescheduled else req.new_time_str.upper(),
                    "status": "rescheduled",
                },
                message=f"Appointment #{req.appointment_id} rescheduled successfully.",
                user_facing_message=f"Your appointment has been successfully rescheduled to {req.new_day.capitalize()} at {req.new_time_str.upper()}.",
            )

            # Outbox event
            await outbox_crud.create_event(
                db,
                event_type="appointment.rescheduled",
                payload=json.dumps({
                    "appointment_id": req.appointment_id,
                    "new_day": req.new_day,
                    "new_time_str": req.new_time_str,
                }),
            )

            # Idempotency record
            if req.idempotency_key:
                await idempotency_crud.create_record(
                    db,
                    tenant_id=tenant_id,
                    key=req.idempotency_key,
                    operation=ToolAction.RESCHEDULE_APPOINTMENT,
                    request_hash=req_hash,
                    status="COMPLETED",
                    result_data=exec_result.model_dump_json(),
                )

            await db.commit()
            return exec_result
        except IntegrityError:
            await db.rollback()
            if req.idempotency_key:
                existing = await idempotency_crud.get_record(
                    db,
                    tenant_id=tenant_id,
                    key=req.idempotency_key,
                    operation=ToolAction.RESCHEDULE_APPOINTMENT,
                )
                if existing and existing.result_data:
                    return ToolExecutionResult(**json.loads(existing.result_data))
            return ToolExecutionResult(
                success=False,
                action=ToolAction.RESCHEDULE_APPOINTMENT,
                status_code=500,
                error_code=FailureMode.ERR_DB_FAILURE,
                message="Database error rescheduling appointment.",
                user_facing_message=failure_msgs[FailureMode.ERR_DB_FAILURE],
            )
        except Exception:
            await db.rollback()
            logger.exception("Database error rescheduling appointment")
            return ToolExecutionResult(
                success=False,
                action=ToolAction.RESCHEDULE_APPOINTMENT,
                status_code=500,
                error_code=FailureMode.ERR_DB_FAILURE,
                message="Database update failed while rescheduling appointment.",
                user_facing_message=failure_msgs[FailureMode.ERR_DB_FAILURE],
            )
