from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ToolAction(str, Enum):
    CHECK_AVAILABILITY = "check_availability"
    CREATE_APPOINTMENT = "create_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    RESCHEDULE_APPOINTMENT = "reschedule_appointment"


class FailureMode(str, Enum):
    ERR_WEEKEND_RESTRICTION = "ERR_WEEKEND_RESTRICTION"
    ERR_OUT_OF_HOURS = "ERR_OUT_OF_HOURS"
    ERR_INVALID_PHONE = "ERR_INVALID_PHONE"
    ERR_SLOT_CONFLICT = "ERR_SLOT_CONFLICT"
    ERR_APPOINTMENT_NOT_FOUND = "ERR_APPOINTMENT_NOT_FOUND"
    ERR_UNAUTHORIZED = "ERR_UNAUTHORIZED"
    ERR_DB_FAILURE = "ERR_DB_FAILURE"
    ERR_MISSING_ENTITIES = "ERR_MISSING_ENTITIES"
    ERR_UNKNOWN_TOOL = "ERR_UNKNOWN_TOOL"
    ERR_IDEMPOTENCY_CONFLICT = "ERR_IDEMPOTENCY_CONFLICT"


class CheckAvailabilityRequest(BaseModel):
    day: str = Field(..., description="Day of the week (e.g. Monday, Friday)")
    time_str: str = Field(..., description="Time string (e.g. 3 PM, 15:00)")
    service: str = Field(default="General Consultation", description="Requested service")
    tenant_id: str = Field(default="default", min_length=1, max_length=50, description="Tenant identifier")


class CreateAppointmentRequest(BaseModel):
    customer_name: str = Field(..., min_length=1, max_length=100, description="Customer full name")
    customer_phone: str = Field(..., description="11-digit Pakistani phone number (e.g. 03001234567)")
    day: str = Field(..., description="Day of the week (e.g. Friday)")
    time_str: str = Field(..., description="Time string (e.g. 3 PM)")
    service: str = Field(default="General Consultation", description="Requested service")
    customer_email: str | None = Field(default=None, description="Optional customer email")
    idempotency_key: str | None = Field(default=None, description="Optional idempotency key for durable deduplication")
    tenant_id: str = Field(default="default", min_length=1, max_length=50, description="Tenant identifier for multi-tenancy")


class CancelAppointmentRequest(BaseModel):
    appointment_id: int = Field(..., gt=0, description="ID of the appointment to cancel")
    caller_phone: str = Field(..., description="Phone number of the caller requesting cancellation for authorization")
    idempotency_key: str | None = Field(default=None, description="Optional idempotency key for durable deduplication")
    tenant_id: str = Field(default="default", min_length=1, max_length=50, description="Tenant identifier for multi-tenancy")


class RescheduleAppointmentRequest(BaseModel):
    appointment_id: int = Field(..., gt=0, description="ID of the appointment to reschedule")
    caller_phone: str = Field(..., description="Phone number of caller for authorization")
    new_day: str = Field(..., description="New day of the week")
    new_time_str: str = Field(..., description="New time string")
    idempotency_key: str | None = Field(default=None, description="Optional idempotency key for durable deduplication")
    tenant_id: str = Field(default="default", min_length=1, max_length=50, description="Tenant identifier for multi-tenancy")


class ToolExecutionResult(BaseModel):
    success: bool
    action: str
    status_code: int = 200
    data: dict[str, Any] | None = None
    error_code: str | None = None
    message: str
    user_facing_message: str
