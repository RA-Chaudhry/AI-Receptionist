from enum import Enum
import time
from typing import Any
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    GENERAL = "general"
    INQUIRING = "inquiring"
    BOOKING = "booking"
    CANCELLING = "cancelling"
    RESCHEDULING = "rescheduling"


class CollectedEntities(BaseModel):
    name: str | None = None
    day: str | None = None
    time_str: str | None = None
    date_str: str | None = None
    service: str | None = None
    phone_number: str | None = None
    email: str | None = None
    appointment_id: int | None = None

    def merge(self, other: "CollectedEntities") -> None:
        """Merges non-None fields from another CollectedEntities instance."""
        for field, value in other.model_dump().items():
            if value is not None:
                setattr(self, field, value)

    def is_booking_ready(self) -> bool:
        """Returns True if day, time, and phone number are all captured."""
        return bool(self.day and self.time_str and self.phone_number)


class ConversationState(BaseModel):
    session_id: str
    user_id: int | None = None
    phone_number: str | None = None
    current_intent: IntentType = IntentType.GENERAL
    awaiting_entity: str | None = None  # "day", "time", "phone", "confirmation", or None
    collected_entities: CollectedEntities = Field(default_factory=CollectedEntities)
    appointment_context: dict[str, Any] = Field(default_factory=dict)
    pending_tool_action: str | None = None
    last_assistant_action: str | None = None
    turn_count: int = 0
    clarification_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    expires_at: float = Field(default_factory=lambda: time.time() + 1800.0)

    def is_expired(self, current_time: float | None = None) -> bool:
        now = current_time if current_time is not None else time.time()
        return now >= self.expires_at

    def touch(self, ttl_seconds: float = 1800.0) -> None:
        now = time.time()
        self.updated_at = now
        self.expires_at = now + ttl_seconds
        self.turn_count += 1
