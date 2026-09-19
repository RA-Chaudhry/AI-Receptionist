import asyncio
import logging
import re
import time
from typing import Any

from app.schemas.conversation_state import CollectedEntities, ConversationState, IntentType
from app.utils.entity_normalizer import (
    extract_day,
    extract_time_hour,
    extract_phone_number,
    format_hour_display,
)

logger = logging.getLogger(__name__)


class ConversationStateManager:
    """
    Session-isolated conversation state manager.
    Tracks collected entities, active intents, appointment contexts, and tool triggers.
    Supports in-memory caching with TTL expiration, thread-safety, and SQLite history recovery.
    """

    def __init__(self, default_ttl_seconds: float = 1800.0) -> None:
        self.default_ttl = default_ttl_seconds
        self._store: dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_state(
        self,
        session_id: str,
        user_id: int | None = None,
        phone_number: str | None = None,
    ) -> ConversationState:
        async with self._lock:
            state = self._store.get(session_id)
            if state is not None:
                if state.is_expired():
                    logger.debug("Session %s expired; creating fresh state.", session_id)
                    del self._store[session_id]
                    state = None
                else:
                    state.touch(self.default_ttl)
                    if user_id is not None:
                        state.user_id = user_id
                    if phone_number is not None:
                        state.phone_number = phone_number
                        state.collected_entities.phone_number = phone_number
                    return state

            new_state = ConversationState(
                session_id=session_id,
                user_id=user_id,
                phone_number=phone_number,
                collected_entities=CollectedEntities(phone_number=phone_number),
            )
            self._store[session_id] = new_state
            return new_state

    async def get_state(self, session_id: str) -> ConversationState | None:
        async with self._lock:
            state = self._store.get(session_id)
            if state is not None and state.is_expired():
                del self._store[session_id]
                return None
            return state

    async def update_entities(
        self,
        session_id: str,
        entities: CollectedEntities | dict[str, Any],
    ) -> ConversationState:
        async with self._lock:
            state = self._store.get(session_id)
            if state is None:
                state = ConversationState(session_id=session_id)
                self._store[session_id] = state

            if isinstance(entities, dict):
                entities = CollectedEntities(**entities)

            state.collected_entities.merge(entities)
            state.touch(self.default_ttl)
            return state

    async def set_intent(self, session_id: str, intent: IntentType) -> ConversationState:
        async with self._lock:
            state = self._store.get(session_id)
            if state is None:
                state = ConversationState(session_id=session_id)
                self._store[session_id] = state
            state.current_intent = intent
            state.touch(self.default_ttl)
            return state

    async def set_pending_tool(
        self,
        session_id: str,
        tool_action: str | None,
    ) -> ConversationState:
        async with self._lock:
            state = self._store.get(session_id)
            if state is None:
                state = ConversationState(session_id=session_id)
                self._store[session_id] = state
            state.pending_tool_action = tool_action
            state.touch(self.default_ttl)
            return state

    async def clear_state(self, session_id: str) -> None:
        async with self._lock:
            self._store.pop(session_id, None)

    async def cleanup_expired(self) -> int:
        async with self._lock:
            now = time.time()
            expired_keys = [k for k, v in self._store.items() if v.is_expired(now)]
            for k in expired_keys:
                del self._store[k]
            return len(expired_keys)

    def recover_state_from_messages(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        user_id: int | None = None,
        phone_number: str | None = None,
    ) -> ConversationState:
        """
        Deterministic recovery of collected entities and intent from prior message history.
        Used when in-memory cache has been evicted or across worker reloads.
        """
        entities = CollectedEntities(phone_number=phone_number)
        intent = IntentType.GENERAL
        awaiting_entity = None

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            if role == "user":
                # Check for booking keywords
                if any(w in content.lower() for w in ["appointment", "schedule", "book", "meeting"]):
                    intent = IntentType.BOOKING
                elif any(w in content.lower() for w in ["cancel", "cancellation"]):
                    intent = IntentType.CANCELLING
                elif any(w in content.lower() for w in ["reschedule", "change time", "move"]):
                    intent = IntentType.RESCHEDULING

                # Extract phone if present
                is_phone, phone_val, _ = extract_phone_number(content)
                if is_phone and phone_val:
                    entities.phone_number = phone_val
                else:
                    # Only extract day/time if not phone number
                    day_val = extract_day(content)
                    if day_val:
                        entities.day = day_val

                    hour_val = extract_time_hour(content)
                    if hour_val is not None:
                        entities.time_str = format_hour_display(hour_val)

            elif role == "assistant":
                content_lower = content.lower()
                if "phone number" in content_lower or "share your number" in content_lower:
                    awaiting_entity = "phone"
                elif "what day" in content_lower or "which day" in content_lower:
                    awaiting_entity = "day"
                elif "what time" in content_lower or "which time" in content_lower:
                    awaiting_entity = "time"
                elif "shall i confirm" in content_lower or "confirm this booking" in content_lower:
                    awaiting_entity = "confirmation"

        state = ConversationState(
            session_id=session_id,
            user_id=user_id,
            phone_number=entities.phone_number or phone_number,
            current_intent=intent,
            awaiting_entity=awaiting_entity,
            collected_entities=entities,
            turn_count=len(messages),
        )
        self._store[session_id] = state
        return state


# Global state manager singleton with per-session isolation
conversation_state_manager = ConversationStateManager()
