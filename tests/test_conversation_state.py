import asyncio
import time
import pytest

from app.schemas.conversation_state import CollectedEntities, ConversationState, IntentType
from app.services.conversation_state_service import ConversationStateManager


def test_conversation_state_creation_and_touch():
    """Verify ConversationStateManager creates fresh state and touches TTL."""
    async def _run():
        manager = ConversationStateManager(default_ttl_seconds=1800.0)
        state = await manager.get_or_create_state("session_101", user_id=1, phone_number="03001234567")

        assert state.session_id == "session_101"
        assert state.user_id == 1
        assert state.phone_number == "03001234567"
        assert state.turn_count == 0
        assert not state.is_expired()

    asyncio.run(_run())


def test_multi_turn_piece_by_piece_entity_accumulation():
    """Verify entities accumulate incrementally over multiple conversational turns."""
    async def _run():
        manager = ConversationStateManager(default_ttl_seconds=1800.0)
        session_id = "call_turn_test_1"

        # Turn 1: Caller mentions day
        s1 = await manager.update_entities(session_id, {"day": "Friday"})
        assert s1.collected_entities.day == "Friday"
        assert s1.collected_entities.time_str is None
        assert s1.collected_entities.is_booking_ready() is False

        # Turn 2: Caller mentions time
        s2 = await manager.update_entities(session_id, {"time_str": "3 PM"})
        assert s2.collected_entities.day == "Friday"  # Preserved
        assert s2.collected_entities.time_str == "3 PM"
        assert s2.collected_entities.is_booking_ready() is False

        # Turn 3: Caller shares phone number
        s3 = await manager.update_entities(session_id, {"phone_number": "03001234567", "name": "Ahmad"})
        assert s3.collected_entities.day == "Friday"
        assert s3.collected_entities.time_str == "3 PM"
        assert s3.collected_entities.phone_number == "03001234567"
        assert s3.collected_entities.name == "Ahmad"
        assert s3.collected_entities.is_booking_ready() is True

    asyncio.run(_run())


def test_session_isolation():
    """Verify distinct sessions do not leak entities or context to each other."""
    async def _run():
        manager = ConversationStateManager()

        # Session A
        await manager.update_entities("session_A", {"name": "Caller A", "day": "Monday", "phone_number": "03001111111"})

        # Session B
        await manager.update_entities("session_B", {"name": "Caller B", "day": "Friday", "phone_number": "03002222222"})

        state_a = await manager.get_state("session_A")
        state_b = await manager.get_state("session_B")

        assert state_a is not None and state_b is not None
        assert state_a.collected_entities.name == "Caller A"
        assert state_a.collected_entities.day == "Monday"
        assert state_a.collected_entities.phone_number == "03001111111"

        assert state_b.collected_entities.name == "Caller B"
        assert state_b.collected_entities.day == "Friday"
        assert state_b.collected_entities.phone_number == "03002222222"

    asyncio.run(_run())


def test_session_expiration_and_cleanup():
    """Verify expired sessions are recognized and cleaned up."""
    async def _run():
        manager = ConversationStateManager(default_ttl_seconds=1.0)
        state = await manager.get_or_create_state("session_exp")

        # Manually expire state
        state.expires_at = time.time() - 10.0
        assert state.is_expired() is True

        # Cleanup expired sessions
        cleared = await manager.cleanup_expired()
        assert cleared == 1
        assert await manager.get_state("session_exp") is None

    asyncio.run(_run())


def test_recover_state_from_messages():
    """Verify deterministic state recovery from previous SQLite chat messages."""
    manager = ConversationStateManager()
    history = [
        {"role": "user", "content": "Hi, I want to book an appointment with Muhammad Ramzan."},
        {"role": "assistant", "content": "I would be happy to help. What day and time works for you?"},
        {"role": "user", "content": "How about Friday at 4 PM?"},
        {"role": "assistant", "content": "Friday at 4 PM is open. Please share your phone number to confirm."},
        {"role": "user", "content": "Sure, my number is 03001234567."},
    ]

    recovered = manager.recover_state_from_messages("recovered_session", history, user_id=42)
    assert recovered.session_id == "recovered_session"
    assert recovered.user_id == 42
    assert recovered.current_intent == IntentType.BOOKING
    assert recovered.collected_entities.day == "Friday"
    assert recovered.collected_entities.time_str == "4 PM"
    assert recovered.collected_entities.phone_number == "03001234567"
    assert recovered.collected_entities.is_booking_ready() is True


def test_intent_switching_and_pending_tools():
    """Verify intent transitions and pending tool action tracking."""
    async def _run():
        manager = ConversationStateManager()
        session_id = "switch_intent_session"

        await manager.set_intent(session_id, IntentType.INQUIRING)
        s1 = await manager.get_state(session_id)
        assert s1 is not None
        assert s1.current_intent == IntentType.INQUIRING

        # Switch to booking
        await manager.set_intent(session_id, IntentType.BOOKING)
        await manager.set_pending_tool(session_id, "create_appointment")
        s2 = await manager.get_state(session_id)
        assert s2 is not None
        assert s2.current_intent == IntentType.BOOKING
        assert s2.pending_tool_action == "create_appointment"

    asyncio.run(_run())
