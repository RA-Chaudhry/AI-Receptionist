"""
tests/test_phone_and_entity_hardening.py
Comprehensive regression tests for Phase 13.5: Voice Conversation Reliability & Entity Hardening.

Verifies:
1. Spoken digit normalization (English words, Roman Urdu, Urdu words, multipliers like double/triple).
2. Spoken phone sequences ("0, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8") never trigger 12 AM midnight outside working hours.
3. State isolation: phone utterances never mutate day or time entities.
4. Malformed noise rejection ("0-345-9543-A-Double-Pipe") without LLM hallucinations.
5. Deterministic Monday 3 PM acceptance (working day + working hour).
6. Entity context locking when awaiting phone number.
7. Enterprise phone number confirmation formatting (0345 678 8888).
"""

import pytest
from app.utils.entity_normalizer import (
    normalize_spoken_digits,
    extract_phone_number,
    format_phone_display,
    extract_day,
    validate_day,
    extract_time_hour,
    validate_working_hour,
    format_hour_display,
)
from app.services.receptionist_router import ReceptionistRouter
from app.services.conversation_state_service import ConversationStateManager
from app.schemas.conversation_state import IntentType, CollectedEntities


# ──────────────────────────────────────────────────────────────────────────────
# 1. Spoken Digits & Multiplier Normalization
# ──────────────────────────────────────────────────────────────────────────────

class TestSpokenDigitsNormalization:
    def test_comma_separated_digits(self):
        raw = "0, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8"
        norm = normalize_spoken_digits(raw)
        is_p, p_val, reason = extract_phone_number(raw)
        assert is_p is True
        assert p_val == "03456788888"
        assert reason == "valid"

    def test_english_words_digits(self):
        raw = "zero three four five six seven eight eight eight eight eight"
        is_p, p_val, reason = extract_phone_number(raw)
        assert is_p is True
        assert p_val == "03456788888"

    def test_roman_urdu_words_digits(self):
        raw = "sifar teen char panch chhe saat aath aath aath aath aath"
        is_p, p_val, reason = extract_phone_number(raw)
        assert is_p is True
        assert p_val == "03456788888"

    def test_urdu_script_digits(self):
        raw = "۰۳۴۵۶۷۸۸۸۸۸"
        is_p, p_val, reason = extract_phone_number(raw)
        assert is_p is True
        assert p_val == "03456788888"

    def test_double_and_triple_multipliers(self):
        raw = "zero three four five nine five four three double five one"
        is_p, p_val, reason = extract_phone_number(raw)
        assert is_p is True
        assert p_val == "03459543551"
        assert reason == "valid"

        raw11 = "0 3 4 5 9 5 4 double 3 double 5"
        is_p11, p_val11, _ = extract_phone_number(raw11)
        assert is_p11 is True
        assert p_val11 == "03459543355"

    def test_phone_formatting(self):
        assert format_phone_display("03456788888") == "0345 678 8888"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Golden Bug Regression: "0, 3, 4, 5..." NEVER triggers 12 AM
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeExtractionGuards:
    def test_spoken_phone_digits_not_parsed_as_hour_zero(self):
        raw = "0, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8"
        hour = extract_time_hour(raw)
        assert hour is None, "Long digit phone sequence must NEVER be parsed as time hour"

    def test_phone_with_prefix_not_parsed_as_time(self):
        raw = "0345-9543123"
        hour = extract_time_hour(raw)
        assert hour is None

    def test_explicit_times_parsed_correctly(self):
        assert extract_time_hour("3 PM") == 15
        assert extract_time_hour("Monday at 3 p.m.") == 15
        assert extract_time_hour("9 AM") == 9
        assert extract_time_hour("3 baje") == 15
        assert extract_time_hour("teen baje") == 15
        assert extract_time_hour("12 PM") == 12
        assert extract_time_hour("12 AM") == 0


# ──────────────────────────────────────────────────────────────────────────────
# 3. Malformed Noise Rejection ("0-345-9543-A-Double-Pipe")
# ──────────────────────────────────────────────────────────────────────────────

class TestMalformedNoiseRejection:
    def test_double_pipe_rejected(self):
        raw = "0-345-9543-A-Double-Pipe"
        is_p, p_val, reason = extract_phone_number(raw)
        assert is_p is False
        assert reason == "malformed_noise"

    def test_router_handles_malformed_noise_cleanly(self):
        history = [
            {"role": "user", "content": "I am free Monday at 3 PM"},
            {"role": "assistant", "content": "Your appointment with our CEO, Muhammad Ramzan, is scheduled for Monday at 3 PM. Please confirm, and also share your phone number to receive the details via SMS."}
        ]
        r = ReceptionistRouter.get_hardcoded_response("0-345-9543-A-Double-Pipe", history=history)
        assert r is not None
        assert "couldn't recognize" in r.lower() or "valid phone number" in r.lower() or "03" in r


# ──────────────────────────────────────────────────────────────────────────────
# 4. Deterministic Date / Time Validation (Monday at 3 PM)
# ──────────────────────────────────────────────────────────────────────────────

class TestDeterministicWorkingHours:
    def test_monday_is_valid_working_day(self):
        day = extract_day("Monday at 3 p.m.")
        assert day == "Monday"
        is_valid, reason = validate_day(day)
        assert is_valid is True
        assert reason == "working_day"

    def test_3pm_is_valid_working_hour(self):
        hour = extract_time_hour("Monday at 3 p.m.")
        assert hour == 15
        is_valid, reason = validate_working_hour(hour)
        assert is_valid is True
        assert reason == "working_hour"

    def test_monday_3pm_router_response(self):
        r = ReceptionistRouter.get_hardcoded_response("Monday at 3 p.m.")
        assert r is not None
        assert "outside our working hours" not in r
        assert "Muhammad Ramzan" in r
        assert "Monday" in r
        assert "3 PM" in r


# ──────────────────────────────────────────────────────────────────────────────
# 5. Entity Context Locking & State Machine Flow
# ──────────────────────────────────────────────────────────────────────────────

class TestEntityLockingAndStateFlow:
    def test_phone_utterance_does_not_mutate_day_or_time(self):
        history = [
            {"role": "user", "content": "Monday at 3 PM please"},
            {"role": "assistant", "content": "Your appointment with our CEO, Muhammad Ramzan, is scheduled for Monday at 3 PM. Please confirm, and also share your phone number to receive the details via SMS."}
        ]
        user_utterance = "0, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8"
        r = ReceptionistRouter.get_hardcoded_response(user_utterance, history=history)

        assert r is not None
        # Must NOT complain about 12 AM midnight or working hours!
        assert "12 AM" not in r
        assert "outside our working hours" not in r
        # Must recognize appointment for Monday at 3 PM with formatted phone
        assert "Monday" in r
        assert "3 PM" in r
        assert "0345 678 8888" in r or "03456788888" in r

    def test_state_recovery_from_messages(self):
        mgr = ConversationStateManager()
        history = [
            {"role": "user", "content": "I want to schedule an appointment with the CEO on Monday at 3 PM"},
            {"role": "assistant", "content": "Your appointment with our CEO, Muhammad Ramzan, is scheduled for Monday at 3 PM. Could you please share your phone number?"},
            {"role": "user", "content": "0, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8"}
        ]
        recovered_state = mgr.recover_state_from_messages("test_session_1", history)
        assert recovered_state.current_intent == IntentType.BOOKING
        assert recovered_state.collected_entities.day == "Monday"
        assert recovered_state.collected_entities.time_str == "3 PM"
        assert recovered_state.collected_entities.phone_number == "03456788888"
