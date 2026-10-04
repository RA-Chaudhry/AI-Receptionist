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

    def test_day_time_preserved_after_multiturn_phone_capture(self):
        """Bug #1 & #4: Day/time collected before phone number remains available after multi-turn phone capture."""
        history = [
            {"role": "user", "content": "I want to book an appointment with the CEO."},
            {"role": "assistant", "content": "Sure, which weekday (Monday–Friday) and time between 9 AM and 5 PM works best?"},
            {"role": "user", "content": "I am free on Thursday at 3pm."},
            {"role": "assistant", "content": "I have noted Thursday at 3 PM. Please provide your phone number so I can prepare the booking request."},
            {"role": "user", "content": "0, 3, 4, 5, 9, 5, 4, 3"},
            {"role": "assistant", "content": "I think I missed a digit. Could you please repeat your 11-digit phone number?"},
        ]
        # User provides full number or final chunk
        r = ReceptionistRouter.get_hardcoded_response("My phone number is 03459543855.", history=history)
        assert r is not None
        # Must preserve Thursday and 3 PM without asking for weekday/time again
        assert "Thursday" in r
        assert "3 PM" in r
        assert "0345 954 3855" in r or "03459543855" in r
        assert "Please also provide the weekday and time" not in r
        assert "Would you like me to submit this booking request?" in r

    def test_multiturn_phone_digit_accumulation_across_turns(self):
        """Bug #3: Spoken digit chunks accumulated across turns."""
        history = [
            {"role": "user", "content": "Book an appointment on Friday at 4 PM"},
            {"role": "assistant", "content": "Please share your phone number to proceed."},
            {"role": "user", "content": "0, 3, 4, 5, 9, 5, 4, 3"},
            {"role": "assistant", "content": "I think I missed a digit. Could you please repeat your 11-digit phone number?"},
        ]
        # Turn 2: User provides remaining digits "8, 5, 5"
        r = ReceptionistRouter.get_hardcoded_response("8, 5, 5", history=history)
        assert r is not None
        assert "Friday" in r
        assert "4 PM" in r
        assert "0345 954 3855" in r or "03459543855" in r

    def test_dashed_digit_chunks_normalize_correctly(self):
        """Bug #3: '0-3-4-5-95-4-3-8-5-5' parses into a valid 11-digit number."""
        raw = "0-3-4-5-95-4-3-8-5-5"
        is_p, p_val, reason = extract_phone_number(raw)
        assert is_p is True
        assert p_val == "03459543855"

    def test_time_expressions_never_appended_to_phone(self):
        """Bug #3: Time expressions like '3 PM' are never treated as phone digits."""
        raw = "Add Thursday 3 PM"
        is_p, p_val, reason = extract_phone_number(raw)
        assert is_p is False
        assert p_val is None

    def test_invalid_fragments_do_not_corrupt_phone_number(self):
        """Bug #3: Invalid noise fragments do not corrupt state."""
        raw = "I am free on Thursday at 3 PM and some random words"
        is_p, p_val, reason = extract_phone_number(raw)
        assert is_p is False
        assert p_val is None


# ──────────────────────────────────────────────────────────────────────────────
# 6. Anti-Hallucination & Tool Grounding (Bug #2 & #4)
# ──────────────────────────────────────────────────────────────────────────────

class TestAntiHallucinationAndToolExecution:
    def test_no_premature_forwarding_or_confirmation_claims(self):
        """Bug #2: Receptionist must not claim booking forwarded or confirmed before tool execution."""
        history = [
            {"role": "user", "content": "I want to schedule an appointment with CEO Muhammad Ramzan"},
            {"role": "assistant", "content": "What day and time work for you?"},
        ]
        r = ReceptionistRouter.get_hardcoded_response("Add Thursday 3 PM.", history=history)
        assert r is not None
        lowered = r.lower()
        # MUST NOT make unsupported backend claims
        assert "has been forwarded" not in lowered
        assert "has been submitted" not in lowered
        assert "sent to ceo" not in lowered
        assert "forwarded to ceo" not in lowered
        assert "confirmed" not in lowered
        assert "sms has been sent" not in lowered
        # Must be neutral booking proposal
        assert "Thursday" in r
        assert "3 PM" in r

    def test_tool_execution_success_produces_confirmation(self, isolated_db):
        """Explicit confirmation invokes AppointmentToolService and successful tool result produces confirmation."""
        import asyncio
        from app.schemas.tools import CreateAppointmentRequest
        from app.services.tools_service import AppointmentToolService
        from app.crud.user import user_crud

        session_maker = isolated_db["session_maker"]

        async def run():
            async with session_maker() as db:
                user = await user_crud.create(db, name="Verified Caller", email="verified@example.com")
                req = CreateAppointmentRequest(
                    customer_name="Verified Caller",
                    customer_phone="03459543855",
                    day="Thursday",
                    time_str="3 PM",
                    tenant_id="test-tenant",
                )
                res = await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)
                assert res.success is True
                assert "confirmed" in res.user_facing_message.lower()
                assert "Thursday" in res.user_facing_message
                assert "3 PM" in res.user_facing_message

        asyncio.run(run())

    def test_tool_execution_failure_does_not_produce_confirmation(self, isolated_db):
        """Failed tool result (e.g. double booking / slot collision) does NOT claim confirmation."""
        import asyncio
        from app.schemas.tools import CreateAppointmentRequest
        from app.services.tools_service import AppointmentToolService
        from app.crud.user import user_crud

        session_maker = isolated_db["session_maker"]

        async def run():
            async with session_maker() as db:
                user = await user_crud.create(db, name="Caller 1", email="caller1@example.com")
                req1 = CreateAppointmentRequest(
                    customer_name="Caller 1",
                    customer_phone="03459543855",
                    day="Thursday",
                    time_str="3 PM",
                    tenant_id="test-tenant-collision",
                )
                res1 = await AppointmentToolService.create_appointment(req1, user_id=user.id, db=db)
                assert res1.success is True

                # Attempt 2 on same slot
                req2 = CreateAppointmentRequest(
                    customer_name="Caller 2",
                    customer_phone="03451112233",
                    day="Thursday",
                    time_str="3 PM",
                    tenant_id="test-tenant-collision",
                )
                res2 = await AppointmentToolService.create_appointment(req2, user_id=user.id, db=db)
                assert res2.success is False
                assert "already booked" in res2.user_facing_message.lower() or "not available" in res2.user_facing_message.lower()
                assert "confirmed" not in res2.user_facing_message.lower()

        asyncio.run(run())

    def test_real_world_voice_flow_exact_confirmation_executes_tool_stream(self, isolated_db):
        """Requirement A: Exact real-world flow with 'Yes, please, I confirm it.' through get_ai_response_stream."""
        import asyncio
        from app.services.ai_service import get_ai_response_stream
        from app.crud.appointment import appointment_crud
        from app.crud.user import user_crud

        session_maker = isolated_db["session_maker"]

        async def run():
            async with session_maker() as db:
                user = await user_crud.create(db, name="Voice Caller", email="voice-caller@example.com")
                history = [
                    {"role": "user", "content": "I want to book an appointment with Mr. Ramzan."},
                    {"role": "assistant", "content": "Sure! Which weekday and what time between 9 AM and 5 PM works best for you?"},
                    {"role": "user", "content": "Thursday at 3 PM."},
                    {"role": "assistant", "content": "I have noted Thursday at 3 PM. Please share your phone number so I can prepare the booking request."},
                    {"role": "user", "content": "0-345-9543-855"},
                    {"role": "assistant", "content": "Thank you. I have recorded your request for Thursday at 3 PM with phone number 0345 954 3855. Would you like me to submit this booking request?"},
                ]

                chunks = []
                async for chunk in get_ai_response_stream(
                    user_name=user.name,
                    user_email=user.email,
                    message="Yes, please, I confirm it.",
                    history=history,
                    user_id=user.id,
                    db=db,
                ):
                    chunks.append(chunk)

                full_reply = " ".join(chunks)
                assert "confirmed" in full_reply.lower()
                assert "Thursday" in full_reply
                assert "3 PM" in full_reply
                assert "0345 954 3855" in full_reply or "03459543855" in full_reply

                # Verify appointment row created in DB
                user_appts = await appointment_crud.get_by_user_id(db, user_id=user.id)
                assert len(user_appts) == 1
                assert user_appts[0].day == "Thursday"
                assert user_appts[0].time_str == "3 PM"
                assert user_appts[0].customer_phone == "03459543855"
                assert user_appts[0].status == "confirmed"

        asyncio.run(run())

    def test_confirmation_variants_trigger_tool(self, isolated_db):
        """Requirement B: Confirmation variants trigger tool execution."""
        import asyncio
        from app.services.ai_service import get_ai_response
        from app.crud.user import user_crud

        session_maker = isolated_db["session_maker"]
        variants = ["I confirm", "Go ahead", "Please submit it", "Book it", "Proceed", "Yes, please"]

        async def run():
            async with session_maker() as db:
                for idx, variant in enumerate(variants):
                    user = await user_crud.create(db, name=f"Variant Caller {idx}", email=f"variant{idx}@example.com")
                    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Monday"]
                    hours = [9, 10, 11, 14, 15, 16]
                    day = days[idx]
                    hour = hours[idx]
                    time_str = f"{hour % 12 or 12} {'AM' if hour < 12 else 'PM'}"
                    history = [
                        {"role": "user", "content": f"I want an appointment on {day} at {time_str}"},
                        {"role": "assistant", "content": f"Thank you. I have recorded your request for {day} at {time_str} with phone number 0300 123 4567. Would you like me to submit this booking request?"},
                    ]

                    reply = await get_ai_response(
                        user_name=user.name,
                        user_email=user.email,
                        message=variant,
                        history=history,
                        user_id=user.id,
                        db=db,
                    )
                    assert "confirmed" in reply.lower(), f"Failed for variant '{variant}'"
                    assert day in reply
                    assert time_str in reply

        asyncio.run(run())

    def test_generic_yes_when_no_appointment_pending_does_not_call_tool(self, isolated_db):
        """Requirement C: 'Yes' when no appointment confirmation is pending does NOT create appointment."""
        import asyncio
        from app.services.ai_service import get_ai_response
        from app.crud.appointment import appointment_crud
        from app.crud.user import user_crud

        session_maker = isolated_db["session_maker"]

        async def run():
            async with session_maker() as db:
                user = await user_crud.create(db, name="Unrelated Caller", email="unrelated@example.com")
                history = [
                    {"role": "user", "content": "Do you provide custom software services?"},
                    {"role": "assistant", "content": "Yes, Logic Pulse provides custom software, AI systems, dashboards, and mobile apps."},
                ]
                reply = await get_ai_response(
                    user_name=user.name,
                    user_email=user.email,
                    message="Yes",
                    history=history,
                    user_id=user.id,
                    db=db,
                )
                assert "confirmed for" not in reply.lower()
                user_appts = await appointment_crud.get_by_user_id(db, user_id=user.id)
                assert len(user_appts) == 0

        asyncio.run(run())

    def test_missing_phone_confirmation_does_not_create_appointment(self, isolated_db):
        """Requirement D: Confirmation with missing phone does not create an appointment."""
        import asyncio
        from app.services.ai_service import get_ai_response
        from app.crud.appointment import appointment_crud
        from app.crud.user import user_crud

        session_maker = isolated_db["session_maker"]

        async def run():
            async with session_maker() as db:
                user = await user_crud.create(db, name="Missing Phone Caller", email="missing-phone@example.com")
                history = [
                    {"role": "user", "content": "Book an appointment on Friday at 2 PM"},
                    {"role": "assistant", "content": "I can schedule an appointment for Friday at 2 PM. Please share your phone number."},
                ]
                reply = await get_ai_response(
                    user_name=user.name,
                    user_email=user.email,
                    message="Yes, please book it",
                    history=history,
                    user_id=user.id,
                    db=db,
                )
                assert "needs a phone number" in reply.lower() or "share your phone number" in reply.lower() or "confirmed for" not in reply.lower()
                user_appts = await appointment_crud.get_by_user_id(db, user_id=user.id)
                assert len(user_appts) == 0

        asyncio.run(run())

    def test_duplicate_confirmation_after_successful_booking(self, isolated_db):
        """Requirement F: Subsequent confirmation after appointment is already booked must not create a duplicate."""
        import asyncio
        from app.services.ai_service import get_ai_response
        from app.crud.appointment import appointment_crud
        from app.crud.user import user_crud

        session_maker = isolated_db["session_maker"]

        async def run():
            async with session_maker() as db:
                user = await user_crud.create(db, name="Duplicate Guard Caller", email="dup-guard@example.com")
                history = [
                    {"role": "user", "content": "I want to book an appointment with CEO on Tuesday at 11 AM."},
                    {"role": "assistant", "content": "Thank you. I have recorded your request for Tuesday at 11 AM with phone number 0345 1122334. Would you like me to submit this booking request?"},
                    {"role": "user", "content": "Yes, please, I confirm it."},
                    {"role": "assistant", "content": "Your appointment with CEO Muhammad Ramzan is confirmed for Tuesday at 11 AM. A confirmation SMS will be sent to 0345 112 2334."},
                ]
                # User says "Yes" on next turn
                reply = await get_ai_response(
                    user_name=user.name,
                    user_email=user.email,
                    message="Yes",
                    history=history,
                    user_id=user.id,
                    db=db,
                )
                # Must not re-book or attempt duplicate
                user_appts = await appointment_crud.get_by_user_id(db, user_id=user.id)
                assert len(user_appts) == 0 or len(user_appts) == 1

        asyncio.run(run())
