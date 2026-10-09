"""
tests/test_receptionist_router.py
Unit tests for the local rule-based ReceptionistRouter.
Covers all 7 bug fixes described in the prompt.
"""
import pytest
from app.services.receptionist_router import (
    ReceptionistRouter,
    normalise_input,
    validate_day,
    validate_time,
    hour_to_display,
    _extract_day,
    _extract_time_hour,
    WORKING_DAYS,
    WEEKEND_DAYS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Input Normalisation (Fix #6)
# ──────────────────────────────────────────────────────────────────────────────

class TestNormaliseInput:
    def test_compact_pm(self):
        assert "4 PM" in normalise_input("4p").upper()

    def test_compact_am(self):
        assert "9 AM" in normalise_input("9a").upper()

    def test_attached_pm(self):
        result = normalise_input("5pm")
        assert "5" in result and "PM" in result.upper()

    def test_typo_schedule(self):
        assert "schedule" in normalise_input("shadule").lower()
        assert "schedule" in normalise_input("shedule").lower()

    def test_typo_appointment(self):
        assert "appointment" in normalise_input("apointment").lower()

    def test_short_day_sat(self):
        assert "saturday" in normalise_input("sat").lower()

    def test_short_day_fri(self):
        # "fri" should expand to "friday", not "saturday"
        assert "friday" in normalise_input("fri").lower()


# ──────────────────────────────────────────────────────────────────────────────
# Weekend / Weekday Logic (Fix #1)
# ──────────────────────────────────────────────────────────────────────────────

class TestDayValidation:
    """Friday must be a working day. Only Saturday & Sunday are weekends."""

    @pytest.mark.parametrize("day", ["monday", "tuesday", "wednesday", "thursday", "friday"])
    def test_working_days_valid(self, day):
        valid, reason = validate_day(day)
        assert valid is True, f"{day} should be valid but got reason={reason}"

    @pytest.mark.parametrize("day", ["saturday", "sunday"])
    def test_weekend_days_invalid(self, day):
        valid, reason = validate_day(day)
        assert valid is False
        assert reason == "weekend"

    def test_friday_is_NOT_weekend(self):
        """Critical: Friday must never be treated as a weekend."""
        valid, reason = validate_day("friday")
        assert valid is True, "Friday is a working day and must be valid"

    def test_unknown_day(self):
        valid, reason = validate_day("holyday")
        assert valid is False
        assert reason == "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Time Validation (Fix #2)
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeValidation:
    @pytest.mark.parametrize("hour", [9, 10, 11, 12, 13, 14, 15, 16, 17])
    def test_valid_hours(self, hour):
        valid, _ = validate_time(hour)
        assert valid is True, f"Hour {hour} should be valid"

    def test_5pm_is_valid(self):
        """5 PM = hour 17 must be accepted."""
        valid, _ = validate_time(17)
        assert valid is True

    def test_midnight_12am_rejected(self):
        """12 AM = hour 0 must always be rejected."""
        valid, reason = validate_time(0)
        assert valid is False
        assert reason == "midnight"

    @pytest.mark.parametrize("hour", [1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23])
    def test_out_of_hours(self, hour):
        valid, reason = validate_time(hour)
        assert valid is False
        assert reason == "out_of_hours"


# ──────────────────────────────────────────────────────────────────────────────
# Hour Display
# ──────────────────────────────────────────────────────────────────────────────

class TestHourToDisplay:
    def test_9am(self):
        assert hour_to_display(9) == "9 AM"

    def test_12pm(self):
        assert hour_to_display(12) == "12 PM"

    def test_5pm(self):
        assert hour_to_display(17) == "5 PM"

    def test_midnight(self):
        assert hour_to_display(0) == "12 AM"


# ──────────────────────────────────────────────────────────────────────────────
# Time Extraction from message text
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractTimeHour:
    def test_4pm(self):
        assert _extract_time_hour("I am free at 4 PM") == 16

    def test_5pm(self):
        assert _extract_time_hour("available at 5 PM") == 17

    def test_9am(self):
        assert _extract_time_hour("9 AM works") == 9

    def test_12am_midnight(self):
        assert _extract_time_hour("12 AM please") == 0  # midnight

    def test_12pm_noon(self):
        assert _extract_time_hour("12 PM") == 12  # noon


# ──────────────────────────────────────────────────────────────────────────────
# Router — CEO query
# ──────────────────────────────────────────────────────────────────────────────

class TestCEOQuery:
    def test_who_is_ceo(self):
        r = ReceptionistRouter.get_hardcoded_response("Who is the CEO?")
        assert "Muhammad Ramzan" in r

    def test_ceo_name(self):
        r = ReceptionistRouter.get_hardcoded_response("What is the CEO name?")
        assert "Muhammad Ramzan" in r


# ──────────────────────────────────────────────────────────────────────────────
# Router — Timing intent
# ──────────────────────────────────────────────────────────────────────────────

class TestTimingIntent:
    def test_office_hours(self):
        r = ReceptionistRouter.get_hardcoded_response("What are your office hours?")
        assert "9 AM" in r or "Monday" in r

    def test_weekend_query(self):
        r = ReceptionistRouter.get_hardcoded_response("Are you open on Saturday?")
        assert "Saturday" in r or "weekend" in r.lower() or "closed" in r.lower()

    def test_friday_NOT_weekend(self):
        """Friday query must NOT return a 'closed on weekend' message."""
        r = ReceptionistRouter.get_hardcoded_response("Are you open on Friday?")
        # Should say open/working — must NOT say closed on weekends
        assert "closed on weekend" not in r.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Router — Appointment booking (Fix #1 #2 #3 #4 #5)
# ──────────────────────────────────────────────────────────────────────────────

class TestAppointmentBooking:
    def test_valid_friday_5pm(self):
        """Friday at 5 PM is valid — must confirm, not reject."""
        r = ReceptionistRouter.get_hardcoded_response("I am free Friday at 5 PM")
        assert "Muhammad Ramzan" in r or "appointment" in r.lower()
        assert "outside our working hours" not in r

    def test_saturday_rejected(self):
        r = ReceptionistRouter.get_hardcoded_response("Can I book Saturday at 10 AM?")
        assert "weekend" in r.lower() or "saturday" in r.lower() or "weekday" in r.lower()

    def test_midnight_rejected(self):
        r = ReceptionistRouter.get_hardcoded_response("Book me at 12 AM Monday")
        assert "midnight" in r.lower() or "outside" in r.lower() or "12 am" in r.lower()

    def test_5pm_accepted(self):
        r = ReceptionistRouter.get_hardcoded_response("Wednesday at 5 PM please")
        assert "outside our working hours" not in r
        assert "Muhammad Ramzan" in r or "appointment" in r.lower()

    def test_missing_day_asks_for_day(self):
        r = ReceptionistRouter.get_hardcoded_response("Book me at 3 PM")
        assert "day" in r.lower() or "monday" in r.lower() or "which day" in r.lower()

    def test_missing_time_asks_for_time(self):
        r = ReceptionistRouter.get_hardcoded_response("Book me on Monday")
        assert "time" in r.lower() or "9 am" in r.lower() or "what time" in r.lower()

    def test_context_memory_day_reused(self):
        """System should remember the day from history and not ask again."""
        history = [
            {"role": "user",      "content": "I want to book on Thursday"},
            {"role": "assistant", "content": "Got it — Thursday works! What time would you prefer?"},
        ]
        r = ReceptionistRouter.get_hardcoded_response("3 PM", history=history)
        # Should confirm Thursday at 3 PM, not ask for the day again
        assert "Thursday" in r or "appointment" in r.lower()
        assert "which day" not in r.lower()

    def test_confirmation_step_present(self):
        """Valid day+time should ask for confirmation or phone before finalising."""
        r = ReceptionistRouter.get_hardcoded_response("Tuesday at 11 AM")
        # Must mention CEO name and some confirmation prompt
        assert "Muhammad Ramzan" in r
        assert any(w in r.lower() for w in ["confirm", "phone", "sms", "number"])


# ──────────────────────────────────────────────────────────────────────────────
# Router — Escalation after 2 clarifications (Fix #7)
# ──────────────────────────────────────────────────────────────────────────────

class TestEscalation:
    def _make_history(self, n_clarifications: int) -> list[dict]:
        """Build history with n consecutive clarification assistant messages."""
        history = []
        for _ in range(n_clarifications):
            history.append({"role": "user",      "content": "xyz?????"})
            history.append({"role": "assistant",  "content": "I didn't fully understand. Could you please clarify your preferred day and time?"})
        return history

    def test_no_escalation_on_first_unrecognised(self):
        """With no prior clarification in history, first unrecognised msg → None (goes to LLM)."""
        r = ReceptionistRouter.get_hardcoded_response("xyz????")
        # No history → no escalation, delegates to LLM (returns None)
        assert r is None

    def test_escalation_after_one_clarification_in_history(self):
        """If assistant already clarified once and user still unclear → escalate."""
        history = self._make_history(1)
        r = ReceptionistRouter.get_hardcoded_response("xyz?????", history=history)
        assert r is not None
        assert "human agent" in r.lower() or "connect" in r.lower()

    def test_escalation_after_two_clarifications_in_history(self):
        """2 prior clarifications → immediate escalation gate fires (count >= 2)."""
        history = self._make_history(2)
        r = ReceptionistRouter.get_hardcoded_response("xyz?????", history=history)
        assert r is not None
        assert "human agent" in r.lower() or "connect" in r.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Router — Simple intents (services, pricing, greeting, closing)
# ──────────────────────────────────────────────────────────────────────────────

class TestSimpleIntents:
    def test_greeting(self):
        r = ReceptionistRouter.get_hardcoded_response("Hello!")
        assert "Logic Pulse" in r

    def test_services(self):
        r = ReceptionistRouter.get_hardcoded_response("What services do you offer?")
        assert r is not None and len(r) > 5

    def test_pricing(self):
        r = ReceptionistRouter.get_hardcoded_response("How much does it cost?")
        assert r is not None and "pricing" in r.lower() or "cost" in r.lower() or "consultation" in r.lower()

    def test_closing(self):
        r = ReceptionistRouter.get_hardcoded_response("Goodbye!")
        assert "Logic Pulse" in r or "goodbye" in r.lower()

    def test_free_consultation(self):
        r = ReceptionistRouter.get_hardcoded_response("Do you offer free consultation?")
        assert "consultation" in r.lower() or "free" in r.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Router — Pakistani Phone Number Validation Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPakistaniPhoneNumberValidation:
    def test_valid_pakistani_phone_numbers(self):
        history = [
            {"role": "user", "content": "I want to book an appointment"},
            {"role": "assistant", "content": "Sure, please confirm your phone number to receive the details via SMS."}
        ]
        valid_numbers = ["03001234567", "0312-3456789", "0333 9876543", "+923001234567", "923001234567"]
        for num in valid_numbers:
            r = ReceptionistRouter.get_hardcoded_response(num, history=history)
            assert "recorded" in r or "saved" in r or "confirmation" in r
            assert "missed a digit" not in r

    def test_invalid_pakistani_phone_numbers(self):
        history = [
            {"role": "user", "content": "I want to book an appointment"},
            {"role": "assistant", "content": "Sure, please confirm your phone number to receive the details via SMS."}
        ]
        invalid_numbers = ["0300123456", "12345678901", "030012345678", "02001234567"]
        for num in invalid_numbers:
            r = ReceptionistRouter.get_hardcoded_response(num, history=history)
            assert r == "I think I missed a digit. Could you please repeat your 11-digit phone number?"


# ──────────────────────────────────────────────────────────────────────────────
# Router — Urdu / Roman Urdu & Microphone Check Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestUrduAndMicCheckIntents:
    def test_urdu_greeting(self):
        r1 = ReceptionistRouter.get_hardcoded_response("السلام علیکم")
        assert "وعلیکم السلام" in r1 and "Logic Pulse" in r1

        r2 = ReceptionistRouter.get_hardcoded_response("اسلام علیکم")
        assert "وعلیکم السلام" in r2 and "Logic Pulse" in r2

    def test_roman_urdu_greeting(self):
        r = ReceptionistRouter.get_hardcoded_response("Assalam o Alaikum")
        assert "Walaikum Assalam" in r and "Logic Pulse" in r

    def test_urdu_mic_check(self):
        r = ReceptionistRouter.get_hardcoded_response("کیا میری آواز آ رہی ہے؟")
        assert "جی ہاں" in r and "آواز" in r

    def test_roman_urdu_mic_check(self):
        r = ReceptionistRouter.get_hardcoded_response("Kya meri awaz aa rahi hai?")
        assert "Ji haan" in r and "awaaz" in r

    def test_urdu_salam_and_mic_check_combo(self):
        # User says both greeting and mic check together
        r1 = ReceptionistRouter.get_hardcoded_response("السلام علیکم! کیا میری آواز آ رہی ہے؟")
        assert "وعلیکم السلام" in r1 and "آواز" in r1

        r2 = ReceptionistRouter.get_hardcoded_response("اسلام علیکم كير ميرى وعلا رہے")
        assert "وعلیکم السلام" in r2 and "آواز" in r2

    def test_appointment_context_does_not_hijack_mic_check_or_greeting(self):
        # If user booked appointment earlier, a new greeting or mic check must not be hijacked by old appointment
        history = [
            {"role": "user", "content": "I am free Thursday at 3 PM"},
            {"role": "assistant", "content": "Your appointment with our CEO, Muhammad Ramzan, is scheduled for Thursday at 3 PM. Please confirm..."},
            {"role": "user", "content": "Yes, confirm it"},
            {"role": "assistant", "content": "Your appointment is confirmed!"}
        ]
        r = ReceptionistRouter.get_hardcoded_response("السلام علیکم، کیا میری آواز آ رہی ہے؟", history=history)
        assert "Thursday" not in r and "3 PM" not in r
        assert "وعلیکم السلام" in r and "آواز" in r


# ──────────────────────────────────────────────────────────────────────────────
# Router — Gratitude vs Closing Tests (Fix #4)
# ──────────────────────────────────────────────────────────────────────────────

class TestGratitudeAndClosing:
    def test_standalone_english_gratitude_does_not_close(self):
        gratitude_phrases = ["thank you", "thanks", "ok, thank you", "okay, thank you", "thank you very much", "many thanks"]
        for phrase in gratitude_phrases:
            r = ReceptionistRouter.get_hardcoded_response(phrase)
            assert r == "You're welcome! Is there anything else I can help you with?"

    def test_standalone_roman_urdu_gratitude_does_not_close(self):
        for phrase in ["shukriya", "boht shukriya", "bohat shukriya"]:
            r = ReceptionistRouter.get_hardcoded_response(phrase)
            assert r == "Khush amdeed! Kya main aapki kisi aur cheez mein madad kar sakta hoon?"

    def test_standalone_urdu_script_gratitude_does_not_close(self):
        for phrase in ["شکریہ", "بہت شکریہ"]:
            r = ReceptionistRouter.get_hardcoded_response(phrase)
            assert r == "خوش آمدید! کیا میں آپ کی کسی اور چیز میں مدد کر سکتا ہوں؟"

    def test_gratitude_with_query_not_closing(self):
        # "shukriya, mujhe pricing batao" must not be treated as closing or standalone thanks
        r = ReceptionistRouter.get_hardcoded_response("shukriya, mujhe pricing batao")
        # Should route to pricing response or LLM
        assert r != "You're welcome! Is there anything else I can help you with?"
        assert r != "Khush amdeed! Kya main aapki kisi aur cheez mein madad kar sakta hoon?"
        assert "Aap ke rabte ka shukriya" not in str(r)
        assert "Goodbye" not in str(r)

    def test_closing_phrases_english(self):
        for phrase in ["bye", "goodbye", "ok bye", "thank you, bye", "thanks bye", "goodbye now", "that's all", "have a nice day"]:
            r = ReceptionistRouter.get_hardcoded_response(phrase)
            assert "Goodbye" in r or "thank you" in r.lower()

    def test_closing_phrases_roman_urdu(self):
        for phrase in ["allah hafiz", "khuda hafiz", "alvida", "ok allah hafiz"]:
            r = ReceptionistRouter.get_hardcoded_response(phrase)
            assert "Allah Hafiz" in r

    def test_closing_phrases_urdu_script(self):
        for phrase in ["اللہ حافظ", "خدا حافظ", "الوداع", "بس شکریہ"]:
            r = ReceptionistRouter.get_hardcoded_response(phrase)
            assert "اللہ حافظ" in r

    def test_greeting_how_are_you_natural_reply(self):
        r_en = ReceptionistRouter.get_hardcoded_response("Hi, how are you?")
        assert r_en is not None
        assert "well" in r_en.lower() or "great" in r_en.lower()
        assert "Logic Pulse" in r_en

        r_ur = ReceptionistRouter.get_hardcoded_response("Salam, kese ho aap?")
        assert r_ur is not None
        assert "theek" in r_ur.lower()
        assert "Logic Pulse" in r_ur

    def test_greeting_natural_variations_across_turns(self):
        h0 = []
        h1 = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello"}]
        r0 = ReceptionistRouter.get_hardcoded_response("Hello", history=h0)
        r1 = ReceptionistRouter.get_hardcoded_response("Hello", history=h1)
        assert r0 is not None and r1 is not None
        assert "Logic Pulse" in r0 and "Logic Pulse" in r1

    def test_gratitude_in_appointment_flow_retains_context(self):
        history = [
            {"role": "user", "content": "I want an appointment with Mr. Ramzan"},
            {"role": "assistant", "content": "I'd be happy to schedule an appointment with our CEO, Muhammad Ramzan. What day and time works for you?"},
        ]
        r = ReceptionistRouter.get_hardcoded_response("Thank you.", history=history)
        assert r is not None
        assert "appointment" in r.lower() or "day" in r.lower()
        assert "Is there anything else I can help you with?" not in r

    def test_repeated_gratitude_closes_gracefully(self):
        history = [
            {"role": "user", "content": "Thank you."},
            {"role": "assistant", "content": "You're welcome! Is there anything else I can help you with?"},
        ]
        r = ReceptionistRouter.get_hardcoded_response("Thank you.", history=history)
        assert r is not None
        assert "Goodbye" in r or "thank you for contacting" in r.lower()

    def test_hesitation_phrase_does_not_escalate(self):
        history = [
            {"role": "user", "content": "I took it away"},
            {"role": "assistant", "content": "Could you please clarify what you mean by 'I took it away'?"},
        ]
        r = ReceptionistRouter.get_hardcoded_response("I think", history=history)
        # Should NOT trigger fallback to human agent; should delegate to LLM (None)
        assert r is None

    def test_standalone_acknowledgments_zero_api_cost(self):
        # Conversational fillers / acknowledgments must be answered locally with zero LLM API calls
        fillers = ["All right.", "Alright", "Right.", "So,", "Okay.", "Ok", "Yeah.", "Sure."]
        for filler in fillers:
            r = ReceptionistRouter.get_hardcoded_response(filler)
            assert r is not None, f"Expected local response for filler '{filler}', got None"
            assert ("help" in r.lower() or "assist" in r.lower() or "listening" in r.lower() or "services" in r.lower())

    def test_standalone_acknowledgments_urdu_and_roman(self):
        # Urdu and Roman Urdu acknowledgments
        r_roman = ReceptionistRouter.get_hardcoded_response("Theek hai")
        assert r_roman is not None
        assert "madad" in r_roman.lower() or "sun raha hoon" in r_roman.lower()

        r_urdu = ReceptionistRouter.get_hardcoded_response("ٹھیک ہے")
        assert r_urdu is not None
        assert "مدد" in r_urdu or "سن رہا ہوں" in r_urdu

    def test_acknowledgment_during_appointment_booking(self):
        history = [
            {"role": "user", "content": "I want to book an appointment with your CEO."},
            {"role": "assistant", "content": "I'd be happy to schedule an appointment with our CEO, Muhammad Ramzan. What day and time works for you?"},
        ]
        r = ReceptionistRouter.get_hardcoded_response("All right.", history=history)
        assert r is not None
        assert "day" in r.lower() or "time" in r.lower() or "appointment" in r.lower()

    def test_all_right_confirms_appointment(self):
        history = [
            {"role": "user", "content": "I want to book an appointment with your CEO for Wednesday at 3 PM. My number is 03001234567."},
            {"role": "assistant", "content": "I have recorded your request for Wednesday at 3 PM with phone number 0300 1234567. Would you like me to submit this booking request?"},
        ]
        r = ReceptionistRouter.get_hardcoded_response("All right.", history=history)
        assert r is not None
        assert "submitted" in r.lower() or "confirm" in r.lower()

