"""
tests/test_appointments_and_tools.py — Appointment Multi-Turn State & Business Rules Tests
"""

from app.services.receptionist_router import ReceptionistRouter


def test_appointment_full_multiturn_flow():
    """Test full multi-turn conversation from intent to confirmation and phone number capture."""
    history = []

    # Turn 1: Initial booking request without day/time
    m1 = "I would like to book an appointment with Muhammad Ramzan."
    r1 = ReceptionistRouter.get_hardcoded_response(m1, history)
    assert any(k in r1.lower() for k in ["day", "time", "works for you", "schedule"])
    history.extend([{"role": "user", "content": m1}, {"role": "assistant", "content": r1}])

    # Turn 2: Providing valid day and time (Friday 3 PM)
    m2 = "How about Friday at 3 PM?"
    r2 = ReceptionistRouter.get_hardcoded_response(m2, history)
    assert "Friday" in r2
    assert "3 PM" in r2
    assert "Muhammad Ramzan" in r2
    assert "confirm" in r2.lower()
    history.extend([{"role": "user", "content": m2}, {"role": "assistant", "content": r2}])

    # Turn 3: User confirms.  The pure dialogue layer must not claim a side
    # effect before the deterministic booking service has executed it.
    m3 = "Yes, please confirm."
    r3 = ReceptionistRouter.get_hardcoded_response(m3, history)
    assert "phone number" in r3.lower()
    assert "confirmed" not in r3.lower()
    assert "phone number" in r3.lower()
    history.extend([{"role": "user", "content": m3}, {"role": "assistant", "content": r3}])

    # Turn 4: User shares valid Pakistani phone number
    m4 = "My number is 03001234567"
    r4 = ReceptionistRouter.get_hardcoded_response(m4, history)
    assert "0300 123 4567" in r4
    assert "booking request" in r4.lower()
    assert "confirmation SMS" not in r4


def test_appointment_weekend_rejection():
    """Verify Saturday and Sunday are strictly rejected with explanation."""
    # Saturday
    r_sat = ReceptionistRouter.get_hardcoded_response("Can I book for Saturday at 2 PM?", [])
    assert "weekend" in r_sat.lower()
    assert "Saturday" in r_sat or "saturday" in r_sat.lower()

    # Sunday
    r_sun = ReceptionistRouter.get_hardcoded_response("I want Sunday at 10 AM", [])
    assert "weekend" in r_sun.lower()


def test_appointment_friday_is_accepted():
    """Verify Friday is strictly accepted as a valid working day."""
    r_fri = ReceptionistRouter.get_hardcoded_response("Book Friday at 11 AM", [])
    assert "Friday" in r_fri
    assert "11 AM" in r_fri
    assert "weekend" not in r_fri.lower()


def test_appointment_out_of_hours_rejection():
    """Verify times outside 9 AM–5 PM (including midnight) are rejected."""
    # Midnight 12 AM
    r_mid = ReceptionistRouter.get_hardcoded_response("Schedule meeting for Monday at 12 AM", [])
    assert "outside our working hours" in r_mid.lower() or "12 am" in r_mid.lower()

    # Evening 7 PM
    r_eve = ReceptionistRouter.get_hardcoded_response("Tuesday at 7 PM please", [])
    assert "outside our working hours" in r_eve.lower()

    # Early morning 6 AM
    r_early = ReceptionistRouter.get_hardcoded_response("Wednesday at 6 AM", [])
    assert "outside our working hours" in r_early.lower()


def test_appointment_5pm_is_valid():
    """Verify 5 PM is accepted as a valid time."""
    r_5pm = ReceptionistRouter.get_hardcoded_response("Let's meet Thursday at 5 PM", [])
    assert "Thursday" in r_5pm
    assert "5 PM" in r_5pm


def test_appointment_pakistani_phone_validation():
    """Verify 11-digit Pakistani phone numbers are accepted and incomplete ones rejected."""
    history = [{"role": "assistant", "content": "Could you please confirm your phone number?"}]

    # Valid
    r_valid = ReceptionistRouter.get_hardcoded_response("03217654321", history)
    assert "0321 765 4321" in r_valid
    assert "confirmation SMS" not in r_valid

    # Incomplete / invalid format (e.g. 5 digits)
    r_invalid = ReceptionistRouter.get_hardcoded_response("0300123", history)
    assert "missed a digit" in r_invalid.lower() or "11-digit" in r_invalid.lower()


def test_appointment_two_strike_escalation():
    """Verify escalation to a human agent after 2 consecutive failed clarification responses."""
    clarification_msg = {"role": "assistant", "content": "I didn't fully understand. Could you please clarify your preferred day and time?"}
    history = [
        {"role": "user", "content": "asdfghjk"},
        clarification_msg,
        {"role": "user", "content": "qwertyuiop"},
        clarification_msg,
    ]

    r_escalate = ReceptionistRouter.get_hardcoded_response("zxcvbnm", history)
    assert "connect you to a human agent" in r_escalate.lower() or "transfer you" in r_escalate.lower()
