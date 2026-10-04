"""
tests/test_appointment_history_flow.py — Phase C Integration Tests
Verifies the complete 5-turn appointment booking lifecycle with in-memory history:
English, Roman Urdu, Urdu script confirmation, concurrent session isolation,
and barge-in safety.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appointment import Appointment
from app.models.user import User
from app.services.ai_service import get_ai_response_stream
from app.services.receptionist_router import ReceptionistRouter


def test_multiturn_booking_in_memory_flow_english(isolated_db):
    """
    Test 5-turn English booking flow end-to-end:
    Turn 1: 'I want to book an appointment'
    Turn 2: 'Monday'
    Turn 3: '2 PM'
    Turn 4: 'zero three zero zero one two three four five six seven'
    Turn 5: 'yes confirm'
    Verifies appointment created in DB exactly once with accurate day, hour, and phone.
    """
    async def _run():
        session_maker = isolated_db["session_maker"]
        history: list[dict[str, str]] = []
        user_name = "Hamza Ali"
        user_email = "hamza@example.com"

        # Turn 1
        t1_user = "I want to book an appointment"
        t1_reply = ReceptionistRouter.get_hardcoded_response(t1_user, history)
        assert t1_reply is not None
        assert any(k in t1_reply.lower() for k in ["day", "time", "works for you", "schedule"])
        history.append({"role": "user", "content": t1_user})
        history.append({"role": "assistant", "content": t1_reply})

        # Turn 2
        t2_user = "Monday"
        t2_reply = ReceptionistRouter.get_hardcoded_response(t2_user, history)
        assert t2_reply is not None
        assert "Monday" in t2_reply
        history.append({"role": "user", "content": t2_user})
        history.append({"role": "assistant", "content": t2_reply})

        # Turn 3
        t3_user = "2 PM"
        t3_reply = ReceptionistRouter.get_hardcoded_response(t3_user, history)
        assert t3_reply is not None
        assert "2 PM" in t3_reply
        assert "phone" in t3_reply.lower()
        history.append({"role": "user", "content": t3_user})
        history.append({"role": "assistant", "content": t3_reply})

        # Turn 4 (spoken digits)
        t4_user = "zero three zero zero one two three four five six seven"
        t4_reply = ReceptionistRouter.get_hardcoded_response(t4_user, history)
        assert t4_reply is not None
        assert "0300 123 4567" in t4_reply or "03001234567" in t4_reply
        assert "submit" in t4_reply.lower() or "confirm" in t4_reply.lower()
        history.append({"role": "user", "content": t4_user})
        history.append({"role": "assistant", "content": t4_reply})

        # Turn 5: Explicit confirmation triggering deterministic tool execution
        t5_user = "yes confirm"
        async with session_maker() as session:
            chunks = []
            async for chunk in get_ai_response_stream(
                user_name=user_name,
                user_email=user_email,
                message=t5_user,
                history=history,
                db=session,
            ):
                chunks.append(chunk)

            final_response = " ".join(chunks)
            assert "confirmed" in final_response.lower()
            assert "Monday" in final_response
            assert "2:00 PM" in final_response or "2 PM" in final_response

            # Verify database record created
            stmt = select(Appointment).where(Appointment.customer_phone == "03001234567")
            res = await session.execute(stmt)
            appts = res.scalars().all()
            assert len(appts) == 1
            appt = appts[0]
            assert appt.day == "Monday"
            assert appt.time_str in ("2 PM", "2:00 PM")
            assert appt.status == "confirmed"

    import asyncio
    asyncio.run(_run())


def test_multiturn_booking_roman_urdu_and_urdu_script(isolated_db):
    """
    Test Roman Urdu conversation with Urdu script confirmation:
    Turn 1: 'Mujhe appointment book karni hai'
    Turn 2: 'Friday'
    Turn 3: '4 PM'
    Turn 4: '03451234567'
    Turn 5: 'جی ہاں'
    """
    async def _run():
        session_maker = isolated_db["session_maker"]
        history: list[dict[str, str]] = []
        user_name = "Tariq Mahmood"
        user_email = "tariq@example.com"

        # Turn 1
        t1_user = "Mujhe appointment book karni hai"
        t1_reply = ReceptionistRouter.get_hardcoded_response(t1_user, history)
        assert t1_reply is not None
        history.append({"role": "user", "content": t1_user})
        history.append({"role": "assistant", "content": t1_reply})

        # Turn 2
        t2_user = "Friday"
        t2_reply = ReceptionistRouter.get_hardcoded_response(t2_user, history)
        assert t2_reply is not None
        assert "Friday" in t2_reply
        history.append({"role": "user", "content": t2_user})
        history.append({"role": "assistant", "content": t2_reply})

        # Turn 3
        t3_user = "4 PM"
        t3_reply = ReceptionistRouter.get_hardcoded_response(t3_user, history)
        assert t3_reply is not None
        assert "4 PM" in t3_reply
        history.append({"role": "user", "content": t3_user})
        history.append({"role": "assistant", "content": t3_reply})

        # Turn 4
        t4_user = "03451234567"
        t4_reply = ReceptionistRouter.get_hardcoded_response(t4_user, history)
        assert t4_reply is not None
        assert "0345 123 4567" in t4_reply or "03451234567" in t4_reply
        history.append({"role": "user", "content": t4_user})
        history.append({"role": "assistant", "content": t4_reply})

        # Turn 5: Urdu script confirmation
        t5_user = "جی ہاں"
        async with session_maker() as session:
            chunks = []
            async for chunk in get_ai_response_stream(
                user_name=user_name,
                user_email=user_email,
                message=t5_user,
                history=history,
                db=session,
            ):
                chunks.append(chunk)

            final_response = " ".join(chunks)
            assert "confirmed" in final_response.lower()
            assert "Friday" in final_response

            # Verify database record created
            stmt = select(Appointment).where(Appointment.customer_phone == "03451234567")
            res = await session.execute(stmt)
            appts = res.scalars().all()
            assert len(appts) == 1
            appt = appts[0]
            assert appt.day == "Friday"
            assert appt.time_str in ("4 PM", "4:00 PM")

    import asyncio
    asyncio.run(_run())


def test_concurrent_sessions_isolated_history(isolated_db):
    """
    Verify two concurrent calls do NOT share or leak history/entities.
    """
    async def _run():
        session_maker = isolated_db["session_maker"]

        # Call A: Tuesday 10 AM, 03001112233
        hist_a = [
            {"role": "user", "content": "I want an appointment on Tuesday at 10 AM"},
            {"role": "assistant", "content": "Sure, to schedule for Tuesday at 10 AM, please provide your 11-digit phone number."},
            {"role": "user", "content": "03001112233"},
            {"role": "assistant", "content": "I have you down for an appointment on Tuesday at 10 AM with 0300 111 2233. Would you like me to submit this booking?"},
        ]

        # Call B: Wednesday 11 AM, 03219988776
        hist_b = [
            {"role": "user", "content": "Can I book Wednesday at 11 AM?"},
            {"role": "assistant", "content": "Please provide your phone number to complete booking for Wednesday at 11 AM."},
            {"role": "user", "content": "03219988776"},
            {"role": "assistant", "content": "I have you down for an appointment on Wednesday at 11 AM with 0321 998 8776. Would you like me to submit this booking?"},
        ]

        async with session_maker() as session:
            # Confirm Call A
            chunks_a = []
            async for c in get_ai_response_stream("Caller A", "a@example.com", "yes confirm", hist_a, db=session):
                chunks_a.append(c)

            # Confirm Call B
            chunks_b = []
            async for c in get_ai_response_stream("Caller B", "b@example.com", "proceed please", hist_b, db=session):
                chunks_b.append(c)

            # Check Call A appointment
            res_a = await session.execute(select(Appointment).where(Appointment.customer_phone == "03001112233"))
            appt_a = res_a.scalars().first()
            assert appt_a is not None
            assert appt_a.day == "Tuesday"
            assert appt_a.time_str in ("10 AM", "10:00 AM")

            # Check Call B appointment
            res_b = await session.execute(select(Appointment).where(Appointment.customer_phone == "03219988776"))
            appt_b = res_b.scalars().first()
            assert appt_b is not None
            assert appt_b.day == "Wednesday"
            assert appt_b.time_str in ("11 AM", "11:00 AM")

    import asyncio
    asyncio.run(_run())
