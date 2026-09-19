import asyncio
from datetime import datetime, timedelta, timezone
import json
import pytest
from sqlalchemy import select

from app.crud.appointment import appointment_crud
from app.crud.outbox import outbox_crud
from app.crud.user import user_crud
from app.models.outbox import OutboxEvent
from app.schemas.tools import CreateAppointmentRequest, FailureMode
from app.services.outbox_service import OutboxRunner, OutboxWorker, outbox_worker
from app.services.tools_service import AppointmentToolService


def test_outbox_runner_lifecycle_and_graceful_shutdown(isolated_db):
    """Verify OutboxRunner starts, processes queued events, and shuts down gracefully."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        processed_events = []

        async def track_sms(payload):
            processed_events.append(payload["appointment_id"])

        worker = OutboxWorker()
        worker.set_handlers_for_testing("appointment.created", [track_sms])

        runner = OutboxRunner(
            session_maker=session_maker,
            worker=worker,
            poll_interval_seconds=0.05,
            batch_size=5,
            stale_timeout_seconds=300,
        )

        # Seed 2 appointments
        async with session_maker() as db:
            user = await user_crud.create(db, name="Runner User", email="runner@example.com")
            req1 = CreateAppointmentRequest(
                customer_name="Runner User 1",
                customer_phone="03001234567",
                day="Monday",
                time_str="10 AM",
            )
            req2 = CreateAppointmentRequest(
                customer_name="Runner User 2",
                customer_phone="03009876543",
                day="Tuesday",
                time_str="11 AM",
            )
            res1 = await AppointmentToolService.create_appointment(req1, user_id=user.id, db=db)
            res2 = await AppointmentToolService.create_appointment(req2, user_id=user.id, db=db)
            assert res1.success is True and res2.success is True

        # Launch runner in background task
        runner_task = asyncio.create_task(runner.run_continuous())
        assert runner.is_running or not runner._shutdown_event.is_set()

        # Allow runner to poll and process
        await asyncio.sleep(0.2)

        # Signal graceful shutdown
        runner.stop()
        await asyncio.wait_for(runner_task, timeout=1.0)
        assert runner.is_running is False

        # Verify all events were processed
        assert len(processed_events) == 2

        # Verify DB status is COMPLETED for both
        async with session_maker() as db:
            stmt = select(OutboxEvent).where(OutboxEvent.status == "COMPLETED")
            completed = (await db.execute(stmt)).scalars().all()
            assert len(completed) == 2

    asyncio.run(_run())


def test_outbox_runner_exception_isolation_survives_batch_error(isolated_db, monkeypatch):
    """Verify runner catches unexpected batch-level DB/runtime errors without crashing."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        fail_count = [0]
        worker = OutboxWorker()

        # Simulate a transient session-level failure on first iteration
        orig_claim = outbox_crud.claim_events

        async def flaky_claim(db, limit=50, stale_timeout_seconds=300):
            if fail_count[0] == 0:
                fail_count[0] += 1
                raise ConnectionResetError("Simulated transient DB connection reset")
            return await orig_claim(db, limit, stale_timeout_seconds)

        monkeypatch.setattr(outbox_crud, "claim_events", flaky_claim)

        runner = OutboxRunner(
            session_maker=session_maker,
            worker=worker,
            poll_interval_seconds=0.05,
            batch_size=5,
        )

        async with session_maker() as db:
            user = await user_crud.create(db, name="Error User", email="error@example.com")
            req = CreateAppointmentRequest(
                customer_name="Error User",
                customer_phone="03001234567",
                day="Wednesday",
                time_str="2 PM",
            )
            await AppointmentToolService.create_appointment(req, user_id=user.id, db=db)

        runner_task = asyncio.create_task(runner.run_continuous())
        await asyncio.sleep(0.25)
        runner.stop()
        await asyncio.wait_for(runner_task, timeout=1.0)

        # Verify that the runner recovered from the first failure and processed the event
        assert fail_count[0] == 1
        async with session_maker() as db:
            stmt = select(OutboxEvent).where(OutboxEvent.status == "COMPLETED")
            completed = (await db.execute(stmt)).scalars().all()
            assert len(completed) == 1

    asyncio.run(_run())


def test_outbox_stale_processing_event_recovery(isolated_db):
    """Verify events left in PROCESSING beyond stale timeout are safely reclaimed and retried."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        worker = OutboxWorker()
        processed_ids = []

        async def handler(payload):
            processed_ids.append(payload["appointment_id"])

        worker.set_handlers_for_testing("appointment.created", [handler])

        # 1. Manually insert a stale PROCESSING event (started 10 minutes ago)
        ten_minutes_ago = datetime.now(timezone.utc) - timedelta(seconds=600)
        async with session_maker() as db:
            stale_event = OutboxEvent(
                event_type="appointment.created",
                payload=json.dumps({"appointment_id": 999, "customer_name": "Stale User"}),
                status="PROCESSING",
                retry_count=0,
                max_retries=3,
                processing_started_at=ten_minutes_ago,
            )
            db.add(stale_event)
            await db.commit()
            stale_id = stale_event.id

        # 2. Claim events with stale timeout = 300s (5 mins)
        async with session_maker() as db:
            stats = await worker.process_batch(db, limit=10, stale_timeout_seconds=300)
            assert stats["total"] == 1
            assert stats["succeeded"] == 1
            assert 999 in processed_ids

        # 3. Verify event is now COMPLETED with retry_count incremented to 1
        async with session_maker() as db:
            recovered = await outbox_crud.get_by_id(db, stale_id)
            assert recovered is not None
            assert recovered.status == "COMPLETED"
            assert recovered.retry_count == 1
            assert recovered.processed_at is not None

    asyncio.run(_run())


def test_outbox_permanent_failure_after_max_retries(isolated_db):
    """Verify failing provider handler transitions event to FAILED permanently when max_retries reached."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async def failing_sms(payload):
            raise TimeoutError("Third-party SMS Gateway timed out")

        worker = OutboxWorker()
        worker.set_handlers_for_testing("appointment.created", [failing_sms])

        # Create 1 event with max_retries=2
        async with session_maker() as db:
            event = await outbox_crud.create_event(
                db,
                event_type="appointment.created",
                payload=json.dumps({"appointment_id": 888}),
                max_retries=2,
            )
            await db.commit()
            event_id = event.id

        # Attempt 1
        async with session_maker() as db:
            stats1 = await worker.process_batch(db, limit=10)
            assert stats1["failed"] == 1

        async with session_maker() as db:
            ev = await outbox_crud.get_by_id(db, event_id)
            assert ev.retry_count == 1
            assert ev.status == "FAILED"
            assert "[TimeoutError]" in ev.error_message

        # Attempt 2 (Max reached)
        async with session_maker() as db:
            stats2 = await worker.process_batch(db, limit=10)
            assert stats2["failed"] == 1

        async with session_maker() as db:
            ev = await outbox_crud.get_by_id(db, event_id)
            assert ev.retry_count == 2
            assert ev.status == "FAILED"

        # Attempt 3: Should NOT be claimed because retry_count >= max_retries
        async with session_maker() as db:
            stats3 = await worker.process_batch(db, limit=10)
            assert stats3["total"] == 0  # Not eligible for claiming

    asyncio.run(_run())


def test_outbox_multi_worker_concurrency_no_duplicate_claims(isolated_db):
    """Verify that multiple concurrent worker passes claim distinct non-overlapping events."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        # Seed 6 pending events
        async with session_maker() as db:
            for i in range(6):
                await outbox_crud.create_event(
                    db,
                    event_type="appointment.created",
                    payload=json.dumps({"appointment_id": 100 + i}),
                )
            await db.commit()

        claimed_w1 = []
        claimed_w2 = []

        async def worker_1():
            async with session_maker() as db:
                events = await outbox_crud.claim_events(db, limit=3)
                claimed_w1.extend([e.id for e in events])

        async def worker_2():
            async with session_maker() as db:
                events = await outbox_crud.claim_events(db, limit=3)
                claimed_w2.extend([e.id for e in events])

        # Run both worker claims
        await asyncio.gather(worker_1(), worker_2())

        # Verify no overlap between worker 1 and worker 2 claims
        overlap = set(claimed_w1).intersection(set(claimed_w2))
        assert len(overlap) == 0, f"Found duplicate claims across workers: {overlap}"
        assert len(claimed_w1) + len(claimed_w2) == 6

    asyncio.run(_run())
