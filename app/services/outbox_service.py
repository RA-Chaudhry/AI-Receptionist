import asyncio
import json
import logging
import time
from typing import Any, Callable, Coroutine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.crud.outbox import outbox_crud
from app.core.config import settings
from app.models.outbox import OutboxEvent

logger = logging.getLogger(__name__)


# Default mock asynchronous workers for notifications and CRM
async def default_sms_handler(payload: dict[str, Any]) -> None:
    logger.info(
        "Outbox SMS dispatched successfully | appointment_id=%s",
        payload.get("appointment_id"),
    )


async def default_crm_sync_handler(payload: dict[str, Any]) -> None:
    logger.info(
        "Outbox CRM sync completed successfully | appointment_id=%s",
        payload.get("appointment_id"),
    )


class OutboxWorker:
    """
    Durable asynchronous worker that processes outbox events transactionally created in the DB.
    Guarantees at-least-once delivery, retry with backoff tracking, and isolation from the live voice call path.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]]] = {
            "appointment.created": [default_sms_handler, default_crm_sync_handler],
            "appointment.cancelled": [default_sms_handler, default_crm_sync_handler],
            "appointment.rescheduled": [default_sms_handler, default_crm_sync_handler],
        }

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Registers a custom handler for a given event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def set_handlers_for_testing(
        self,
        event_type: str,
        handlers: list[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]],
    ) -> None:
        """Allows test fixtures to completely override handlers for verification."""
        self._handlers[event_type] = handlers

    async def process_event(self, db: AsyncSession, event: OutboxEvent) -> bool:
        """
        Processes a single claimed outbox event with registered handlers.
        Produces structured logs without logging PII or sensitive secrets.
        """
        t0 = time.monotonic()
        event_id = event.id
        event_type = event.event_type
        attempt = event.retry_count + 1

        try:
            payload = json.loads(event.payload)
            handlers = self._handlers.get(event_type, [])

            for handler in handlers:
                await asyncio.wait_for(
                    handler(payload),
                    timeout=settings.outbox_handler_timeout_seconds,
                )

            completed = await outbox_crud.mark_completed(db, event_id, claim_token=event.claim_token)
            if completed is None:
                logger.warning(
                    "Outbox event claim was lost before completion | event_id=%d | event_type=%s",
                    event_id,
                    event_type,
                )
                return False
            duration_ms = round((time.monotonic() - t0) * 1000, 2)
            logger.info(
                "Outbox event completed | event_id=%d | event_type=%s | transition=PROCESSING->COMPLETED | attempt=%d | duration_ms=%.2f",
                event_id,
                event_type,
                attempt,
                duration_ms,
            )
            return True
        except Exception as exc:
            duration_ms = round((time.monotonic() - t0) * 1000, 2)
            error_category = type(exc).__name__
            err_msg = str(exc)
            formatted_error = f"[{error_category}] {err_msg}"
            
            logger.warning(
                "Outbox event failed | event_id=%d | event_type=%s | transition=PROCESSING->FAILED | attempt=%d/%d | error_category=%s | duration_ms=%.2f",
                event_id,
                event_type,
                attempt,
                event.max_retries,
                error_category,
                duration_ms,
            )
            failed = await outbox_crud.mark_failed(
                db,
                event_id,
                formatted_error,
                claim_token=event.claim_token,
                retry_base_seconds=(
                    0 if getattr(settings, "environment", "") == "testing"
                    else settings.outbox_retry_base_seconds
                ),
                retry_max_seconds=settings.outbox_retry_max_seconds,
            )
            if failed is None:
                logger.warning(
                    "Outbox event claim was lost before failure recording | event_id=%d | event_type=%s",
                    event_id,
                    event_type,
                )
            return False

    async def process_batch(
        self,
        db: AsyncSession,
        limit: int = 50,
        stale_timeout_seconds: int = 300,
    ) -> dict[str, int]:
        """
        Atomically claims a batch of pending/retryable/stale events and processes them.
        """
        claimed_events = await outbox_crud.claim_events(
            db,
            limit=limit,
            stale_timeout_seconds=stale_timeout_seconds,
        )
        stats = {"total": len(claimed_events), "succeeded": 0, "failed": 0}

        for event in claimed_events:
            success = await self.process_event(db, event)
            if success:
                stats["succeeded"] += 1
            else:
                stats["failed"] += 1

        return stats


outbox_worker = OutboxWorker()


class OutboxRunner:
    """
    Continuous, durable background runner that polls and dispatches outbox events.
    Operates independently of the HTTP request lifecycle with graceful shutdown support.
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        worker: OutboxWorker | None = None,
        poll_interval_seconds: float = 2.0,
        batch_size: int = 20,
        stale_timeout_seconds: int = 300,
    ) -> None:
        self.session_maker = session_maker
        self.worker = worker or outbox_worker
        self.poll_interval_seconds = poll_interval_seconds
        self.batch_size = batch_size
        self.stale_timeout_seconds = stale_timeout_seconds
        self._shutdown_event = asyncio.Event()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def stop(self) -> None:
        """Signals the runner to stop polling gracefully."""
        logger.info("OutboxRunner shutdown requested.")
        self._shutdown_event.set()

    async def run_once(self) -> dict[str, int]:
        """Executes a single processing iteration against a clean DB session."""
        async with self.session_maker() as db:
            return await self.worker.process_batch(
                db,
                limit=self.batch_size,
                stale_timeout_seconds=self.stale_timeout_seconds,
            )

    async def run_continuous(self) -> None:
        """
        Continuously polls for and processes outbox events until shutdown is requested.
        Guarantees non-busy sleep on empty queue and resilience to unexpected batch errors.
        """
        self._is_running = True
        self._shutdown_event.clear()
        logger.info(
            "OutboxRunner started | poll_interval=%.1fs | batch_size=%d | stale_timeout=%ds",
            self.poll_interval_seconds,
            self.batch_size,
            self.stale_timeout_seconds,
        )

        try:
            while not self._shutdown_event.is_set():
                try:
                    stats = await self.run_once()
                    if stats["total"] == 0:
                        # Queue is empty: non-busy sleep waiting for next poll or shutdown
                        try:
                            await asyncio.wait_for(
                                self._shutdown_event.wait(),
                                timeout=self.poll_interval_seconds,
                            )
                        except asyncio.TimeoutError:
                            pass
                    else:
                        # Work was processed: yield control briefly before next batch
                        if not self._shutdown_event.is_set():
                            await asyncio.sleep(0.01)

                except Exception as exc:
                    logger.exception(
                        "Unexpected error during OutboxRunner batch execution: %s",
                        str(exc),
                    )
                    # Isolate error so runner does not permanently crash
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self.poll_interval_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass

        finally:
            self._is_running = False
            logger.info("OutboxRunner stopped cleanly.")
