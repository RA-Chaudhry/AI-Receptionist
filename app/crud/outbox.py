from datetime import datetime, timedelta, timezone
import logging
from uuid import uuid4
from typing import Sequence
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outbox import OutboxEvent

logger = logging.getLogger(__name__)


class OutboxCRUD:
    async def create_event(
        self,
        db: AsyncSession,
        event_type: str,
        payload: str,
        status: str = "PENDING",
        max_retries: int = 3,
    ) -> OutboxEvent:
        event = OutboxEvent(
            event_type=event_type,
            payload=payload,
            status=status,
            max_retries=max_retries,
            retry_count=0,
            next_attempt_at=datetime.now(timezone.utc),
        )
        db.add(event)
        return event

    async def get_by_id(self, db: AsyncSession, event_id: int) -> OutboxEvent | None:
        stmt = select(OutboxEvent).where(OutboxEvent.id == event_id)
        result = await db.execute(stmt)
        return result.scalars().first()

    async def claim_events(
        self,
        db: AsyncSession,
        limit: int = 50,
        stale_timeout_seconds: int = 300,
    ) -> list[OutboxEvent]:
        """
        Atomically claims a batch of pending, retryable failed, or stale processing events.
        Uses optimistic status matching so concurrent workers never double-claim the same event.
        """
        now = datetime.now(timezone.utc)
        stale_cutoff = now - timedelta(seconds=stale_timeout_seconds)

        # Build candidate selection query
        stmt = (
            select(OutboxEvent)
            .where(
                (OutboxEvent.status == "PENDING")
                | (
                    (OutboxEvent.status == "FAILED")
                    & (OutboxEvent.retry_count < OutboxEvent.max_retries)
                    & (
                        (OutboxEvent.next_attempt_at.is_(None))
                        | (OutboxEvent.next_attempt_at <= now)
                    )
                )
                | (
                    (OutboxEvent.status == "PROCESSING")
                    & (OutboxEvent.processing_started_at.is_not(None))
                    & (OutboxEvent.processing_started_at <= stale_cutoff)
                    & (OutboxEvent.retry_count < OutboxEvent.max_retries)
                )
            )
            .order_by(OutboxEvent.created_at.asc())
            .limit(limit * 2)  # Over-fetch candidate pool to fulfill limit under contention
        )

        # Dialect awareness: apply row lock in PostgreSQL
        bind = db.bind
        if bind is not None and "postgresql" in bind.dialect.name.lower():
            stmt = stmt.with_for_update(skip_locked=True)

        result = await db.execute(stmt)
        candidates = list(result.scalars().all())

        claimed_events: list[OutboxEvent] = []
        for candidate in candidates:
            if len(claimed_events) >= limit:
                break

            # Optimistic atomic update condition
            was_stale_processing = candidate.status == "PROCESSING"
            condition = (OutboxEvent.id == candidate.id) & (OutboxEvent.status == candidate.status)
            if was_stale_processing:
                update_vals = {
                    "status": "PROCESSING",
                    "processing_started_at": now,
                    "retry_count": candidate.retry_count + 1,
                    "claim_token": str(uuid4()),
                }
            else:
                update_vals = {
                    "status": "PROCESSING",
                    "processing_started_at": now,
                    "claim_token": str(uuid4()),
                }

            update_stmt = (
                update(OutboxEvent)
                .where(condition)
                .values(**update_vals)
            )
            update_res = await db.execute(update_stmt)
            if update_res.rowcount > 0:
                if was_stale_processing:
                    logger.warning(
                        "Recovered stale outbox event #%d (attempt %d/%d) after exceeding timeout of %ds",
                        candidate.id,
                        candidate.retry_count + 1,
                        candidate.max_retries,
                        stale_timeout_seconds,
                    )
                claimed_events.append(candidate)

        if claimed_events:
            await db.commit()
            for ev in claimed_events:
                await db.refresh(ev)

        return claimed_events

    async def get_pending_or_retryable_events(
        self,
        db: AsyncSession,
        limit: int = 50,
        stale_timeout_seconds: int = 300,
    ) -> Sequence[OutboxEvent]:
        """Read-only query for pending/retryable events without claiming."""
        now = datetime.now(timezone.utc)
        stale_cutoff = now - timedelta(seconds=stale_timeout_seconds)
        stmt = (
            select(OutboxEvent)
            .where(
                (OutboxEvent.status == "PENDING")
                | (
                    (OutboxEvent.status == "FAILED")
                    & (OutboxEvent.retry_count < OutboxEvent.max_retries)
                    & (
                        (OutboxEvent.next_attempt_at.is_(None))
                        | (OutboxEvent.next_attempt_at <= now)
                    )
                )
                | (
                    (OutboxEvent.status == "PROCESSING")
                    & (OutboxEvent.processing_started_at.is_not(None))
                    & (OutboxEvent.processing_started_at <= stale_cutoff)
                    & (OutboxEvent.retry_count < OutboxEvent.max_retries)
                )
            )
            .order_by(OutboxEvent.created_at.asc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return result.scalars().all()

    async def mark_processing(self, db: AsyncSession, event_id: int) -> OutboxEvent | None:
        event = await self.get_by_id(db, event_id)
        if event:
            event.status = "PROCESSING"
            event.processing_started_at = datetime.now(timezone.utc)
            event.claim_token = str(uuid4())
            await db.commit()
            await db.refresh(event)
        return event

    async def mark_completed(
        self,
        db: AsyncSession,
        event_id: int,
        claim_token: str | None = None,
    ) -> OutboxEvent | None:
        stmt = update(OutboxEvent).where(
            OutboxEvent.id == event_id,
            OutboxEvent.status == "PROCESSING",
        )
        if claim_token is not None:
            stmt = stmt.where(OutboxEvent.claim_token == claim_token)
        result = await db.execute(
            stmt.values(
                status="COMPLETED",
                processed_at=datetime.now(timezone.utc),
                claim_token=None,
            )
        )
        if result.rowcount == 0:
            await db.rollback()
            return None
        await db.commit()
        return await self.get_by_id(db, event_id)

    async def mark_failed(
        self,
        db: AsyncSession,
        event_id: int,
        error_message: str,
        claim_token: str | None = None,
        retry_base_seconds: int = 0,
        retry_max_seconds: int = 300,
    ) -> OutboxEvent | None:
        event = await self.get_by_id(db, event_id)
        if event is None or event.status != "PROCESSING":
            return None
        if claim_token is not None and event.claim_token != claim_token:
            return None

        next_retry_count = event.retry_count + 1
        delay_seconds = min(
            max(0, retry_base_seconds) * (2 ** max(0, next_retry_count - 1)),
            max(0, retry_max_seconds),
        )
        event.retry_count = next_retry_count
        event.status = "FAILED"
        event.error_message = error_message
        event.next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        event.claim_token = None
        await db.commit()
        await db.refresh(event)
        return event


outbox_crud = OutboxCRUD()
