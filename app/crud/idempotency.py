from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.idempotency import IdempotencyRecord


class IdempotencyCRUD:
    async def get_record(
        self,
        db: AsyncSession,
        tenant_id: str,
        key: str,
        operation: str,
    ) -> IdempotencyRecord | None:
        stmt = select(IdempotencyRecord).where(
            IdempotencyRecord.tenant_id == tenant_id,
            IdempotencyRecord.key == key,
            IdempotencyRecord.operation == operation,
        )
        result = await db.execute(stmt)
        return result.scalars().first()

    async def create_record(
        self,
        db: AsyncSession,
        tenant_id: str,
        key: str,
        operation: str,
        request_hash: str,
        status: str = "COMPLETED",
        result_data: str | None = None,
    ) -> IdempotencyRecord:
        record = IdempotencyRecord(
            tenant_id=tenant_id,
            key=key,
            operation=operation,
            request_hash=request_hash,
            status=status,
            result_data=result_data,
        )
        db.add(record)
        # We don't commit here if it is part of a broader transaction; caller commits
        return record

    async def update_record(
        self,
        db: AsyncSession,
        record_id: int,
        status: str,
        result_data: str | None = None,
    ) -> IdempotencyRecord | None:
        stmt = select(IdempotencyRecord).where(IdempotencyRecord.id == record_id)
        result = await db.execute(stmt)
        record = result.scalars().first()
        if record:
            record.status = status
            if result_data is not None:
                record.result_data = result_data
        return record


idempotency_crud = IdempotencyCRUD()
