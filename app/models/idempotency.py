from datetime import datetime
from sqlalchemy import DateTime, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint("tenant_id", "key", "operation", name="uq_idempotency_tenant_key_operation"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(50), default="default", index=True, nullable=False)
    key: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    operation: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA256 hex string
    status: Mapped[str] = mapped_column(String(20), default="COMPLETED", nullable=False)  # PENDING, COMPLETED, FAILED
    result_data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON serialized result
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
