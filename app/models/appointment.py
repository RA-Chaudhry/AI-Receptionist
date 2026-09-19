from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Index, String, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Appointment(Base):
    __tablename__ = "appointments"
    __table_args__ = (
        # A cancelled appointment releases its slot; every other state remains an
        # active reservation.  This is the authoritative double-booking guard.
        Index(
            "uq_appointments_active_tenant_slot",
            "tenant_id",
            "day",
            "time_str",
            unique=True,
            postgresql_where=text("status <> 'cancelled'"),
            sqlite_where=text("status <> 'cancelled'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(50), default="default", index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    customer_name: Mapped[str] = mapped_column(String(100), nullable=False)
    customer_phone: Mapped[str] = mapped_column(String(30), index=True, nullable=False)
    customer_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    service: Mapped[str] = mapped_column(String(100), default="General Consultation", nullable=False)
    day: Mapped[str] = mapped_column(String(20), nullable=False)
    time_str: Mapped[str] = mapped_column(String(20), nullable=False)
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="confirmed", nullable=False)
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

    user: Mapped["User"] = relationship(back_populates="appointments")
