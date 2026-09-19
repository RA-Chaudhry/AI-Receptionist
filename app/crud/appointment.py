from typing import Sequence
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appointment import Appointment
from app.schemas.tools import CreateAppointmentRequest


class AppointmentCRUD:
    async def create(
        self,
        db: AsyncSession,
        payload: CreateAppointmentRequest,
        user_id: int,
        commit: bool = True,
    ) -> Appointment:
        appointment = Appointment(
            user_id=user_id,
            tenant_id=payload.tenant_id,
            customer_name=payload.customer_name,
            customer_phone=payload.customer_phone,
            customer_email=payload.customer_email,
            service=payload.service,
            day=payload.day.strip().capitalize(),
            time_str=payload.time_str.strip().upper(),
            status="confirmed",
        )
        db.add(appointment)
        if commit:
            await db.commit()
            await db.refresh(appointment)
        else:
            await db.flush()
            await db.refresh(appointment)
        return appointment

    async def get_by_id(
        self,
        db: AsyncSession,
        appointment_id: int,
        tenant_id: str | None = None,
    ) -> Appointment | None:
        stmt = select(Appointment).where(Appointment.id == appointment_id)
        if tenant_id is not None:
            stmt = stmt.where(Appointment.tenant_id == tenant_id)
        result = await db.execute(stmt)
        return result.scalars().first()

    async def get_by_user_id(self, db: AsyncSession, user_id: int) -> Sequence[Appointment]:
        result = await db.execute(
            select(Appointment)
            .where(Appointment.user_id == user_id)
            .order_by(Appointment.created_at.desc())
        )
        return result.scalars().all()

    async def get_by_phone(
        self,
        db: AsyncSession,
        phone: str,
        tenant_id: str | None = None,
    ) -> Sequence[Appointment]:
        stmt = select(Appointment).where(Appointment.customer_phone == phone)
        if tenant_id is not None:
            stmt = stmt.where(Appointment.tenant_id == tenant_id)
        result = await db.execute(stmt.order_by(Appointment.created_at.desc()))
        return result.scalars().all()

    async def get_confirmed_by_slot(
        self,
        db: AsyncSession,
        day: str,
        time_str: str,
        tenant_id: str = "default",
    ) -> Appointment | None:
        canonical_day = day.strip().capitalize()
        canonical_time = time_str.strip().upper()
        result = await db.execute(
            select(Appointment).where(
                Appointment.day == canonical_day,
                Appointment.time_str == canonical_time,
                Appointment.tenant_id == tenant_id,
                Appointment.status != "cancelled",
            )
        )
        return result.scalars().first()

    async def update_status(
        self,
        db: AsyncSession,
        appointment_id: int,
        status: str,
        tenant_id: str | None = None,
        commit: bool = True,
    ) -> Appointment | None:
        appointment = await self.get_by_id(db, appointment_id, tenant_id=tenant_id)
        if not appointment:
            return None
        appointment.status = status
        if commit:
            await db.commit()
            await db.refresh(appointment)
        else:
            await db.flush()
            await db.refresh(appointment)
        return appointment

    async def reschedule(
        self,
        db: AsyncSession,
        appointment_id: int,
        new_day: str,
        new_time_str: str,
        tenant_id: str | None = None,
        commit: bool = True,
    ) -> Appointment | None:
        appointment = await self.get_by_id(db, appointment_id, tenant_id=tenant_id)
        if not appointment:
            return None
        appointment.day = new_day.strip().capitalize()
        appointment.time_str = new_time_str.strip().upper()
        appointment.status = "rescheduled"
        if commit:
            await db.commit()
            await db.refresh(appointment)
        else:
            await db.flush()
            await db.refresh(appointment)
        return appointment


appointment_crud = AppointmentCRUD()
