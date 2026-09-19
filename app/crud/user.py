from sqlalchemy import case, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage
from app.models.user import User


class UserCRUD:
    async def get_by_id(self, db: AsyncSession, user_id: int) -> User | None:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, db: AsyncSession, email: str) -> User | None:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def create(
        self,
        db: AsyncSession,
        *,
        name: str,
        email: str | None = None,
    ) -> User:
        user = User(name=name, email=email)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    async def list_users(self, db: AsyncSession) -> list[User]:
        result = await db.execute(
            select(
                User,
                func.sum(case((ChatMessage.role == "user", 1), else_=0)).label("chat_count"),
            )
            .outerjoin(ChatMessage, User.id == ChatMessage.user_id)
            .group_by(User.id)
            .order_by(User.created_at.desc())
        )
        users: list[User] = []
        for user, chat_count in result.all():
            setattr(user, "chat_count", int(chat_count or 0))
            users.append(user)
        return users

    async def get_or_create(
        self,
        db: AsyncSession,
        *,
        user_id: int | None = None,
        name: str | None = None,
        email: str | None = None,
    ) -> User:
        if user_id is not None:
            user = await self.get_by_id(db, user_id)
            if user:
                return user

        if email:
            user = await self.get_by_email(db, email)
            if user:
                if name and user.name != name:
                    user.name = name
                    await db.commit()
                    await db.refresh(user)
                return user

        if not name:
            name = "Guest"
        try:
            return await self.create(db, name=name, email=email)
        except IntegrityError:
            # Concurrent first contact for the same identity is expected under
            # multiple workers.  The database unique constraint is canonical.
            await db.rollback()
            if email:
                user = await self.get_by_email(db, email)
                if user:
                    return user
            raise


user_crud = UserCRUD()
