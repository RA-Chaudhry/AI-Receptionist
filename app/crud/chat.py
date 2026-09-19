from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage


class ChatCRUD:
    async def create_message(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        role: str,
        content: str,
    ) -> ChatMessage:
        message = ChatMessage(user_id=user_id, role=role, content=content)
        db.add(message)
        await db.commit()
        await db.refresh(message)
        return message

    async def get_recent_messages(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        limit: int,
    ) -> list[ChatMessage]:
        query = (
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
        result = await db.execute(query)
        messages = list(result.scalars().all())
        messages.reverse()
        return messages

    async def get_history_for_user(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> list[ChatMessage]:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        )
        return list(result.scalars().all())


chat_crud = ChatCRUD()

