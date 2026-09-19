"""
tests/test_state_and_database.py — Database CRUD, User State, & History Window Tests
"""

import asyncio
import pytest
from sqlalchemy import select

from app.crud.user import user_crud
from app.crud.chat import chat_crud
from app.models.user import User
from app.models.chat import ChatMessage


def test_user_crud_lifecycle(isolated_db):
    """Test creating, fetching, and updating user records."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            # 1. Create user
            user1 = await user_crud.create(db, name="Zubair", email="zubair@example.com")
            assert user1.id is not None
            assert user1.name == "Zubair"
            assert user1.email == "zubair@example.com"

            # 2. Get by ID & email
            fetched = await user_crud.get_by_id(db, user1.id)
            assert fetched is not None
            assert fetched.name == "Zubair"

            by_email = await user_crud.get_by_email(db, "zubair@example.com")
            assert by_email is not None
            assert by_email.id == user1.id

            # 3. get_or_create with name update
            updated = await user_crud.get_or_create(db, email="zubair@example.com", name="Zubair Ahmed")
            assert updated.id == user1.id
            assert updated.name == "Zubair Ahmed"

    asyncio.run(_run())


def test_chat_crud_and_memory_window(isolated_db):
    """Test message creation, recent memory limit window, and history retrieval."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Bilal", email="bilal@example.com")

            # Create 8 messages (4 turns)
            for i in range(1, 9):
                role = "user" if i % 2 != 0 else "assistant"
                await chat_crud.create_message(db, user_id=user.id, role=role, content=f"Message {i}")

            # 1. Full history should return all 8 messages in ASC order
            history = await chat_crud.get_history_for_user(db, user.id)
            assert len(history) == 8
            assert history[0].content == "Message 1"
            assert history[-1].content == "Message 8"

            # 2. Recent messages with limit=4 should return the last 4 messages in chronological order
            recent = await chat_crud.get_recent_messages(db, user_id=user.id, limit=4)
            assert len(recent) == 4
            assert recent[0].content == "Message 5"
            assert recent[1].content == "Message 6"
            assert recent[2].content == "Message 7"
            assert recent[3].content == "Message 8"

    asyncio.run(_run())


def test_user_list_aggregated_chat_count(isolated_db):
    """Verify list_users correctly computes chat_count per user."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            u1 = await user_crud.create(db, name="User One", email="u1@example.com")
            u2 = await user_crud.create(db, name="User Two", email="u2@example.com")
            u3 = await user_crud.create(db, name="User Three", email="u3@example.com")

            # u1 has 3 user messages, u2 has 1 user message, u3 has 0
            await chat_crud.create_message(db, user_id=u1.id, role="user", content="Msg 1")
            await chat_crud.create_message(db, user_id=u1.id, role="assistant", content="Reply 1")
            await chat_crud.create_message(db, user_id=u1.id, role="user", content="Msg 2")
            await chat_crud.create_message(db, user_id=u1.id, role="user", content="Msg 3")

            await chat_crud.create_message(db, user_id=u2.id, role="user", content="Msg A")

            users = await user_crud.list_users(db)
            counts = {u.email: getattr(u, "chat_count", 0) for u in users}

            assert counts["u1@example.com"] == 3
            assert counts["u2@example.com"] == 1
            assert counts["u3@example.com"] == 0

    asyncio.run(_run())


def test_cascade_delete_messages_on_user_delete(isolated_db):
    """Verify deleting a user cascades and removes all associated chat messages."""
    session_maker = isolated_db["session_maker"]

    async def _run():
        async with session_maker() as db:
            user = await user_crud.create(db, name="Temp User", email="temp@example.com")
            await chat_crud.create_message(db, user_id=user.id, role="user", content="Hi")
            await chat_crud.create_message(db, user_id=user.id, role="assistant", content="Hello")

            # Delete user
            await db.delete(user)
            await db.commit()

            # Verify chat_messages are deleted
            result = await db.execute(select(ChatMessage).where(ChatMessage.user_id == user.id))
            messages = result.scalars().all()
            assert len(messages) == 0

    asyncio.run(_run())
