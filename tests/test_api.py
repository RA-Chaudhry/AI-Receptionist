import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app


class FakeChatGroq:
    async def chat_completion(self, messages):
        if messages and any(kw in messages[0]["content"] for kw in ["professional Google review replies", "professional response writer"]):
            return '"Thank you for your kind review! We appreciate your support and look forward to serving you again."'
        return "Hello! I can help with bookings, services, and general questions."



@pytest.fixture()
def client(tmp_path: Path):
    db_path = tmp_path / "test_ai_receptionist.db"
    database_url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(database_url)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def init_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(init_db())

    async def override_get_db():
        async with session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    old_env = settings.environment
    settings.environment = "testing"

    with TestClient(app) as test_client:
        yield test_client

    settings.environment = old_env
    app.dependency_overrides.clear()
    asyncio.run(engine.dispose())


def test_root_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Receptionist Dashboard" in response.text


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_voice_health_endpoint(client):
    response = client.get("/api/v1/voice/health")
    assert response.status_code == 200
    body = response.json()
    assert "whisper_loaded" in body
    assert "vad_loaded" in body
    assert "tts_available" in body
    assert body["websocket_endpoint"] == "/api/v1/voice/stream"


def test_chat_flow_persists_user_and_history(client, monkeypatch):
    monkeypatch.setattr("app.services.chat_service.GroqService", lambda: FakeChatGroq())

    response = client.post(
        "/api/v1/chat",
        json={
            "user_name": "Ali",
            "user_email": "ali@example.com",
            "message": "I need help booking an appointment.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user"]["name"] == "Ali"
    # The local router handles appointment queries — reply must mention appointment context
    reply = body["assistant_reply"].lower()
    assert any(w in reply for w in ["appointment", "schedule", "muhammad ramzan", "day", "time"])

    users = client.get("/api/v1/users").json()
    assert len(users) == 1
    assert users[0]["chat_count"] == 1

    history = client.get("/api/v1/users/1/history").json()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_review_reply_strips_quotes(client, monkeypatch):
    monkeypatch.setattr("app.services.review_service.GroqService", lambda: FakeChatGroq())

    response = client.post(
        "/api/v1/review-reply",
        json={"review_text": "Great service and friendly staff."},
    )
    assert response.status_code == 200
    assert response.json()["reply"].startswith("Thank you")
    assert '"' not in response.json()["reply"]
