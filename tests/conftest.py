"""
tests/conftest.py — Global Test Fixtures & Reusable Fakes for AI Receptionist
Provides fully offline, isolated mock providers for STT, LLM, TTS, Vonage, and Database.
"""

from dataclasses import dataclass
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.services.groq_service import GroqCompletionResult


# ──────────────────────────────────────────────────────────────────────────────
# 1. Reusable Fake / Mock Providers
# ──────────────────────────────────────────────────────────────────────────────

class FakeGroqService:
    """
    Offline mock for Groq API / Provider Manager.
    Allows configuring custom responses or error raises.
    """
    def __init__(
        self,
        default_reply: str = "This is a mock assistant response.",
        input_tokens: int = 25,
        output_tokens: int = 15,
        should_fail: bool = False,
    ) -> None:
        self.default_reply = default_reply
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.should_fail = should_fail
        self.call_history: List[List[Dict[str, str]]] = []

    async def chat_completion_with_usage(
        self,
        messages: List[Dict[str, str]],
    ) -> GroqCompletionResult:
        self.call_history.append(messages)
        if self.should_fail:
            raise RuntimeError("Mocked Groq API failure")

        # Context-aware dynamic mock responses
        content = self.default_reply
        if messages:
            last_msg = messages[-1].get("content", "")
            system_msg = messages[0].get("content", "") if messages[0].get("role") == "system" else ""

            if "review" in system_msg.lower() or "review" in last_msg.lower():
                content = "Thank you for your feedback! We are delighted to assist you."
            elif "dental" in last_msg.lower():
                content = "We can build custom multi-tenant cloud software for clinics."

        return GroqCompletionResult(
            content=content,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            model="llama-3.1-8b-instant",
        )

    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        res = await self.chat_completion_with_usage(messages)
        return res.content

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        res = await self.chat_completion_with_usage(messages)
        words = res.content.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.001)


@dataclass
class FakeWhisperSegment:
    text: str


class FakeWhisperModel:
    """
    Offline mock for Faster-Whisper.
    Returns simulated speech transcription without loading neural network weights.
    """
    def __init__(self, transcript: str = "Hello, I want to book an appointment.") -> None:
        self.transcript = transcript

    def transcribe(self, audio_file, **kwargs):
        segments = [FakeWhisperSegment(text=self.transcript)]
        info = MagicMock()
        info.language = "en"
        info.duration = 1.5
        return segments, info


class FakeTTSCommunicate:
    """
    Offline mock for edge-tts Communicate stream.
    Streams pre-defined synthetic PCM/MP3 audio chunks without external network calls.
    """
    def __init__(self, text: str, voice: str = "en-US-JennyNeural") -> None:
        self.text = text
        self.voice = voice

    async def stream(self) -> AsyncGenerator[Dict[str, bytes], None]:
        # Yield 2 synthetic audio chunks (16kHz 16-bit mono PCM dummy bytes)
        yield {"type": "audio", "data": b"\x00\x00" * 320}
        yield {"type": "audio", "data": b"\x00\x00" * 320}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Database Isolation & Client Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_db(tmp_path: Path):
    """
    Provides a completely isolated SQLite database per test.
    Automatically creates all tables, binds the sessionmaker, and cleans up after test.
    """
    db_file = tmp_path / "test_isolated.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"
    engine = create_async_engine(db_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # Initialize schema
    async def _init():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init())

    # Patch voice_service sessionmaker as well so voice websocket uses the test DB
    with patch("app.services.voice_service.async_session_maker", session_maker), \
         patch("app.db.session.async_session_maker", session_maker):

        async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
            async with session_maker() as session:
                yield session

        app.dependency_overrides[get_db] = _override_get_db

        old_env = settings.environment
        settings.environment = "testing"

        yield {
            "engine": engine,
            "session_maker": session_maker,
            "db_path": db_file,
        }

        settings.environment = old_env
        app.dependency_overrides.clear()

        # Teardown
        async def _cleanup():
            await engine.dispose()

        asyncio.run(_cleanup())


@pytest.fixture
def client(isolated_db):
    """
    FastAPI TestClient bound to the isolated database fixture.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_groq(monkeypatch):
    """
    Patches GroqService across all services with FakeGroqService.
    """
    fake = FakeGroqService()
    monkeypatch.setattr("app.services.chat_service.GroqService", lambda: fake)
    monkeypatch.setattr("app.services.ai_service.GroqService", lambda: fake)
    monkeypatch.setattr("app.services.review_service.GroqService", lambda: fake)
    return fake


@pytest.fixture
def mock_voice_environment(monkeypatch):
    """
    Patches STT (Whisper), TTS (Edge-TTS / pyttsx3), and VAD so all voice tests run offline.
    """
    fake_whisper = FakeWhisperModel()
    monkeypatch.setattr("app.services.voice_service._whisper_model", fake_whisper)
    monkeypatch.setattr("edge_tts.Communicate", FakeTTSCommunicate)
    return {
        "whisper": fake_whisper,
    }
