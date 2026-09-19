from dataclasses import dataclass
import json
import logging
from typing import AsyncGenerator

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class GroqServiceError(Exception):
    pass


@dataclass
class GroqCompletionResult:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


class GroqService:
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise GroqServiceError("GROQ_API_KEY is missing from environment variables.")

        self.base_url = settings.groq_base_url.rstrip("/")
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model

    async def chat_completion_with_usage(self, messages: list[dict[str, str]]) -> GroqCompletionResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.4,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.exception("Groq API returned an error")
            raise GroqServiceError("Groq API request failed.") from exc
        except httpx.RequestError as exc:
            logger.exception("Could not reach Groq API")
            raise GroqServiceError("Unable to connect to Groq API.") from exc

        try:
            content = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            model_used = data.get("model", self.model)
            return GroqCompletionResult(
                content=content,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model_used,
            )
        except (KeyError, IndexError, AttributeError) as exc:
            raise GroqServiceError("Unexpected response shape from Groq API.") from exc

    async def chat_completion(self, messages: list[dict[str, str]]) -> str:
        result = await self.chat_completion_with_usage(messages)
        return result.content

    async def chat_completion_stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.4,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue
        except GeneratorExit:
            return
        except httpx.HTTPStatusError as exc:
            logger.exception("Groq API streaming returned an error")
            raise GroqServiceError("Groq API streaming failed.") from exc
        except httpx.RequestError as exc:
            logger.exception("Could not reach Groq API for streaming")
            raise GroqServiceError("Unable to connect to Groq API streaming.") from exc

    async def transcribe_audio_with_usage(
        self,
        wav_bytes: bytes,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
    ) -> tuple[str, str, float]:
        """
        Transcribes audio using Groq Cloud Whisper API.
        Returns: (transcript_text, detected_language, audio_duration_seconds)
        """
        model = model or getattr(settings, "groq_whisper_model", "whisper-large-v3-turbo")
        url = f"{self.base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data: dict[str, str] = {
            "model": model,
            "response_format": "verbose_json",
            "temperature": "0.0",
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        else:
            data["prompt"] = "Logic Pulse AI Receptionist: Appointments, timings, services in English, Urdu, Roman Urdu."

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                res_json = response.json()
                text = res_json.get("text", "").strip()
                lang = res_json.get("language", "unknown")
                duration = float(res_json.get("duration", 0.0))
                return text, lang, duration
        except httpx.HTTPStatusError as exc:
            logger.exception("Groq Whisper API returned an error")
            raise GroqServiceError("Groq Whisper transcription failed.") from exc
        except httpx.RequestError as exc:
            logger.exception("Could not reach Groq Whisper API")
            raise GroqServiceError("Unable to connect to Groq Whisper API.") from exc



