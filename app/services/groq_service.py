from dataclasses import dataclass
import json
import logging
from typing import AsyncGenerator

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_shared_client: httpx.AsyncClient | None = None


def get_shared_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
        timeout = httpx.Timeout(30.0, connect=5.0)
        try:
            _shared_client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                http2=True,
            )
        except Exception:
            _shared_client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                http2=False,
            )
    return _shared_client


async def close_shared_client() -> None:
    global _shared_client
    if _shared_client is not None:
        try:
            if not _shared_client.is_closed:
                await _shared_client.aclose()
        except Exception:
            pass
        finally:
            _shared_client = None


class GroqServiceError(Exception):
    pass


@dataclass
class GroqCompletionResult:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


_singleton_instance: "GroqService | None" = None


class GroqService:
    def __new__(cls, *args, **kwargs):
        global _singleton_instance
        if _singleton_instance is None:
            _singleton_instance = super().__new__(cls)
            _singleton_instance._initialized = False
        return _singleton_instance

    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise GroqServiceError("GROQ_API_KEY is missing from environment variables.")
        if (
            getattr(self, "_initialized", False)
            and getattr(self, "api_key", None) == settings.groq_api_key
            and getattr(self, "model", None) == settings.groq_model
        ):
            return

        self.base_url = settings.groq_base_url.rstrip("/")
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model
        self._initialized = True

    def _prepare_payload(self, messages: list[dict[str, str]], stream: bool = False) -> dict:
        max_tokens = getattr(settings, "llm_max_tokens", 200)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        if self.model.startswith("openai/gpt-oss"):
            payload["max_tokens"] = 400
            payload["reasoning_effort"] = "low"
        if stream:
            payload["stream"] = True
        return payload

    async def chat_completion_with_usage(self, messages: list[dict[str, str]]) -> GroqCompletionResult:
        payload = self._prepare_payload(messages, stream=False)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        client = get_shared_client()

        try:
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
        payload = self._prepare_payload(messages, stream=True)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        client = get_shared_client()

        try:
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
                        choices = data.get("choices")
                        if choices:
                            delta = choices[0].get("delta", {})
                            # Yield ONLY delta.content (ignore reasoning fields)
                            content = delta.get("content")
                            if content:
                                yield content
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

        client = get_shared_client()
        try:
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




