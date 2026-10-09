from dataclasses import dataclass
import json
import logging
import re
from typing import AsyncGenerator

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_shared_client: httpx.AsyncClient | None = None


def get_shared_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
        timeout = httpx.Timeout(15.0, connect=3.0, read=6.0)
        try:
            # Check for h2 dependency before enabling HTTP/2
            import h2  # noqa: F401
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



def normalize_whisper_language(lang: str | None) -> str | None:
    """
    Maps supported languages strictly to ISO-639-1 ('en' or 'ur').
    Returns None for any unmapped/unknown value so the language param is omitted.
    """
    if not lang:
        return None
    cleaned = str(lang).strip().lower()
    if cleaned in ("en", "english"):
        return "en"
    if cleaned in ("ur", "urdu", "hi", "hindi"):
        return "ur"
    return None


KNOWN_HALLUCINATIONS = {
    "www.feyyaz.tv",
    "feyyaz.tv",
    "subtitles by",
    "amara.org",
    "subscribe to our channel",
    "please subscribe",
    "www.logic.com.au",
}

SHORT_GENERIC_HALLUCINATIONS = {
    "thank you",
    "thanks",
    "bye",
    "goodbye",
    "you",
    "oh",
    ".",
    "",
    "the end",
    "watching",
    "subscribe to",
    "thank you for watching",
    "thanks for watching",
}

WHISPER_PROMPT_HINT = (
    "Logic Pulse AI Receptionist: Software development, AI automation, chatbots, "
    "web apps, dashboards, appointments, office timings, Muhammad Ramzan. English and Urdu."
)
# FORMER_PROMPTS: Preserved strictly for post-STT echo and hallucination dropping.
# NEVER send these strings to Groq Whisper or any STT provider.
FORMER_PROMPTS = [
    "logic pulse ai receptionist: software development, ai automation, chatbots, web apps, dashboards, appointments, office timings, muhammad ramzan. english and urdu",
    "logic pulse ai receptionist: appointments, timings, services in english and urdu",
    "logic pulse ai receptionist",
    "english and urdu",
]


def is_prompt_echo(text: str, prompt: str | None = None) -> bool:
    """
    Evaluates whether a transcript is an echo/hallucination of the Whisper prompt hint
    or any former prompt text. Also drops transcripts that equal or are a prefix
    of former prompt texts (e.g. 'English and Urdu.', 'Logic Pulse AI Receptionist.').
    """
    if not text:
        return False

    text_clean = text.lower().strip().rstrip(".!?,;:")
    if not text_clean:
        return False

    # Exempt bare company name
    if text_clean in ("logic pulse", "logicpulse"):
        return False

    words = text_clean.split()

    # 1. Exact or prefix match against any former prompt text
    all_prompts = list(FORMER_PROMPTS)
    if prompt:
        p_clean = prompt.lower().strip().rstrip(".!?,;:")
        if p_clean and p_clean not in all_prompts:
            all_prompts.append(p_clean)

    words_clean_str = " ".join(re.findall(r"\w+", text_clean))
    for p in all_prompts:
        p_norm = p.lower().strip().rstrip(".!?,;:")
        if text_clean == p_norm:
            return True
        p_words_str = " ".join(re.findall(r"\w+", p_norm))
        # Prefix match requires 3+ words and at least 8 characters
        if len(words) >= 3 and len(text_clean) >= 8:
            if p_norm.startswith(text_clean) or p_words_str.startswith(words_clean_str):
                return True

    # 2. Longest contiguous phrase overlap (>=70% of 8+ word transcript)
    text_words = re.findall(r"\w+", text_clean)
    if len(text_words) < 8:
        return False

    for p in all_prompts:
        prompt_words = re.findall(r"\w+", p)
        if not prompt_words:
            continue

        max_len = 0
        prompt_len = len(prompt_words)
        text_len = len(text_words)

        for i in range(text_len):
            for j in range(prompt_len):
                k = 0
                while i + k < text_len and j + k < prompt_len and text_words[i + k] == prompt_words[j + k]:
                    k += 1
                if k > max_len:
                    max_len = k

        overlap_ratio = max_len / text_len
        if overlap_ratio >= 0.70:
            return True

    return False


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

        for attempt in range(2):
            client = get_shared_client()
            try:
                response = await client.post(url, json=payload, headers=headers)
                status = getattr(response, "status_code", 200)
                if isinstance(status, int) and status >= 400:
                    logger.warning("Groq API returned HTTP status_code=%d", status)
                    raise GroqServiceError(f"Groq API request failed with status {status}")
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
                if attempt == 0:
                    logger.warning("Groq connection failed, resetting shared client and retrying once: %s", type(exc).__name__)
                    await close_shared_client()
                    continue
                logger.warning("Groq connection failed on retry: %s", type(exc).__name__)
                raise GroqServiceError("Unable to connect to Groq API.") from exc
            except httpx.HTTPStatusError as exc:
                logger.warning("Groq API returned HTTP status_code=%d", exc.response.status_code)
                raise GroqServiceError(f"Groq API request failed with status {exc.response.status_code}.") from exc

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
        yielded_any = False

        for attempt in range(2):
            client = get_shared_client()
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    status = getattr(response, "status_code", 200)
                    if isinstance(status, int) and status >= 400:
                        logger.warning("Groq API streaming returned HTTP status_code=%d", status)
                        raise GroqServiceError(f"Groq API request failed with status {status}")
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
                                # Explicitly yield ONLY delta.content — never reasoning fields
                                content = delta.get("content")
                                if content:
                                    yielded_any = True
                                    yield content
                        except Exception:
                            continue
                return  # Success
            except GeneratorExit:
                return
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
                if attempt == 0 and not yielded_any:
                    logger.warning("Groq stream connection error, retrying once: %s", type(exc).__name__)
                    await close_shared_client()
                    continue
                logger.warning("Groq stream connection error on attempt %d: %s", attempt, type(exc).__name__)
                raise GroqServiceError(f"Groq API streaming failed: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                logger.warning("Groq API streaming HTTP error: status_code=%d", exc.response.status_code)
                raise GroqServiceError(f"Groq API streaming failed with status {exc.response.status_code}") from exc
            except Exception as exc:
                if isinstance(exc, GroqServiceError):
                    raise
                logger.warning("Groq streaming failed: %s", type(exc).__name__)
                raise GroqServiceError(f"Groq streaming failed: {exc}") from exc


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
        norm_lang = normalize_whisper_language(language)
        if norm_lang:
            data["language"] = norm_lang
        if prompt:
            data["prompt"] = prompt

        for attempt in range(2):
            client = get_shared_client()
            try:
                response = await client.post(url, headers=headers, files=files, data=data)
                status = getattr(response, "status_code", 200)

                # If Groq returns 400 and we sent a language param, retry once without language
                if status == 400 and "language" in data:
                    logger.warning(
                        "Groq Whisper rejected language param '%s' with 400. Retrying without language param...",
                        data.get("language"),
                    )
                    data.pop("language", None)
                    response = await client.post(url, headers=headers, files=files, data=data)
                    status = getattr(response, "status_code", 200)

                if isinstance(status, int) and status >= 400:
                    logger.warning("Groq Whisper API returned HTTP status_code=%d", status)
                    raise GroqServiceError(f"Groq Whisper transcription failed with status {status}")
                response.raise_for_status()
                res_json = response.json()
                text = res_json.get("text", "").strip()
                lang = res_json.get("language", "unknown")
                duration = float(res_json.get("duration", 0.0))

                # Segment-level hallucination and no-speech filtering
                segments = res_json.get("segments", [])
                avg_no_speech = 0.0
                avg_logprob = 0.0
                avg_compression = 1.0
                if segments:
                    avg_no_speech = sum(float(s.get("no_speech_prob", 0.0)) for s in segments) / len(segments)
                    avg_logprob = sum(float(s.get("avg_logprob", 0.0)) for s in segments) / len(segments)
                    avg_compression = sum(float(s.get("compression_ratio", 1.0)) for s in segments) / len(segments)

                self.last_segment_metrics = {
                    "no_speech_prob": avg_no_speech,
                    "avg_logprob": avg_logprob,
                    "compression_ratio": avg_compression,
                }
                drop_reason = None
                if segments and (avg_no_speech > 0.6 or avg_logprob < -1.1):
                    logger.info(
                        "Dropping hallucinated/no-speech transcript (avg_no_speech=%.2f, avg_logprob=%.2f): '%s'",
                        avg_no_speech,
                        avg_logprob,
                        text,
                    )
                    drop_reason = f"segment_filter (no_speech={avg_no_speech:.2f}, logprob={avg_logprob:.2f})"
                    text = ""

                # Known repetitive hallucination text filtering
                if text:
                    text_lower = text.lower().strip().rstrip(".!?,")

                    # 1. Whole-phrase / word-boundary matching for known static hallucinations
                    for h in KNOWN_HALLUCINATIONS:
                        if re.search(rf"(?:^|\b){re.escape(h)}(?:\b|$)", text_lower):
                            logger.info("Dropping known hallucination transcript: '%s'", text)
                            drop_reason = f"known_hallucination ('{h}')"
                            text = ""
                            break

                # 2. Short generic transcripts ("thank you", "bye", "you", ".", "oh", "thanks for watching")
                # Drop ONLY when audio is short (< 0.8s) OR segment confidence is poor (avg_no_speech > 0.35 or avg_logprob < -0.7).
                # A clear 3s sentence with good audio must never be dropped.
                if text:
                    text_lower = text.lower().strip().rstrip(".!?,")
                    if text_lower in SHORT_GENERIC_HALLUCINATIONS:
                        if text_lower in ("thank you", "thanks", "thank you very much"):
                            is_suspicious_audio = (duration < 1.3) or (avg_no_speech > 0.15) or (avg_logprob < -0.55)
                        else:
                            is_suspicious_audio = (duration < 0.8) or (avg_no_speech > 0.35) or (avg_logprob < -0.7)
                        if is_suspicious_audio:
                            logger.info(
                                "Dropping short generic transcript '%s' on suspicious/short audio (dur=%.2fs, no_speech=%.2f, logprob=%.2f)",
                                text,
                                duration,
                                avg_no_speech,
                                avg_logprob,
                            )
                            drop_reason = f"short_generic_low_confidence ('{text}')"
                            text = ""

                # 3. Prompt-echo guard
                if text and is_prompt_echo(text, prompt):
                    logger.info("Dropping prompt-echo hallucination transcript: '%s'", text)
                    drop_reason = f"prompt_echo_or_prefix ('{text}')"
                    text = ""

                self.last_drop_reason = drop_reason

                return text, lang, duration
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
                if attempt == 0:
                    logger.warning("Groq Whisper connection error, retrying once: %s", type(exc).__name__)
                    await close_shared_client()
                    continue
                logger.warning("Groq Whisper connection failed: %s", type(exc).__name__)
                raise GroqServiceError("Unable to connect to Groq Whisper API.") from exc
            except httpx.HTTPStatusError as exc:
                logger.warning("Groq Whisper HTTP error: status_code=%d", exc.response.status_code)
                raise GroqServiceError(f"Groq Whisper transcription failed with status {exc.response.status_code}") from exc
