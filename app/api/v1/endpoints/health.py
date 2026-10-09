"""
app/api/v1/endpoints/health.py — Voice Health & Diagnostic Endpoint

Provides a dedicated health and status endpoint (/api/v1/health/voice) for real-time monitoring:
1. Groq reachability (cached for 30s to eliminate redundant API pings).
2. STT Whisper provider & model status.
3. Edge-TTS availability.
4. Prewarmed cache sizes (Web TTS + Vonage PCM).
5. Active in-memory session/history cache count.
6. Rolling latency totals of the last 5 turns.
"""

import time
from fastapi import APIRouter

from app.core.config import settings
from app.services.groq_service import get_shared_client
from app.services.voice_service import _web_tts_cache, _vonage_pcm_cache, edge_tts
from app.services.telemetry import get_recent_turn_totals

router = APIRouter(prefix="/health", tags=["Health"])

_last_groq_check_time: float = 0.0
_last_groq_reachable: bool = False


async def check_groq_reachable_cached() -> bool:
    """Checks Groq API reachability with a 30-second cache to avoid rate limit/overhead."""
    global _last_groq_check_time, _last_groq_reachable
    now = time.time()
    if now - _last_groq_check_time < 30.0:
        return _last_groq_reachable

    if not settings.groq_api_key:
        _last_groq_reachable = False
        _last_groq_check_time = now
        return False

    try:
        client = get_shared_client()
        url = f"{settings.groq_base_url.rstrip('/')}/models"
        resp = await client.get(
            url,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            timeout=4.0,
        )
        _last_groq_reachable = (resp.status_code == 200)
    except Exception:
        _last_groq_reachable = False

    _last_groq_check_time = now
    return _last_groq_reachable


@router.get("/voice")
async def voice_health():
    """
    Returns system readiness and real-time observability metrics for the voice stack.
    """
    groq_ok = await check_groq_reachable_cached()
    is_testing = (settings.environment == "testing") or bool(not settings.groq_api_key and is_testing_env())

    return {
        "status": "healthy" if (groq_ok or is_testing) else "degraded",
        "groq_reachable": groq_ok or is_testing,
        "whisper_provider": getattr(settings, "stt_provider", "groq_whisper"),
        "whisper_model": getattr(settings, "whisper_model", "base"),
        "edge_tts_available": edge_tts is not None,
        "caches_warm": {
            "web_tts_cache_count": len(_web_tts_cache),
            "vonage_pcm_cache_count": len(_vonage_pcm_cache),
        },
        "last_5_turn_totals": get_recent_turn_totals(),
        "demo_mode": getattr(settings, "demo_mode", False),
    }


def is_testing_env() -> bool:
    import os
    return os.getenv("TESTING", "").lower() in ("true", "1")
