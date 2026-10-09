import logging

from fastapi import APIRouter, Query, WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.services.voice_service import get_models_status, process_voice_websocket
from app.core.security import validate_voice_websocket_token

router = APIRouter(prefix="/voice", tags=["Voice WebSocket"])

logger = logging.getLogger(__name__)


@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    phone_number: str | None = Query(default=None, description="User phone number"),
    token: str | None = Query(default=None, description="Voice provider stream token"),
):
    if not validate_voice_websocket_token(token):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    logger.info("Voice websocket connected")
    try:
        await process_voice_websocket(websocket, phone_number)
    except WebSocketDisconnect:
        logger.info("Voice websocket disconnected normally")
    except Exception:
        logger.exception("Voice websocket error")
        try:
            await websocket.send_json({"type": "error", "data": "Internal server error occurred"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Voice websocket session ended")


@router.get("/health")
async def voice_health():
    status = get_models_status()
    provider = "edge-tts" if status.get("tts_available") else "none"
    return {
        "status": "ok" if status["whisper_loaded"] else "degraded",
        "whisper_loaded": status["whisper_loaded"],
        "vad_loaded": status["vad_loaded"],
        "tts_available": status.get("tts_available", False),
        "tts_provider": provider,
        "websocket_endpoint": "/api/v1/voice/stream",
    }
