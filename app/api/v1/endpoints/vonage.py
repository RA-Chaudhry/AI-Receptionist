import logging
from fastapi import APIRouter, HTTPException, Query, Request, Response, WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.services.voice_service import process_vonage_websocket
from app.core.config import settings
from app.core.security import validate_voice_websocket_token

router = APIRouter(tags=["Vonage Webhooks"])
logger = logging.getLogger(__name__)


@router.get("/vonage/answer", operation_id="vonage_answer_get")
@router.post("/vonage/answer", operation_id="vonage_answer_post")
async def vonage_answer(request: Request, token: str | None = Query(default=None)):
    """
    Vonage Answer Webhook endpoint. Returns a Vonage NCCO (Nexmo Call Control Object)
    to connect the call's audio stream to our WebSocket server in raw PCM format.
    """
    logger.info("Vonage incoming call answer webhook triggered.")
    
    if not validate_voice_websocket_token(token):
        raise HTTPException(status_code=401, detail="Invalid voice stream token.")

    if settings.environment.strip().lower() == "production" and not settings.public_base_url:
        raise HTTPException(status_code=503, detail="PUBLIC_BASE_URL must be configured in production.")

    base_url = settings.public_base_url.rstrip("/") or str(request.base_url).rstrip("/")
    ws_uri = base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    ws_uri = f"{ws_uri}/api/v1/vonage/stream"
    
    clean_ws_uri = ws_uri
    if settings.voice_websocket_token:
        ws_uri = f"{ws_uri}?token={settings.voice_websocket_token}"
    
    logger.info("Connecting Vonage voice stream to: %s", clean_ws_uri)
    
    ncco = [
        {
            "action": "connect",
            "endpoint": [
                {
                    "type": "websocket",
                    "uri": ws_uri,
                    "content-type": "audio/l16;rate=16000"
                }
            ]
        }
    ]
    return ncco


@router.get("/vonage/event", operation_id="vonage_event_get")
@router.post("/vonage/event", operation_id="vonage_event_post")
async def vonage_event(request: Request):
    """
    Vonage status updates webhook endpoint (ringing, answered, completed, etc.).
    """
    try:
        body = await request.json()
        logger.debug("Vonage call event received: %s", body.get("status") if isinstance(body, dict) else "unknown")
    except Exception as exc:
        logger.debug("Non-JSON or unreadable Vonage event payload: %s", exc)
    return Response(status_code=200)


@router.websocket("/vonage/stream")
async def vonage_stream(websocket: WebSocket, token: str | None = Query(default=None)):
    """
    Vonage WebSocket audio stream endpoint. Receives raw binary 16kHz PCM chunks
    from the phone call, transcribes it, runs AI response generation, and streams PCM back.
    """
    if not validate_voice_websocket_token(token):
        await websocket.close(code=1008)
        return
    # Accept the socket internally within the service method
    try:
        await process_vonage_websocket(websocket)
    except WebSocketDisconnect:
        logger.info("Vonage WebSocket disconnected normally")
    except Exception as exc:
        logger.exception("Error in Vonage WebSocket handler")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
