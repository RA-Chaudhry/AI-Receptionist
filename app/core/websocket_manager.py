import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages active local WebSocket connections for voice streaming sessions.
    """
    def __init__(self):
        # Maps connection_id to WebSocket object
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, connection_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info("WebSocket connection registered: %s", connection_id)

    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info("WebSocket connection unregistered: %s", connection_id)

    async def send_json(self, connection_id: str, message: dict):
        websocket = self.active_connections.get(connection_id)
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.debug("Failed to send JSON to connection %s: %s", connection_id, e)

    async def send_bytes(self, connection_id: str, data: bytes):
        websocket = self.active_connections.get(connection_id)
        if websocket:
            try:
                await websocket.send_bytes(data)
            except Exception as e:
                logger.debug("Failed to send bytes to connection %s: %s", connection_id, e)

    async def broadcast_json(self, message: dict):
        for connection_id in list(self.active_connections.keys()):
            await self.send_json(connection_id, message)


# Global connection manager instance
connection_manager = ConnectionManager()
