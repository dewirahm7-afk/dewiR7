# backend/api/websockets.py
from fastapi import WebSocket
from typing import Dict, List
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        # Remove from session mappings
        for session_id, clients in self.session_connections.items():
            if client_id in clients:
                clients.remove(client_id)

    def add_to_session(self, client_id: str, session_id: str):
        if session_id not in self.session_connections:
            self.session_connections[session_id] = []
        if client_id not in self.session_connections[session_id]:
            self.session_connections[session_id].append(client_id)

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(message)

    async def broadcast_to_session(self, session_id: str, message: dict):
        if session_id in self.session_connections:
            disconnected = []
            for client_id in self.session_connections[session_id]:
                if client_id in self.active_connections:
                    websocket = self.active_connections[client_id]
                    try:
                        await websocket.send_json(message)
                    except:
                        disconnected.append(client_id)
                else:
                    disconnected.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected:
                self.session_connections[session_id].remove(client_id)

# Global instance
websocket_manager = ConnectionManager()