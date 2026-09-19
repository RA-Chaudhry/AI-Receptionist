"""
tests/test_fastapi_routes.py — Comprehensive REST API Route & Error Handling Tests
"""

import pytest
from fastapi.testclient import TestClient

from app.services.groq_service import GroqServiceError
from app.services.chat_service import ChatService


def test_root_frontend_endpoint(client: TestClient):
    """Verify GET / returns the HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text
    assert "AI Receptionist Dashboard" in response.text


def test_health_check_endpoint(client: TestClient):
    """Verify GET /health returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_voice_health_endpoint(client: TestClient):
    """Verify GET /api/v1/voice/health returns model and service status."""
    response = client.get("/api/v1/voice/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "whisper_loaded" in data
    assert "vad_loaded" in data
    assert "tts_available" in data
    assert data["websocket_endpoint"] == "/api/v1/voice/stream"


def test_chat_endpoint_valid_request(client: TestClient, mock_groq):
    """Verify POST /api/v1/chat processes user query and returns structured response."""
    payload = {
        "user_name": "Hamza",
        "user_email": "hamza@example.com",
        "phone_number": "03001234567",
        "message": "Hello, what services do you provide?",
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["name"] == "Hamza"
    assert data["user"]["email"] == "hamza@example.com"
    assert data["user_message"] == payload["message"]
    assert len(data["assistant_reply"]) > 0


def test_chat_endpoint_validation_errors(client: TestClient):
    """Verify POST /api/v1/chat rejects invalid or missing fields with 422."""
    # 1. Missing message
    res1 = client.post("/api/v1/chat", json={"user_name": "Hamza"})
    assert res1.status_code == 422

    # 2. Empty user_name
    res2 = client.post("/api/v1/chat", json={"user_name": "", "message": "Hi"})
    assert res2.status_code == 422

    # 3. Invalid email format
    res3 = client.post("/api/v1/chat", json={"user_name": "Hamza", "user_email": "not-an-email", "message": "Hi"})
    assert res3.status_code == 422


def test_review_reply_endpoint(client: TestClient, mock_groq):
    """Verify POST /api/v1/review-reply generates and cleans review reply."""
    payload = {"review_text": "Excellent service and prompt communication from Logic Pulse!"}
    response = client.post("/api/v1/review-reply", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 0
    # Double quotes should be stripped
    assert '"' not in data["reply"]


def test_review_reply_validation_error(client: TestClient):
    """Verify POST /api/v1/review-reply rejects empty review with 422."""
    response = client.post("/api/v1/review-reply", json={"review_text": ""})
    assert response.status_code == 422


def test_users_crm_endpoints(client: TestClient, mock_groq):
    """Verify GET /api/v1/users, /users/{id}, and /users/{id}/history."""
    # Create two chats for a user
    client.post("/api/v1/chat", json={"user_name": "Tariq", "user_email": "tariq@example.com", "message": "First query"})
    client.post("/api/v1/chat", json={"user_name": "Tariq", "user_email": "tariq@example.com", "message": "Second query"})

    # 1. List users
    list_res = client.get("/api/v1/users")
    assert list_res.status_code == 200
    users = list_res.json()
    assert len(users) == 1
    assert users[0]["name"] == "Tariq"
    assert users[0]["chat_count"] == 2
    user_id = users[0]["id"]

    # 2. Get user by ID
    get_res = client.get(f"/api/v1/users/{user_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Tariq"

    # 3. Get non-existent user
    not_found_res = client.get("/api/v1/users/99999")
    assert not_found_res.status_code == 404
    assert not_found_res.json()["detail"] == "User not found"

    # 4. Get chat history
    hist_res = client.get(f"/api/v1/users/{user_id}/history")
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) == 4  # 2 user queries + 2 assistant responses
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    # 5. Get history for non-existent user
    hist_404 = client.get("/api/v1/users/99999/history")
    assert hist_404.status_code == 404


def test_groq_service_error_handler(client: TestClient, monkeypatch):
    """Verify GroqServiceError translates into HTTP 502 Bad Gateway."""
    async def mock_handle_chat(*args, **kwargs):
        raise GroqServiceError("Groq service unreachable")

    monkeypatch.setattr(ChatService, "handle_chat", mock_handle_chat)

    response = client.post(
        "/api/v1/chat",
        json={"user_name": "Ali", "message": "Hello"},
    )
    assert response.status_code == 502
    assert "Groq service unreachable" in response.json()["detail"]
