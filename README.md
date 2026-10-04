# AI Receptionist Backend

Production-style FastAPI backend for:

- AI Receptionist chat API
- Google review reply generator
- Basic CRM with users and chat history

## Tech Stack

- FastAPI
- Python 3.11+
- Groq API
- SQLite
- SQLAlchemy ORM

## Project Structure

```text
app/
  api/
    v1/
      endpoints/
  core/
  crud/
  db/
  models/
  schemas/
  services/
  utils/
  main.py
```

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your Groq API key.
4. Run the app:

```bash
uvicorn app.main:app --reload
```

5. Open the UI:

```text
http://127.0.0.1:8000/
```

6. Open Swagger docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

- `POST /api/v1/chat`
- `POST /api/v1/review-reply`
- `GET /api/v1/users`
- `GET /api/v1/users/{user_id}`
- `GET /api/v1/users/{user_id}/history`

## Production Extras

- Frontend dashboard served from `/`
- Basic rate limiting and security headers
- Alembic migration scaffold
- Dockerfile and `docker-compose.yml`
- Pytest coverage for API flow

## Real-Time Voice Configuration & Low-Latency Tuning

The AI receptionist includes an ultra-low-latency real-time voice pipeline (WebSockets + Vonage VoIP) with Groq LLM streaming, Groq Whisper / Faster-Whisper STT, and Edge-TTS pipelining.

### Key Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `STT_PROVIDER` | `groq_whisper` | STT engine (`groq_whisper` or `faster_whisper`). |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq LLM model for real-time speech dialogue. |
| `LLM_MAX_TOKENS` | `200` | Max generation tokens per turn. |
| `VAD_AGGRESSIVENESS` | `2` | WebRTC VAD aggressiveness mode (0–3). |
| `VAD_MIN_RMS` | `100.0` | Minimum audio RMS energy threshold for speech detection. |
| `VAD_SILENCE_DURATION_MS` | `350` | Trailing silence window required to endpoint speech turn. |
| `VAD_PRE_BUFFER_MS` | `200` | Audio pre-buffering duration before detected speech onset. |
| `VAD_POST_BUFFER_MS` | `150` | Maximum trailing silence preserved for STT context. |
| `TTS_VOICE` | `en-US-JennyNeural` | Edge-TTS voice model. |

### Measuring Latency & Telemetry Reports

Run the latency report utility against recorded session logs:

```bash
python scripts/latency_report.py
```

Prints p50 and p95 timings across all pipeline stages:
- **Endpointing Delay:** Silence duration before turn detection.
- **STT:** Speech-to-text transcription latency.
- **LLM TTFT:** Time to first token.
- **First Clause:** Time until first sentence/clause punctuation chunk.
- **TTS TTFB:** Time to first synthesized audio byte.
- **TTFA:** Time from user speech end to first audio byte played/sent.

## Database Migration

```bash
alembic upgrade head
```

## Tests

```bash
pytest -q
```

## Docker

```bash
docker compose up --build
```

## Notes

- Tables are created automatically on startup.
- The chatbot keeps a short memory from recent chat history.
- The review reply endpoint does not store messages by default.

