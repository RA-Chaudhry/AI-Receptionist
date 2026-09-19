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
