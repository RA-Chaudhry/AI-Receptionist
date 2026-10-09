import os
import json
import logging
import hashlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Structured JSON log path
LOG_FILE = "logs/interactions.jsonl"

def log_interaction(
    identifier: str,
    query: str,
    response: str,
    delivery_status: dict,  # e.g. {"email": "sent", "sms": "sent", "call": "not_applicable"}
):
    """
    Logs every interaction in a structured JSON log file.
    """
    # Interaction content and contact identifiers are PII.  Operational logs
    # retain only stable, non-reversible correlation digests and lengths.
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "identifier_hash": digest(identifier),
        "query_hash": digest(query),
        "query_length": len(query),
        "response_hash": digest(response),
        "response_length": len(response),
        "delivery_status": delivery_status
    }
    
    # Ensure logs directory exists
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as exc:
        logger.error(f"Failed to write structured log: {exc}")
