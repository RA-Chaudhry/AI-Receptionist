import logging

from app.services.groq_service import GroqService
from app.utils.prompts import review_reply_prompt

logger = logging.getLogger(__name__)


class ReviewService:
    def __init__(self) -> None:
        self.groq = GroqService()

    async def generate_reply(self, review_text: str) -> str:
        system_prompt = review_reply_prompt()
        logger.debug("Using review_reply_prompt for review reply generation")
        user_input = review_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        reply = await self.groq.chat_completion(messages)
        return self._normalize_reply(reply)

    @staticmethod
    def _normalize_reply(reply: str) -> str:
        cleaned = reply.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()
        return cleaned
