import logging

from fastapi import APIRouter

from app.schemas.review import ReviewReplyRequest, ReviewReplyResponse
from app.services.review_service import ReviewService

router = APIRouter(tags=["Review Replies"])
logger = logging.getLogger(__name__)


@router.post("/review-reply", response_model=ReviewReplyResponse)
async def review_reply_endpoint(payload: ReviewReplyRequest):
    service = ReviewService()
    reply = await service.generate_reply(payload.review_text)
    logger.info("Review reply generated")
    return ReviewReplyResponse(reply=reply)

