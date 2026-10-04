from fastapi import APIRouter

from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.review import router as review_router
from app.api.v1.endpoints.users import router as users_router
from app.api.v1.endpoints.vonage import router as vonage_router
from app.api.v1.endpoints.health import router as health_router

api_router = APIRouter()
api_router.include_router(chat_router)
api_router.include_router(review_router)
api_router.include_router(users_router)
api_router.include_router(vonage_router)
api_router.include_router(health_router)

