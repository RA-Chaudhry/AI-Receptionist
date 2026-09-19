from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.chat import chat_crud
from app.crud.user import user_crud
from app.db.session import get_db
from app.core.security import require_admin_api_key
from app.schemas.user import UserRead
from app.schemas.chat import ChatMessageRead

router = APIRouter(tags=["CRM"])


@router.get("/users", response_model=list[UserRead])
async def list_users(
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_admin_api_key),
):
    users = await user_crud.list_users(db)
    return users


@router.get("/users/{user_id}", response_model=UserRead)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_admin_api_key),
):
    user = await user_crud.get_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/users/{user_id}/history", response_model=list[ChatMessageRead])
async def get_chat_history(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_admin_api_key),
):
    user = await user_crud.get_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    history = await chat_crud.get_history_for_user(db, user_id)
    return history
