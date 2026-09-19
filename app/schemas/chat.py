from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.schemas.user import UserRead


class ChatRequest(BaseModel):
    user_id: int | None = None
    user_name: str = Field(min_length=2, max_length=100)
    user_email: EmailStr | None = None
    phone_number: str | None = None
    message: str = Field(min_length=1, max_length=4000)

    model_config = ConfigDict(str_strip_whitespace=True)


class ChatResponse(BaseModel):
    user: UserRead
    user_message: str
    assistant_reply: str
    created_at: datetime
    source: str = Field(default="local_rule", description="Indicates if response was from 'local_rule', 'ai_llm', or 'fallback'")
    provider: str = Field(default="Local Rule Engine", description="Readable name of provider (e.g., 'Local Rule Router' or 'Groq AI')")

    model_config = ConfigDict(from_attributes=True)


class ChatMessageRead(BaseModel):
    id: int
    user_id: int
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
