from uuid import UUID
from pydantic import BaseModel

class ChatRequest(BaseModel):
    pergunta: str
    threadId: UUID | None = None