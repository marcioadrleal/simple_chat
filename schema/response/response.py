from uuid import UUID
from pydantic import BaseModel

class ChatResponse(BaseModel):
    threadId: UUID
    resposta: str