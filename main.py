
from fastapi import FastAPI
from uuid import uuid4

from schema.request.request import ChatRequest
from schema.response.response import ChatResponse
from chat.chat import enviar_pergunta

app = FastAPI()


@app.post("/enviar/pergunta")
def chat(chat_request: ChatRequest):
    thread_id = chat_request.threadId or uuid4()

    resposta = enviar_pergunta(
        pergunta=chat_request.pergunta,
        thread_id=str(thread_id)
    )

    return ChatResponse(
        threadId=thread_id,
        resposta=resposta
    )