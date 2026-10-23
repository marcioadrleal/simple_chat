
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
import os
from langgraph.prebuilt import ToolNode, tools_condition

from services.vectorServices import VectorSearchService 
from langchain_core.tools import tool


# Carregar configurações
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Inicializa o serviço uma única vez
vector_service = VectorSearchService(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# Modelo de IA
llm = ChatOpenAI(
    api_key= os.getenv("API_KEY"),
    model="gpt-4o-mini",
    temperature=0
)

# Estado compartilhado da conversa
class ChatState(MessagesState):
    assunto: str


# Estrutura de classificação
class Classificacao(BaseModel):
    assunto: Literal["INSS", "GERAL"]


classificador = llm.with_structured_output(Classificacao)


@tool
def pesquisar_inss(pergunta: str) -> str:
    """Pesquisa informações sobre INSS nos documentos do banco vetorial."""

    print("Executando Tool pesquisar_inss")

    documentos = vector_service.buscar(pergunta)

    if not documentos:
        return "Nenhum documento encontrado."

    return "\n\n".join(
        doc["content"] for doc in documentos
    )


tools_inss = [pesquisar_inss]

llm_inss = llm.bind_tools(tools_inss)

# 1 - Classificador
def classificar(state: ChatState):

    resultado = classificador.invoke([
        SystemMessage(content=(
            "Você é um classificador de perguntas. "
            "Classifique como INSS quando o assunto envolver "
            "aposentadoria, previdência social brasileira, "
            "benefícios previdenciários, contribuições, "
            "tempo de contribuição, CNIS ou INSS. "
            "Considere o histórico para perguntas de continuidade. "
            "Caso contrário, classifique como GERAL."
        )),
        *state["messages"]
    ])

    print("Assunto identificado:", resultado.assunto)

    return {"assunto": resultado.assunto}


# 2 - Agente especializado em INSS
def agente_inss(state: ChatState):
    print("Passei pelo Agente INSS")

    resposta = llm_inss.invoke([
        SystemMessage(content=(
            "Você é um agente especializado em INSS "
            "e previdência social brasileira. "
            "Utilize a ferramenta pesquisar_inss para consultar "
            "os documentos antes de responder perguntas sobre INSS. "
            "Responda com base nas informações recuperadas. "
            "Se os documentos não contiverem a resposta, "
            "informe que não encontrou informações suficientes. "
            "Não invente leis, requisitos ou valores."
        )),
        *state["messages"]
    ])

    return {"messages": [resposta]}


# 3 - Agente geral
def agente_geral(state: ChatState):
    print("Passei pelo Agente Geral")
    resposta = llm.invoke([
        SystemMessage(content=(
            "Você é um assistente de IA. "
            "Responda às perguntas de forma clara, "
            "objetiva e em português brasileiro."
        )),
        *state["messages"]
    ])

    return {"messages": [resposta]}


# 4 - Decisão da rota
def decidir_rota(state: ChatState):

    if state["assunto"] == "INSS":
        return "agente_inss"

    return "agente_geral"



# 5 - Construção do workflow
workflow = StateGraph(ChatState)

# Registrar os nós
workflow.add_node("classificar", classificar)
workflow.add_node("agente_inss", agente_inss)
workflow.add_node("agente_geral", agente_geral)

# Registrar o nó de ferramentas
workflow.add_node(
    "ferramentas_inss",
    ToolNode(tools_inss)
)

# Início
workflow.add_edge(START, "classificar")

# Classificação da pergunta
workflow.add_conditional_edges(
    "classificar",
    decidir_rota,
    {
        "agente_inss": "agente_inss",
        "agente_geral": "agente_geral"
    }
)

# Agente INSS decide se chama uma Tool
workflow.add_conditional_edges(
    "agente_inss",
    tools_condition,
    {
        "tools": "ferramentas_inss",
        "__end__": END
    }
)

# Depois da Tool, retorna ao agente INSS
workflow.add_edge(
    "ferramentas_inss",
    "agente_inss"
)

# Agente geral finaliza diretamente
workflow.add_edge("agente_geral", END)



# 6 - Memória da conversa
memory = InMemorySaver()

graph = workflow.compile(checkpointer=memory)


# 7 - Entrada do chat
def enviar_pergunta(pergunta: str, thread_id: str):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    resultado = graph.invoke(
        {
            "messages": [
                HumanMessage(content=pergunta)
            ]
        },
        config=config
    )

    return resultado["messages"][-1].content
