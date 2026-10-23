
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from os import getenv

load_dotenv()

# Modelo de IA
llm = ChatOpenAI(
    api_key=getenv("API_KEY"),
    model="gpt-4o-mini",
    temperature=0.7
)

# Nó responsável por chamar a IA
def responder(state: MessagesState):
    resposta = llm.invoke(state["messages"])

    return {"messages": [resposta]}


# Construção do grafo
workflow = StateGraph(MessagesState)

workflow.add_node("responder", responder)

workflow.add_edge(START, "responder")
workflow.add_edge("responder", END)

# Memória temporária
memory = InMemorySaver()

# Compilação do grafo
graph = workflow.compile(checkpointer=memory)


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
