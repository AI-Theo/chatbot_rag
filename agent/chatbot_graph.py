from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated
import operator
from agent.chatbot_tools import search_internal_docs

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

tools = [search_internal_docs]
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """Tu es le chatbot, un assistant IA interne.
Tu réponds aux questions en te basant sur les documents internes de l'entreprise.
Utilise toujours l'outil search_internal_docs avant de répondre.

IMPORTANT : À la fin de chaque réponse, liste toujours les sources utilisées sous ce format exact :

Sources :
  nom_du_fichier — page X

Si tu ne trouves pas l'information, dis-le clairement."""

# -- Nodes du graphe --
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", tool_node)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")

chatbot_agent = graph.compile()

def ask_chatbot(question: str, history: list = []) -> str:
    messages = history + [HumanMessage(content=question)]
    result = chatbot_agent.invoke({"messages": messages})
    full_response = result["messages"][-1].content

    if "Sources" in full_response:
        parts = full_response.split("Sources")
        return {
            "response": parts[0].strip(),
            "sources": "Sources" + parts[1]
        }
    
    return {
        "response": full_response,
        "sources": None
    }