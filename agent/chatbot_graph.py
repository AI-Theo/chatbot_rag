import operator
import re
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.chatbot_tools import search_internal_docs


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


class ChatbotResult(TypedDict):
    response: str
    sources: Optional[str]


tools = [search_internal_docs]
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """Tu es un assistant juridique qui répond UNIQUEMENT en te basant sur le Code civil fourni.
Tu n'as PAS de connaissances générales. Tu ne connais rien sur aucun sujet.
La seule façon d'obtenir des informations est d'appeler l'outil search_internal_docs.
Tu DOIS appeler search_internal_docs pour CHAQUE question, sans exception.

Règles de réponse :
- Cite toujours le numéro d'article exact (ex : "Article 1134 du Code civil")
- Si la question porte sur plusieurs articles liés, cite-les tous
- Si un article a été modifié, précise-le si l'information est disponible
- Ne donne jamais de conseil juridique personnel — rappelle que seul un avocat peut conseiller sur un cas précis
- Si tu ne trouves pas l'information dans les documents, dis : "Je n'ai pas trouvé cette information dans les documents fournis."

À la fin de chaque réponse, liste les sources sous ce format :
Sources :
  nom_du_fichier — page X — Article XXX

Si tu ne trouves pas l'information, dis-le clairement."""

# Le bloc final "Sources :" en début de ligne (et non le mot "sources" dans le texte)
SOURCES_BLOCK = re.compile(r"\n\s*\**Sources\s*\**\s*:", re.IGNORECASE)


# -- Nodes du graphe --
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")

chatbot_agent = graph.compile()


def split_sources(text: str) -> ChatbotResult:
    """Sépare la réponse du bloc "Sources :" final, s'il existe."""
    matches = list(SOURCES_BLOCK.finditer(text))
    if not matches:
        return {"response": text.strip(), "sources": None}
    start = matches[-1].start()
    return {"response": text[:start].strip(), "sources": text[start:].strip()}


def ask_chatbot(question: str, history: Optional[list[BaseMessage]] = None) -> ChatbotResult:
    messages = list(history or []) + [HumanMessage(content=question)]
    result = chatbot_agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 5},
    )
    return split_sources(result["messages"][-1].content)
