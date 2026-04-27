from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from agent.chatbot_graph import ask_chatbot
from ingestion.vectorstore import build_vectorstore
import os

load_dotenv()

app = FastAPI(title="CHATBOT POC", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_history = []

class Question(BaseModel):
    question: str
    reset: bool = False

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(body: Question):
    global conversation_history
    if body.reset:
        conversation_history = []

    result = ask_chatbot(body.question, conversation_history)

    from langchain_core.messages import HumanMessage, AIMessage
    conversation_history.append(HumanMessage(content=body.question))
    conversation_history.append(AIMessage(content=result["response"]))

    return {
        "response": result["response"],
        "sources": result["sources"]
    }

@app.post("/ingest")
async def ingest():
    """Relance l'ingestion des fichiers dans ./data"""
    build_vectorstore()
    return {"status": "✅ Ingestion terminée"}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "gpt-4o"}