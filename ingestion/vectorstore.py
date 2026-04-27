from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from ingestion.data_loader import load_all

CHROMA_PATH = "./chroma_db"

def build_vectorstore() -> Chroma:
    docs = load_all("./data")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ Vector store créé avec {len(docs)} chunks")
    return vectorstore

def load_vectorstore() -> Chroma:
    """Charge un vector store existant."""
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )