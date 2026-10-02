import shutil
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from ingestion.data_loader import load_all

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-large"


def build_vectorstore(data_dir: str = "./data") -> Chroma:
    """(Re)construit la base vectorielle à partir des documents de `data_dir`.

    L'ancienne base est supprimée avant reconstruction, pour éviter les doublons.
    """
    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)

    docs = load_all(data_dir)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_PATH,
    )
    load_vectorstore.cache_clear()
    print(f"✅ Vector store créé avec {len(docs)} chunks")
    return vectorstore


@lru_cache(maxsize=1)
def load_vectorstore() -> Chroma:
    """Charge la base vectorielle existante (une seule fois par processus)."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
