from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_core.documents import Document
import os

def load_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(pages)

def load_excel(path: str) -> list[Document]:
    df = pd.read_excel(path)
    documents = []
    for _, row in df.iterrows():
        # Chaque ligne devient un document
        content = " | ".join([f"{col}: {val}" for col, val in row.items()])
        documents.append(Document(
            page_content=content,
            metadata={"source": path, "type": "excel"}
        ))
    return documents

def load_all(data_dir: str = "./data") -> list[Document]:
    docs = []
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            docs.extend(load_pdf(path))
            print(f"✅ PDF chargé : {filename}")
        elif filename.endswith((".xlsx", ".xls")):
            docs.extend(load_excel(path))
            print(f"✅ Excel chargé : {filename}")
    print(f"📄 Total chunks : {len(docs)}")
    return docs