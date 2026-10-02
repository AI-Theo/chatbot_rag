import os

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)


def load_pdf(path: str) -> list[Document]:
    pages = PyPDFLoader(path).load()
    return SPLITTER.split_documents(pages)


def load_excel(path: str) -> list[Document]:
    df = pd.read_excel(path)
    documents = []
    for index, row in df.iterrows():
        # Chaque ligne devient un document ; les cellules vides sont ignorées
        content = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
        if content:
            documents.append(Document(
                page_content=content,
                metadata={"source": path, "type": "excel", "row": int(index) + 2},
            ))
    return documents


def load_all(data_dir: str = "./data") -> list[Document]:
    docs = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        lower = filename.lower()
        if lower.endswith(".pdf"):
            docs.extend(load_pdf(path))
            print(f"✅ PDF chargé : {filename}")
        elif lower.endswith((".xlsx", ".xls")):
            docs.extend(load_excel(path))
            print(f"✅ Excel chargé : {filename}")
    print(f"📄 Total chunks : {len(docs)}")
    return docs
