from langchain.tools import tool
from ingestion.vectorstore import load_vectorstore

@tool
def search_internal_docs(query: str) -> str:
    """
    Recherche dans les documents internes (PDF et Excel).
    Utilise cet outil pour répondre à des questions sur les données internes.
    """
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=4)
    
    if not results:
        return "Aucun document pertinent trouvé."

    context = ""
    for doc in results:
        source = doc.metadata.get('source', 'inconnue')
        page = doc.metadata.get('page', None)
        source_label = f"{source}, page {page + 1}" if page is not None else source
        context += f"[SOURCE: {source_label}]\n{doc.page_content}\n\n---\n\n"

    return context