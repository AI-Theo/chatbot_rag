from langchain.tools import tool

from ingestion.vectorstore import load_vectorstore


@tool
def search_internal_docs(query: str) -> str:
    """
    Recherche dans les documents internes (PDF et Excel).
    Utilise cet outil pour répondre à des questions sur les données internes.
    """
    results = load_vectorstore().similarity_search(query, k=8)

    if not results:
        return "Aucun document pertinent trouvé."

    context = ""
    for doc in results:
        source = doc.metadata.get("source", "inconnue")
        page = doc.metadata.get("page")
        row = doc.metadata.get("row")
        if page is not None:
            source_label = f"{source}, page {page + 1}"
        elif row is not None:
            source_label = f"{source}, ligne {row}"
        else:
            source_label = source
        context += f"[SOURCE: {source_label}]\n{doc.page_content}\n\n---\n\n"

    return context
