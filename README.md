# chatbot_rag

Assistant IA conversationnel basé sur la technologie RAG (Retrieval-Augmented Generation). Pose des questions en langage naturel à vos propres documents — PDF et Excel — et obtenez des réponses précises avec sources citées.

## Architecture

```
chatbot_rag/
├── agent/
│   ├── chatbot_graph.py     # Orchestration LangGraph + GPT-4o
│   └── chatbot_tools.py     # Outil de recherche vectorielle
├── ingestion/
│   ├── data_loader.py       # Chargement et découpage PDF / Excel
│   └── vectorstore.py       # Création et chargement du vectorstore Chroma
├── data/                    # Vos documents sources (PDF, Excel) — non versionné
├── chroma_db/               # Base vectorielle de démonstration (Code civil)
├── app.py                   # Interface Streamlit
└── requirements.txt
```

## Comment ça marche

1. **Ingestion** — les documents dans `data/` sont découpés en chunks et vectorisés via OpenAI Embeddings, puis stockés dans ChromaDB
2. **Question** — la question de l'utilisateur est transformée en vecteur et comparée aux chunks stockés
3. **Réponse** — GPT-4o reçoit les chunks les plus pertinents et génère une réponse sourcée

## Prérequis

- Python 3.11+
- Une clé API OpenAI

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/AI-Theo/chatbot_rag.git
cd chatbot_rag

# 2. Créer le venv avec Python 3.11+
python3.11 -m venv .venv

# 3. Activer le venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Créer le fichier .env
echo "OPENAI_API_KEY=sk-..." > .env
```

## Ajouter vos documents

Placez vos fichiers dans le dossier `data/` :

```
data/
├── mon_document.pdf
├── autre_fichier.pdf
└── tableau.xlsx
```

> **Important** — évitez les espaces dans les noms de fichiers, utilisez `_` ou `-` à la place.

## Construire le vectorstore

À faire à chaque fois que vous modifiez le contenu de `data/` (ajout, suppression ou remplacement de fichiers). L'ancienne base est supprimée automatiquement avant reconstruction :

```bash
python -c "from ingestion.vectorstore import build_vectorstore; build_vectorstore()"
```

> **Confidentialité** — la base vectorielle contient le texte des documents indexés. Ne commitez jamais une base construite à partir de documents confidentiels : seule la base de démonstration (Code civil, texte public) est versionnée.

Vous devriez voir :
```
✅ PDF chargé : mon_document.pdf
📄 Total chunks : 1234
✅ Vector store créé avec 1234 chunks
```

## Lancer l'interface Streamlit

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`

## Variables d'environnement

Créez un fichier `.env` à la racine :

```env
OPENAI_API_KEY=sk-...
```

## Stack technique

| Composant | Technologie |
|-----------|------------|
| LLM | GPT-4o (OpenAI) |
| Embeddings | text-embedding-3-large (OpenAI) |
| Orchestration | LangGraph |
| Vectorstore | ChromaDB |
| Interface | Streamlit |

## Formats supportés

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Excel | `.xlsx`, `.xls` |

## Licence

© Théo Algaze — Tetria. Code publié à titre de démonstration ; toute réutilisation nécessite un accord préalable.
