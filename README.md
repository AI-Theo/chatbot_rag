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
├── data/                    # Vos documents sources (PDF, Excel)
├── chroma_db/               # Base vectorielle (générée automatiquement)
├── app.py                   # Interface Streamlit
├── main.py                  # API FastAPI (optionnel)
├── static/index.html        # Interface HTML pour FastAPI
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
git clone https://github.com/TON_USERNAME/chatbot_rag.git
cd chatbot_rag

# 2. Créer le venv avec Python 3.11
/Users/theo/.pyenv/versions/3.11.9/bin/python -m venv .venv

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

À faire à chaque fois que vous modifiez le contenu de `data/` (ajout, suppression ou remplacement de fichiers) :

```bash
# Supprimer l'ancien vectorstore
rm -rf chroma_db/

# Reconstruire
python -c "from ingestion.vectorstore import build_vectorstore; build_vectorstore()"
```

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

## Lancer l'API FastAPI (optionnel)

```bash
uvicorn main:app --reload
```

L'API est disponible sur `http://localhost:8000`

Endpoints :
- `GET  /`         — interface HTML
- `POST /chat`     — envoyer une question
- `POST /ingest`   — relancer l'ingestion depuis `data/`
- `GET  /health`   — statut de l'API

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
| API | FastAPI |

## Formats supportés

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Excel | `.xlsx`, `.xls` |

## .gitignore recommandé

```
.venv/
__pycache__/
*.pyc
.env
chroma_db/
data/
```

> `chroma_db/` et `data/` sont exclus du versioning — le vectorstore se reconstruit en local, et les documents sources peuvent être confidentiels.

## Licence

Projet privé — Tetria
