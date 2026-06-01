# PDF ChatBot 📄🤖

A full-stack **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF documents and chat with them using natural language. Built with a modular pipeline that combines keyword search, semantic search, and neural reranking for accurate, context-grounded answers.

## ✨ Features

- **PDF Upload & Ingestion** — Upload PDFs via the UI; documents are automatically chunked, embedded, and indexed.
- **Hybrid Retrieval** — Combines BM25 keyword search with dense vector search (ChromaDB + Sentence Transformers) using LangChain's `EnsembleRetriever`.
- **Cohere Reranking** — Retrieved candidates are reranked using Cohere's `rerank-v3.5` model for higher precision.
- **Conversational Memory** — Supports multi-turn conversations with LLM-based query rewriting that reformulates follow-up questions using chat history.
- **Source Attribution** — Displays the relevant document chunks used to generate the answer, along with their relevance scores.
- **Dockerized Deployment** — One-command deployment with Docker Compose (backend + frontend + persistent storage).

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Rewriting     │ ◄── Chat history context
│  (Llama-3.3-70B)     │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Hybrid Retrieval    │
│  ┌───────┐ ┌───────┐│
│  │ BM25  │ │Vector ││
│  │Search │ │Search ││
│  └───┬───┘ └───┬───┘│
│      └────┬────┘    │
│    EnsembleRetriever │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Cohere Reranking    │
│  (rerank-v3.5)       │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Answer Generation   │
│  (Llama-3.3-70B)     │
└─────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Python 3.12 |
| **Frontend** | Streamlit |
| **LLM** | Llama-3.3-70B-Versatile (via Groq API) |
| **Embeddings** | `BAAI/bge-base-en-v1.5` (Sentence Transformers) |
| **Vector Database** | ChromaDB (Persistent) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Reranking** | Cohere Rerank v3.5 |
| **Orchestration** | LangChain |
| **PDF Parsing** | PyMuPDF |
| **Containerization** | Docker, Docker Compose |

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [Groq API Key](https://console.groq.com/keys)
- [Cohere API Key](https://dashboard.cohere.com/api-keys)

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/divyansh-iitk/PDF-ChatBot.git
cd PDF-ChatBot

# Create .env file
cp .env.example .env
# Edit .env and add your API keys

# Build and run
docker compose up --build
```

- **Frontend:** [http://localhost:8501](http://localhost:8501)
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Local Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/divyansh-iitk/PDF-ChatBot.git
    cd PDF-ChatBot
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your API keys:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    COHERE_API_KEY=your_cohere_api_key_here
    ```

5. **Start the FastAPI backend:**
    ```bash
    cd backend
    uvicorn app.main:app --reload
    ```
    The API will run at `http://127.0.0.1:8000` (interactive docs at `/docs`).

6. **Start the Streamlit frontend** (new terminal):
    ```bash
    streamlit run frontend/app.py
    ```
    The frontend will open at `http://localhost:8501`.

## 📁 Project Structure

```
PDF-ChatBot/
├── backend/                    # FastAPI backend service
│   ├── app/
│   │   ├── main.py             # FastAPI app initialization & lifespan
│   │   ├── routes/
│   │   │   ├── upload.py       # PDF upload endpoint
│   │   │   └── query.py        # Query endpoint (retrieval + generation)
│   │   └── schemas/
│   │       └── query.py        # Pydantic request models
│   ├── rag/                    # Core RAG pipeline
│   │   ├── loader.py           # PDF loading with PyMuPDF
│   │   ├── splitter.py         # Recursive text chunking
│   │   ├── embeddings.py       # Sentence Transformers embedding manager
│   │   ├── vectorstore.py      # ChromaDB persistent vector store
│   │   ├── BM25.py             # BM25 keyword retriever
│   │   ├── retriever.py        # Custom vector store retriever
│   │   ├── cohere_reranker.py  # Cohere reranking integration
│   │   ├── ingest.py           # PDF processing orchestrator
│   │   └── llm.py              # Groq LLM (query rewriting + answer generation)
│   ├── utils/
│   │   └── config.py           # Centralized pipeline configuration
│   ├── logger/                 # Custom logging setup
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                   # Streamlit frontend service
│   ├── app.py                  # Chat UI
│   ├── Dockerfile
│   └── requirements.txt
├── compose.yaml                # Docker Compose orchestration
├── .env.example                # Required environment variables
├── pyproject.toml              # Python project metadata (uv)
├── requirements.txt            # Root-level Python dependencies
└── LICENSE                     # MIT License
```

## ⚙️ Configuration

All RAG pipeline parameters are centralized in [`backend/utils/config.py`](backend/utils/config.py):

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 1000 | Characters per text chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `embedding_model` | `BAAI/bge-base-en-v1.5` | Sentence Transformers model |
| `top_k` (retriever) | 10 | Documents retrieved per strategy |
| `score_threshold` | 0.5 | Minimum cosine similarity |
| `BM25 weight` | 0.5 | Weight for keyword search in ensemble |
| `Vector weight` | 0.5 | Weight for semantic search in ensemble |
| `Reranker top_n` | 4 | Final documents after reranking |
| `LLM model` | `llama-3.3-70b-versatile` | Groq-hosted LLM |
| `temperature` | 0.1 | LLM temperature |
| `last_n_chats` | 3 | Chat history window for query rewriting |

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/upload` | Upload and ingest a PDF file |
| `POST` | `/api/query` | Query ingested documents |

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.