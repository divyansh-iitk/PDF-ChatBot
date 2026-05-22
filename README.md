# PDF ChatBot 📄🤖

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them using natural language queries.

## 🌟 Features

*   **PDF Ingestion:** Upload and process PDF documents for search.
*   **Hybrid Search:** Combines BM25 keyword search with semantic vector search (ChromaDB + Sentence Transformers) using LangChain's `EnsembleRetriever` for highly accurate retrieval.
*   **Context-Aware Answering:** Uses Groq-hosted LLMs (Llama-3.3-70B) to generate accurate answers strictly based on the provided document context.
*   **Source Attribution:** Displays the relevant document chunks used to generate the answer, along with their similarity scores.
*   **Modern UI:** Interactive and responsive frontend built with Streamlit.
*   **FastAPI Backend:** Robust and scalable REST API for handling uploads and queries.

## 🛠️ Technology Stack

*   **Backend:** FastAPI, Python
*   **Frontend:** Streamlit
*   **LLM:** Groq API (Llama-3.3-70B-Versatile)
*   **Embeddings:** `BAAI/bge-base-en-v1.5` (via `sentence-transformers`)
*   **Vector Database:** ChromaDB (Persistent)
*   **Orchestration:** LangChain
*   **PDF Processing:** PyMuPDF (`pymupdf`)

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   [Groq API Key](https://console.groq.com/keys)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/divyansh-iitk/PDF-ChatBot.git
    cd PDF-ChatBot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables:**
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

### Running the Application

The application consists of a FastAPI backend and a Streamlit frontend. You need to run both simultaneously.

1.  **Start the FastAPI Backend:**
    ```bash
    uvicorn app.main:app --reload
    ```
    *The API will run at `http://127.0.0.1:8000`. You can view the interactive API docs at `http://127.0.0.1:8000/docs`.*

2.  **Start the Streamlit Frontend:**
    Open a new terminal window/tab, activate your virtual environment, and run:
    ```bash
    streamlit run frontend/app.py
    ```
    *The frontend will open in your browser at `http://localhost:8501`.*

## 📁 Project Structure

```
PDF-ChatBot/
├── app/                  # FastAPI application setup and routing
│   ├── routes/           # API endpoints (upload, query)
│   ├── schemas/          # Pydantic models for request validation
│   └── main.py           # FastAPI app initialization
├── frontend/             # Streamlit user interface
│   └── app.py            # Streamlit app script
├── rag/                  # Core RAG implementation
│   ├── chain.py          # LangChain LLM setup with Groq
│   ├── embeddings.py     # SentenceTransformers embedding generation
│   ├── ingest.py         # PDF processing pipeline
│   ├── loader.py         # Document loading
│   ├── retriever.py      # Custom vector store retriever
│   ├── splitter.py       # Text chunking logic
│   ├── vectorstore.py    # ChromaDB management
│   └── BM25.py           # BM25 keyword search implementation
├── utils/                # Utility scripts
│   └── config.py         # Centralized configuration (chunk size, model names, etc.)
├── logger/               # Custom logging setup
├── db/                   # Persistent ChromaDB storage (created on run)
├── data/                 # Directory for uploaded PDFs (created on run)
├── .env                  # Environment variables (not tracked by git)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## ⚙️ Configuration

You can tweak the RAG pipeline settings (chunk size, overlap, embedding model, top-k retrieval, etc.) in `utils/config.py`.

## 📜 License

This project is open-source. Please see the [LICENSE](LICENSE) file for more information.