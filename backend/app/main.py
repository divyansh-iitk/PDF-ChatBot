from fastapi import FastAPI
from contextlib import asynccontextmanager

from rag.embeddings import Embedding_manager
from rag.vectorstore import VectorStore
from rag.llm import GroqLLM
from app.routes import upload, query
from logger import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔹 Startup logic
    try:
        app.state.vector_store = VectorStore()
        app.state.embedding_manager = Embedding_manager()
        app.state.llm = GroqLLM()
        app.state.chat_history = []
    except Exception as e:
        logging.error(f"Error while app startup: {e}")
        raise

    print("✅ Models loaded")

    yield  # <-- app runs here

    # 🔹 Shutdown logic (optional)
    print("🛑 Shutting down")

app = FastAPI(
    title="ChatPDF API",
    description="RAG-based PDF Question Answering System",
    version="1.0",
    lifespan=lifespan
)

# Register routes
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(query.router, prefix="/api", tags=["Query"])


@app.get("/")
def root():
    return {"message": "ChatPDF API is running 🚀"}