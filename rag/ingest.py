from rag.loader import  process_all_pdfs
from rag.splitter import split_documents
from rag.embeddings import Embedding_manager
from rag.vectorstore import VectorStore
from logger import logging
from utils.config import IngestConfig
from typing import Tuple

def ingest_pdfs(pdf_dir: str = IngestConfig.pdf_dir) -> Tuple[VectorStore, Embedding_manager]:
    documents = process_all_pdfs(pdf_dir)
    chunks = split_documents(documents)
    texts = [doc.page_content for doc in chunks]

    embedding_manager = Embedding_manager()
    embeddings = embedding_manager.create_embeddings(texts)

    vector_store = VectorStore()
    vector_store.add_documents(documents=chunks, embeddings=embeddings)

    logging.info(f"PDFs ingested into vector DB")
    return vector_store, embedding_manager

