from rag.loader import  process_pdf
from rag.splitter import split_documents
from rag.embeddings import Embedding_manager
from rag.vectorstore import VectorStore
from logger import logging


#---------------------#

def ingest_pdf(pdf_file: str, embedding_manager: Embedding_manager = Embedding_manager(), vector_store: VectorStore = VectorStore()) -> None:
    documents = process_pdf(pdf_file)
    chunks = split_documents(documents)
    texts = [doc.page_content for doc in chunks]

    embeddings = embedding_manager.create_embeddings(texts)

    vector_store.add_documents(documents=chunks, embeddings=embeddings)

    logging.info(f"PDFs ingested into vector DB")
    return None

#----------------------#
