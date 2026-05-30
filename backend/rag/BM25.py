from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List
from utils.config import BM25Config

def BM25_retriever(chunks: List[Document]) -> BM25Retriever:
    """_summary_

    Args:
        chunks (List[Document]): _description_

    Returns:
        BM25Retriever: _description_
    """
    
    # Creating lowercase of the content
    lower_chunks = []
    for doc in chunks:
        doc_lower = doc.page_content.lower().strip()
        metadata = doc.metadata
        lower_chunks.append(Document(page_content=doc_lower, metadata=metadata))
        
    retriever = BM25Retriever.from_documents(lower_chunks)
    retriever.k = BM25Config.top_k
    
    return retriever