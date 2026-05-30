from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from rag.vectorstore import VectorStore
from rag.embeddings import Embedding_manager
from typing import List
from utils.config import RetrieverConfig
from logger import logging

class RAGRetriver(BaseRetriever):
    """Handles query-based retrieval from the vector store"""

    vector_store: VectorStore
    embedding_manager: Embedding_manager
    top_k: int = RetrieverConfig.top_k
    score_threshold: float = RetrieverConfig.score_threshold
        
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        ) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        
        # Generate query embedding
        query_embedding = self.embedding_manager.create_embeddings([query])[0]
        
        # Search in vector store
        try:
            collection = self.vector_store.collection

            if collection is None:
                raise ValueError("Collection not initialized")

            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.top_k
            )
            # Process result
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
            
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                        # Convert distance to similarity score (ChromaDB uses cosine distance)
                        similarity_score = 1 - distance
                        
                        if similarity_score >= self.score_threshold:
                            doc = Document(
                                page_content=document,
                                metadata={
                                    **metadata,
                                    # 'id': doc_id,
                                    # 'similarity_score': similarity_score,
                                    # 'distance': distance,
                                    # 'rank': i + 1
                                    }
                                )
                            retrieved_docs.append(doc)
                logging.info(f"Retrived {len(retrieved_docs)} documents after filtering")
            else:
                logging.info(f"No documents found")
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error in retrieving documents: {e}")
            raise
        