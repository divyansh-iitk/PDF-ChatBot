from rag.vectorstore import VectorStore
from rag.embeddings import Embedding_manager
from typing import List, Dict, Any
from utils.config import RetrieverConfig
from logger import logging

class RAGRetriver:
    """Handles query-based retrieval from the vector store"""
    def __init__(self, vector_store: VectorStore, embedding_manager: Embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
    def retrieve(self, query: str, top_k = RetrieverConfig.top_k, score_threshold = RetrieverConfig.score_threshold) -> List[Dict[str, Any]]:
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
                n_results=top_k
            )
            # Process result
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                if results['metadatas']: metadatas = results['metadatas'][0]
                if results['distances']: distances = results['distances'][0]
                ids = results['ids'][0]
            
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                        # Convert distance to similarity score (ChromaDB uses cosine distance)
                        similarity_score = 1 - distance
                        
                        if similarity_score >= score_threshold:
                            retrieved_docs.append({
                                'id': doc_id,
                                'content': document,
                                'metadata': metadata,
                                'similarity_score': similarity_score,
                                'distance': distance,
                                'rank': i + 1
                            })
                logging.info(f"Retrived {len(retrieved_docs)} documents after filtering")
            else:
                logging.info(f"No documents found")
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error in retrieving documents: {e}")
            raise
            return []
        