import chromadb
from logger import logging
from typing import List, Any
import numpy as np
import uuid
import os
from utils.config import VectorStoreConfig


class VectorStore:
    def __init__(self, collection_name:str = VectorStoreConfig.collection_name, persist_dir: str = VectorStoreConfig.persist_dir):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        self._initialise_store()

    def _initialise_store(self):
        ## initialising ChromaDB client and collection
        try:
            # Creating persistant ChromaDB client
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            
            # Get or creat collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            logging.info(f"Vector store initialised with collection name: {self.collection_name}")
            logging.info(f"Existing document in collection: {self.collection.count()}")
        except Exception as e:
            logging.error(f"Error in initialising vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        # Preparing data for ChromaDB
        ids = []
        metadatas = []
        document_texts = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata["index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Prepare document_texts
            document_texts.append(doc.page_content)
            
            # Prepare embeddings_list
            embeddings_list.append(embedding.tolist())
            
            # Add to collection
        try:
            if self.collection is None:
                raise ValueError("Collection is not initialized")

            if not (len(ids) == len(embeddings_list) == len(metadatas) == len(document_texts)):
                raise ValueError("Length mismatch in inputs")

            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=document_texts
            )

            logging.info(f"Successfully added {len(document_texts)} documents to vector store")
            logging.info(f"Total documents in the collection: {self.collection.count()}")

        except Exception as e:
            logging.error(f"Error adding to vector store: {e}")
            raise