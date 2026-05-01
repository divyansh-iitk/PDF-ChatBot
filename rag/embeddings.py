import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from utils.config import EmbeddingConfig
from logger import logging
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class Embedding_manager:
    def __init__(self, model_name: str = EmbeddingConfig.embedding_model):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            logging.info(f"Embedding model loaded successfully. Embedding domensions = {self.model.get_embedding_dimension()}")
        except Exception as e:
            logging.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        logging.info(f"Generated embedding of shape: {embeddings.shape}")
        return embeddings