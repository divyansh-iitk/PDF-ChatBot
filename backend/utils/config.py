from dataclasses import dataclass


@dataclass
class SplitterConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class EmbeddingConfig:
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    
    
@dataclass
class VectorStoreConfig:
    collection_name: str = "pdf_documents"
    persist_dir: str = "app_data/db"
    
@dataclass
class RetrieverConfig:
    top_k: int = 10
    score_threshold: float = 0.5
    
@dataclass
class IngestConfig:
    pdf_dir: str = "app_data/data/pdfs"
    
    
@dataclass
class BM25Config:
    top_k: int = 10
    

@dataclass
class EnsembleConfig:
    weight_BM25 = 0.5
    weigth_vector_retriever = 0.5
    weights = [weight_BM25, weigth_vector_retriever]
    id_key = "uid"
    

@dataclass
class CohereConfig:
    model_name = "rerank-v3.5"
    top_n = 4
    

@dataclass
class LLM_GroqConfig:
    model_name = "llama-3.3-70b-versatile"
    temperature=0.1
    max_tokens=1024
    
    
@dataclass
class ChathistoryConfig:
    last_n_chats = 3
    