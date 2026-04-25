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
    persist_dir: str = "db"
    
@dataclass
class RetrieverConfig:
    top_k: int = 5
    score_threshold: float = 0.0
    
@dataclass
class IngestConfig:
    pdf_dir: str = "data/pdfs"