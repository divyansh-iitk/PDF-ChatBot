from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from logger import logging
from typing import List
from utils.config import SplitterConfig
import uuid

def split_documents(documents: List[Document], chunk_size = SplitterConfig.chunk_size, chunk_overlap = SplitterConfig.chunk_overlap) -> List[Document]:
    """Chunking

    Args:
        documents (List[Document]): _description_
        chunk_size (int, optional): _description_. Defaults to 1000.
        chunk_overlap (int, optional): _description_. Defaults to 200.

    Returns:
        List[Document]: _description_
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )
    try:
        splitted_docs = text_splitter.split_documents(documents)
        
        # Adding UUID to every chunk
        for doc in splitted_docs:
            uid = str(uuid.uuid4())
            doc.metadata["uid"] = uid
        
    except Exception as e:
        logging.error(f"Error: {e}")
        raise
    logging.info(f"Splitted {len(documents)} documents into {len(splitted_docs)} chunks")
    return splitted_docs