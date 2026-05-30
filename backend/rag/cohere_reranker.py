from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from typing import List
from logger import logging


class Cohere_Reranker:
    def __init__(self, model_name:str, top_n:int, base_retriever):
        self.model_name = model_name
        self.top_n = top_n
        self.base_retriever = base_retriever
        self.compressor = self._setting_compressor()

    def _setting_compressor(self):
        try:
            compressor = CohereRerank( 
                model=self.model_name,
                top_n=self.top_n
            )
            return compressor
        except Exception as e:
            logging.error(f"Error while loading cohere compressor: {e}")
            raise
        
        
    def rerank(self, query:str) -> List[Document]:
        """_summary_

        Args:
            query (str): _description_

        Returns:
            List[Document]: _description_
        """
        try:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=self.base_retriever
            )
            
            docs = compression_retriever.invoke(query)
            
            return docs
        except Exception as e:
            logging.error(f"Error while loading cohere retriever: {e}")
            raise
    
if __name__ == "__main__":
    
    reranker = Cohere_Reranker("rerank-v3.5", 4)