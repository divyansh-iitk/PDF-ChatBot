from langchain_classic.retrievers.ensemble import EnsembleRetriever
from typing import List
from langchain_core.documents import Document
from app.schemas.query import QueryRequest
from utils.config import EnsembleConfig

def ensemble_retriever(retriever1, retriever2, body: QueryRequest) -> List[Document]:
    """It takes two retrievers and combine there retrived results so that there are no 
    duplicated, also it provides a way to add weight to the retrievers.

    Args:
        retriever1 (_type_): _description_
        retriever2 (_type_): _description_
        body (QueryRequest): _description_

    Returns:
        List[Document]: _description_
    """
    ensemble_retriever = EnsembleRetriever(

        retrievers=[retriever1, retriever2],

        weights=EnsembleConfig.weights,
        id_key=EnsembleConfig.id_key

    )
    retrieved_docs = ensemble_retriever.invoke(body.question.lower().strip())
    
    return retrieved_docs
    