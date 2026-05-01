from ragas import evaluate
from typing import List
from rag.vectorstore import VectorStore
from rag.embeddings import Embedding_manager
from datasets import Dataset
from rag.retriever import RAGRetriver
from rag.chain import GroqLLM

from ragas.metrics import (
    faithfulness,
    answer_relevancy
)

def eval_result(queries: List[str]):
    answers = []
    contexts = []
    vector_store = VectorStore()
    embedding_manager = Embedding_manager()
    retriever = RAGRetriver(vector_store, embedding_manager)
    llm = GroqLLM()
    for query in queries:
        retrieved_doc = retriever.retrieve(query)
        llm_response= llm.generate_response(query, retrieved_doc)
        answers.append(llm_response)
        contexts.append([doc["content"] for doc in retrieved_doc])
    
    data = {
        "question": queries,
        "answer": answers,
        "contexts": contexts
    }
    
    dataset = Dataset.from_dict(data)
    
    results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy
    ]
    )
    return results