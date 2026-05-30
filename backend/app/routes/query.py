from fastapi import APIRouter, HTTPException, Request
from app.schemas.query import QueryRequest
from logger import logging
from rag.retriever import RAGRetriver
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from rag.cohere_reranker import Cohere_Reranker
from utils.config import EnsembleConfig, CohereConfig, ChathistoryConfig

router = APIRouter()


@router.post("/query")
async def query_pdf(request: Request, body: QueryRequest):
    """
    Query ingested PDFs
    """

    try:
        # Step 1: Retrieve documents
        llm = request.app.state.llm
        vector_store = request.app.state.vector_store
        embedding_manager = request.app.state.embedding_manager
        vector_retriever = RAGRetriver(vector_store=vector_store, embedding_manager=embedding_manager)
        bm25_retriever = request.app.state.bm25_retriever
        ensemble_retriever = EnsembleRetriever(

            retrievers=[bm25_retriever, vector_retriever],

            weights=EnsembleConfig.weights,
            id_key=EnsembleConfig.id_key

        )
        
        cohere_reranker = Cohere_Reranker(model_name=CohereConfig.model_name,
                                          top_n=CohereConfig.top_n,
                                          base_retriever=ensemble_retriever)
        
        if not request.app.state.chat_history:
            rewritten_query = body.question
            history = "There is no chat history"
        else:
            history = "\n".join([
                f"User: {chat['HumanMessage']}\nAssistant: {chat['AIMessage']}"
                for chat in request.app.state.chat_history[-ChathistoryConfig.last_n_chats:]
            ])
            rewritten_query = llm.rewrite_query(query=body.question, chat_history=history)
           
        retrieved_docs = cohere_reranker.rerank(query=rewritten_query.lower().strip())

        if not retrieved_docs:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        # Step 2: Generate answer
        answer = llm.generate_response(
            query=body.question,
            retrieved_docs=retrieved_docs,
            chat_history=history
        )

        chat = {
                "HumanMessage":body.question,
                "AIMessage":answer
            }
        
        request.app.state.chat_history.append(chat)
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in retrieved_docs
            ],
            "num_sources": len(retrieved_docs),
            "chat_history": history,
            "rewritten_query": rewritten_query
            
        }

    except Exception as e:
        logging.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")