from fastapi import APIRouter, HTTPException, Request
from app.schemas.query import QueryRequest
from logger import logging
from rag.retriever import RAGRetriver
from rag.ensemble_retriever import ensemble_retriever

router = APIRouter()


@router.post("/query")
async def query_pdf(request: Request, body: QueryRequest):
    """
    Query ingested PDFs
    """

    try:
        # Step 1: Retrieve documents
        vector_store = request.app.state.vector_store
        embedding_manager = request.app.state.embedding_manager
        vector_retriever = RAGRetriver(vector_store=vector_store, embedding_manager=embedding_manager)
        bm25_retriever = request.app.state.bm25_retriever        
        retrieved_docs = ensemble_retriever(bm25_retriever, vector_retriever, body)

        if not retrieved_docs:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        # Step 2: Generate answer
        llm = request.app.state.llm
        answer = llm.generate_response(
            query=body.question,
            retrieved_docs=retrieved_docs
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in retrieved_docs
            ],
            "num_sources": len(retrieved_docs)
        }

    except Exception as e:
        logging.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")