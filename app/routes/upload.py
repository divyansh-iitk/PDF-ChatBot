from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import os
from rag.ingest import ingest_pdf
from utils.config import IngestConfig
from logger import logging

router = APIRouter()

UPLOAD_DIR = IngestConfig.pdf_dir
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF and ingest into vector DB
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save file if needed
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(await file.read())
            logging.info(f"File saved at {file_path}")
            flag = True
        else:
            logging.info(f"File already exists at {file_path}, skipping save.")
            flag = False

        # ALWAYS ingest
        vector_store = request.app.state.vector_store
        embedding_manager = request.app.state.embedding_manager

        request.app.state.bm25_retriever = ingest_pdf(
            file_path, embedding_manager, vector_store,flag = flag
        )

        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename
        }

    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Error processing PDF")