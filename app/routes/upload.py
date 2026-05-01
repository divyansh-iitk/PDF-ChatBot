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
        # Save file
        if os.path.exists(file_path):
            logging.info(f"File already exists at {file_path}, skipping save.")
        else:
            with open(file_path, "wb") as f:
                f.write(await file.read())
            logging.info(f"File saved at {file_path}")

            # Run ingestion
            vector_store = request.app.state.vector_store
            embedding_manager = request.app.state.embedding_manager
            ingest_pdf(file_path, embedding_manager, vector_store)

            return {
                "message": "PDF uploaded and processed successfully",
                "filename": file.filename
            }

    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Error processing PDF")