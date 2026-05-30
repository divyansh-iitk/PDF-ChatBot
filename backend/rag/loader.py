from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from typing import List
from logger import logging

            
#-------------#
def process_pdf(pdf_file) -> List[Document]:
    """ Reads pdf, add some metadata

    Args:
        pdf_file (_type_): _description_

    Returns:
        Document: _description_
    """
    pdf_path = Path(pdf_file)
    try:
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        
        ## Assigning aditional metadata
        for doc in documents:
            doc.metadata["source_file"] = pdf_path.name
            doc.metadata["file_type"] = "pdf"
    except Exception as e:
        logging.error(f"Error: {e}")
        raise
    logging.info(f"PDF loaded.")
    return list(documents)
#-------------------#