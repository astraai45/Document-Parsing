from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import os
import shutil
import logging
from .embedding_utils import load_pdf, load_docx, chunk_text, embed_with_bge, embed_with_sentence_transformer, store_embeddings, clear_embeddings

router = APIRouter()

UPLOAD_DIRECTORY = "./uploaded_documents"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@router.post("/")
async def upload_file(file: UploadFile = File(...), embed_method: str = Form(...)):
    logging.info("I'm in Upload Files Section")
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if file.filename.endswith(".pdf"):
        text = load_pdf(file_path)
    elif file.filename.endswith(".docx"):
        text = load_docx(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Chunking Part
    chunks = chunk_text(text)
    logging.info('Chunking of the Uploaded Document Done')

    if embed_method == "bge":
        embed_func = embed_with_bge
    elif embed_method == "sentence_transformer":
        embed_func = embed_with_sentence_transformer
    else:
        raise HTTPException(status_code=400, detail="Invalid embedding method")

    embeddings = [embed_func(chunk) for chunk in chunks]
    logging.info(f'Embeddings: {embeddings}')
    logging.info(f'Chunks: {chunks}')

    logging.info(f'Embeddings of the document done with {embed_func.__name__}')
    logging.info("Embeddings of the Text Done !!!")
    
    # Clear existing embeddings for this document
    clear_embeddings(file.filename)
    
    # Store new embeddings
    store_embeddings(file.filename, chunks, embeddings)
    logging.info("Stored the Embeddings in the ChromaDB")
    
    return {"filename": file.filename, "file_path": file_path}
