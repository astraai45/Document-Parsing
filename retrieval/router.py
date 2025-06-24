from fastapi import APIRouter, HTTPException, Form
import logging
from .embedding_utils import chroma_retrival



router = APIRouter()

@router.post("/")
async def retrieve_information(query: str = Form(...)):
    logging.info("Retrieving Information from the ChromaDB")
    retrieval_results = chroma_retrival(query)
    logging.info("Retrieved Information from the ChromaDB")
    
    return {"query": query, "retrieval_results": retrieval_results}
