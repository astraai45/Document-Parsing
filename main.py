from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from upload.router import router as upload_router
from retrieval.router import router as retrieval_router
import logging

# Logging the Information
logging.basicConfig(level=logging.INFO, filename='Application.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the Application
app = FastAPI()

app.include_router(upload_router, prefix="/upload")
app.include_router(retrieval_router, prefix="/retrieve")

@app.get("/")
def read_root():
    logger.info("This is the main Welcome Page")
    return {"message": "Welcome to the Document Ingestion API!"}

app.mount("/static", StaticFiles(directory="static"), name="statics")
