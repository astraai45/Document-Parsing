import fitz  # PyMuPDF
import docx
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import logging

# Document Loading
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    logging.info("Loaded the pdf file")
    return text

def load_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    logging.info("Loaded the docx file")
    return text

# Text Chunking   
def chunk_text(text, chunk_size=50):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Embedding Functions
# BGE
bge_model_name = "sentence-transformers/all-MiniLM-L6-v2"
bge_tokenizer = AutoTokenizer.from_pretrained(bge_model_name)
bge_model = AutoModel.from_pretrained(bge_model_name)

def embed_with_bge(text):
    inputs = bge_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bge_model(**inputs)
    # Convert the numpy array to a list
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

# Sentence Transformer
st_model_name = "sentence-transformers/all-MiniLM-L6-v2"
st_tokenizer = AutoTokenizer.from_pretrained(st_model_name)
st_model = AutoModel.from_pretrained(st_model_name)

def embed_with_sentence_transformer(text):
    inputs = st_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = st_model(**inputs)
    # Convert the numpy array to a list
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

# Storing in ChromeDB
client = chromadb.Client()
collection = client.get_or_create_collection("document_embeddings")

def clear_embeddings(doc_id):
    embeddings_to_delete = []
    results = collection.get(include=["metadatas"])
    for embedding_id in results["metadatas"]:
        if results["metadatas"][embedding_id]["doc_id"] == doc_id:
            embeddings_to_delete.append(embedding_id)
    
    if embeddings_to_delete:
        collection.delete(ids=embeddings_to_delete)
        logging.info(f"Cleared existing embeddings for {doc_id}")


def store_embeddings(doc_id, chunks, embeddings):
    for i, embedding in enumerate(embeddings):
        chroma_doc_id = f"{doc_id}_{i}"
        collection.add(chroma_doc_id, [embedding], {"chunk": chunks[i], "doc_id": doc_id})
        logging.info(f"Stored embedding for {chroma_doc_id}: {embedding}")

def chroma_retrival(query_texts):
    results = collection.query(
        query_texts=query_texts,  
        n_results=3,
        #include = ['metadatas', 'documents', 'distances']
    )
    logging.info(f'Query Results: {results}')
    
    return results
