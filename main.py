# --- Dependencies ---
# pip install fastapi uvicorn pydantic httpx python-dotenv "pinecone-client[grpc]" langchain-pinecone

import os
import logging
from dotenv import load_dotenv
from typing import List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from pinecone import Pinecone as PineconeClient

# Load environment variables from .env file
load_dotenv()

# --- Import functions from your RAG pipeline script ---
from rag_pipeline import (
    get_or_create_vectors,
    find_answers_with_pinecone,
)

# =====================================================================================
# LOGGING & AUTHENTICATION
# =====================================================================================
app_logger = logging.getLogger("app_logger") # Basic setup assumed
# ... full logging setup ...

security = HTTPBearer()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency function to validate the bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        app_logger.warning("Invalid authentication attempt.")
        raise HTTPException(status_code=403, detail="Invalid or missing authentication token")
    return credentials

# =====================================================================================
# FASTAPI APPLICATION LIFESPAN & SETUP
# =====================================================================================

# A dictionary to hold application state, like our Pinecone client
app_state: Dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY must be set in the environment!")
    app_state["pinecone_client"] = PineconeClient(api_key=pinecone_api_key)
    print(f"Pinecone client initialized.")
    yield
    print("Application shutdown...")
    app_state.clear()


app = FastAPI(
    title="Insurance RAG API with Pinecone (Corrected)",
    description="An API that uses a persistent Pinecone vector store with corrected client logic.",
    lifespan=lifespan
)

class QueryInput(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerOutput(BaseModel):
    answers: List[str]

@app.post("/api/v1/hackrx/run", response_model=AnswerOutput, dependencies=[Depends(validate_token)])
async def run_rag_pipeline(payload: QueryInput):
    """
    This endpoint executes the full RAG pipeline with corrected, manual Pinecone operations.
    """
    pinecone_client = app_state.get("pinecone_client")
    pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")
    if not pinecone_index_host:
        raise HTTPException(status_code=500, detail="PINECONE_INDEX_HOST is not set.")
    if not pinecone_client:
        raise HTTPException(status_code=500, detail="Pinecone client not initialized.")
    
    try:
        doc_url = str(payload.documents)

        # Step 1: Ensure vectors for the document exist in Pinecone.
        # This function now handles ingestion if necessary.
        await get_or_create_vectors(pinecone_client, pinecone_index_host, doc_url)
        
        queries_with_ids = [{"id": f"q_{i+1}", "question": q} for i, q in enumerate(payload.questions)]
        
        # Step 2: Find answers using the persistent vector store.
        # This function now handles structuring, querying, and generation.
        answer_list = await find_answers_with_pinecone(pinecone_client, pinecone_index_host, doc_url, queries_with_ids)
        
        return AnswerOutput(answers=answer_list)
        
    except Exception as e:
        app_logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "API is running. Use /docs for documentation."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}