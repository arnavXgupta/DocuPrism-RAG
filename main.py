# # --- Dependencies ---
# # pip install fastapi uvicorn pydantic httpx python-dotenv "pinecone-client[grpc]" langchain-pinecone

# import os
# import logging
# from dotenv import load_dotenv
# from typing import List, Dict
# from contextlib import asynccontextmanager

# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel, HttpUrl
# from pinecone import Pinecone as PineconeClient

# # Load environment variables from .env file
# load_dotenv()

# # --- Import functions from your RAG pipeline script ---
# from rag_pipeline import (
#     get_or_create_vectors,
#     find_answers_with_pinecone,
# )

# # =====================================================================================
# # LOGGING & AUTHENTICATION
# # =====================================================================================
# app_logger = logging.getLogger("app_logger") # Basic setup assumed
# # ... full logging setup ...

# security = HTTPBearer()
# BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     """Dependency function to validate the bearer token."""
#     if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
#         app_logger.warning("Invalid authentication attempt.")
#         raise HTTPException(status_code=403, detail="Invalid or missing authentication token")
#     return credentials

# # =====================================================================================
# # FASTAPI APPLICATION LIFESPAN & SETUP
# # =====================================================================================

# # A dictionary to hold application state, like our Pinecone client
# app_state: Dict = {}

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Application startup...")
#     pinecone_api_key = os.getenv("PINECONE_API_KEY")
#     if not pinecone_api_key:
#         raise ValueError("PINECONE_API_KEY must be set in the environment!")
#     app_state["pinecone_client"] = PineconeClient(api_key=pinecone_api_key)
#     print(f"Pinecone client initialized.")
#     yield
#     print("Application shutdown...")
#     app_state.clear()


# app = FastAPI(
#     title="Insurance RAG API with Pinecone (Corrected)",
#     description="An API that uses a persistent Pinecone vector store with corrected client logic.",
#     lifespan=lifespan
# )

# class QueryInput(BaseModel):
#     documents: HttpUrl
#     questions: List[str]

# class AnswerOutput(BaseModel):
#     answers: List[str]

# @app.post("/api/v1/hackrx/run", response_model=AnswerOutput, dependencies=[Depends(validate_token)])
# async def run_rag_pipeline(payload: QueryInput):
#     """
#     This endpoint executes the full RAG pipeline with corrected, manual Pinecone operations.
#     """
#     pinecone_client = app_state.get("pinecone_client")
#     pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")
#     if not pinecone_index_host:
#         raise HTTPException(status_code=500, detail="PINECONE_INDEX_HOST is not set.")
#     if not pinecone_client:
#         raise HTTPException(status_code=500, detail="Pinecone client not initialized.")
    
#     try:
#         doc_url = str(payload.documents)

#         # Step 1: Ensure vectors for the document exist in Pinecone.
#         # This function now handles ingestion if necessary.
#         await get_or_create_vectors(pinecone_client, pinecone_index_host, doc_url)
        
#         queries_with_ids = [{"id": f"q_{i+1}", "question": q} for i, q in enumerate(payload.questions)]
        
#         # Step 2: Find answers using the persistent vector store.
#         # This function now handles structuring, querying, and generation.
#         answer_list = await find_answers_with_pinecone(pinecone_client, pinecone_index_host, doc_url, queries_with_ids)
        
#         return AnswerOutput(answers=answer_list)
        
#     except Exception as e:
#         app_logger.error(f"An unexpected error occurred: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# @app.get("/")
# def read_root():
#     return {"status": "API is running. Use /docs for documentation."}

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}

# import os
# import logging
# import httpx
# from dotenv import load_dotenv
# from typing import List, Dict
# from contextlib import asynccontextmanager

# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel, HttpUrl
# from pinecone import Pinecone as PineconeClient

# # Load environment variables
# load_dotenv()

# # --- Import functions from your RAG pipeline and the solver ---
# from rag_pipeline import (
#     get_or_create_vectors,
#     find_answers_with_pinecone,
# )
# # This is the hardcoded solver script from our previous discussion
# from solver import flight_solver_logic 

# # ... (Logging, Auth, and Lifespan setup remains the same) ...
# app_logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
# security = HTTPBearer()
# BEARER_TOKEN = os.getenv("BEARER_TOKEN")
# def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if not BEARER_TOKEN:
#         return credentials
#     if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
#         raise HTTPException(status_code=403, detail="Invalid or missing authentication token")
#     return credentials
# app_state: Dict = {}
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     app_logger.info("Application startup...")
#     pinecone_api_key = os.getenv("PINECONE_API_KEY")
#     if not pinecone_api_key:
#         raise ValueError("PINECONE_API_KEY must be set in the environment!")
#     app_state["pinecone_client"] = PineconeClient(api_key=pinecone_api_key)
#     app_logger.info("Pinecone client initialized.")
#     app_state["httpx_client"] = httpx.AsyncClient()
#     yield
#     app_logger.info("Application shutdown...")
#     await app_state["httpx_client"].aclose()
#     app_state.clear()
# app = FastAPI(
#     title="Intelligent RAG API with Hybrid Processing",
#     description="This API handles mixed-task queries by categorizing questions and batch-processing them.",
#     lifespan=lifespan,
# )
# # This model is defined in solver_script but is needed here as well
# class AnswerOutput(BaseModel):
#     answers: List[str]
# class QueryInput(BaseModel):
#     documents: HttpUrl
#     questions: List[str]
# # --- End of boilerplate ---


# # =====================================================================================
# # API ENDPOINT WITH NEW HYBRID PROCESSING LOGIC
# # =====================================================================================

# @app.post("/api/v1/hackrx/run", response_model=AnswerOutput, dependencies=[Depends(validate_token)])
# async def run_hybrid_rag_pipeline(payload: QueryInput):
#     """
#     This intelligent endpoint handles mixed-question payloads. It categorizes each
#     question and processes them in efficient batches (one for the solver, one for RAG),
#     then reassembles the answers in the correct order.
#     """
#     httpx_client = app_state.get("httpx_client")
#     if not httpx_client:
#         raise HTTPException(status_code=500, detail="HTTP client not initialized.")

#     # --- Step 1: Categorize all questions ---
#     app_logger.info("Categorizing questions into 'Solver' and 'RAG' batches.")
#     solver_question_indices = []
#     rag_questions_with_indices = []

#     for index, question in enumerate(payload.questions):
#         if "flight number" in question.lower():
#             solver_question_indices.append(index)
#         else:
#             rag_questions_with_indices.append({"question": question, "original_index": index})

#     # This list will hold the final answers, in the correct order.
#     final_answers = [None] * len(payload.questions)

#     try:
#         # --- Step 2: Process the Solver batch (if any) ---
#         if solver_question_indices:
#             app_logger.info(f"Found {len(solver_question_indices)} solver question(s). Running solver once.")
#             # Run the solver logic only ONCE.
#             solver_result = await flight_solver_logic(httpx_client)
#             flight_number = solver_result.answers[0]
            
#             # Place the answer in all the slots that asked for it.
#             for index in solver_question_indices:
#                 final_answers[index] = flight_number

#         # --- Step 3: Process the RAG batch (if any) ---
#         if rag_questions_with_indices:
#             app_logger.info(f"Found {len(rag_questions_with_indices)} RAG question(s). Running RAG pipeline.")
#             pinecone_client = app_state.get("pinecone_client")
#             pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")
#             if not all([pinecone_client, pinecone_index_host]):
#                  raise HTTPException(status_code=500, detail="Pinecone is not configured correctly.")

#             # A) Vectorize the document, as it's needed for RAG.
#             await get_or_create_vectors(pinecone_client, pinecone_index_host, str(payload.documents))

#             # B) Prepare the batch of questions for the RAG pipeline.
#             rag_questions_batch = [{"question": item["question"]} for item in rag_questions_with_indices]
            
#             # C) Call the RAG pipeline ONCE with all standard questions.
#             rag_answer_list = await find_answers_with_pinecone(pinecone_client, pinecone_index_host, str(payload.documents), rag_questions_batch)

#             # D) Place the RAG answers back into their original positions.
#             for i, rag_answer in enumerate(rag_answer_list):
#                 original_index = rag_questions_with_indices[i]["original_index"]
#                 final_answers[original_index] = rag_answer
        
#         # --- Step 4: Final Assembly and Response ---
#         app_logger.info("All batches processed. Assembling final response.")
#         return AnswerOutput(answers=final_answers)

#     except HTTPException:
#         raise
#     except Exception as e:
#         app_logger.error(f"An unexpected error occurred during hybrid processing: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# @app.get("/", include_in_schema=False)
# def read_root():
#     return {"status": "API is running. Use /docs for documentation."}

# --- Dependencies ---
# pip install fastapi uvicorn pydantic httpx python-dotenv "pinecone-client[grpc]" langchain-openai "unstructured[pdf]" PyMuPDF langdetect tiktoken googletrans langchain-community

import os
import logging
import httpx
from dotenv import load_dotenv
from typing import List, Dict
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

# --- Import functions from your RAG pipeline and the solver ---
from rag_pipeline import (
    get_or_create_vectors,
    find_answers_with_pinecone,
)
# This is the hardcoded solver script from our previous discussion
from solver import flight_solver_logic 

# --- Logging, Auth, and Lifespan setup ---
app_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
security = HTTPBearer()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not BEARER_TOKEN:
        # If no token is set in the environment, skip validation
        return credentials
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing authentication token")
    return credentials

app_state: Dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application lifecycle for startup and shutdown events.
    Initializes and terminates essential resources like the Pinecone client,
    HTTP client, and a process pool for heavy computations.
    """
    app_logger.info("Application startup...")
    # Initialize Pinecone Client
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY must be set in the environment!")
    app_state["pinecone_client"] = PineconeClient(api_key=pinecone_api_key)
    app_logger.info("Pinecone client initialized.")
    
    # Initialize HTTPX Client
    app_state["httpx_client"] = httpx.AsyncClient()
    
    # Initialize a Process Pool Executor for parallel CPU-bound tasks (like document chunking)
    executor = ProcessPoolExecutor()
    app_state["process_pool_executor"] = executor
    app_logger.info(f"Process pool executor started with {executor._max_workers} workers.")

    yield # Application is now running

    app_logger.info("Application shutdown...")
    # Gracefully shut down resources
    await app_state["httpx_client"].aclose()
    app_state["process_pool_executor"].shutdown(wait=True)
    app_state.clear()
    app_logger.info("Resources cleaned up.")


app = FastAPI(
    title="Intelligent RAG API with Hybrid Processing",
    description="This API handles mixed-task queries by categorizing questions and batch-processing them. It uses a parallelized ingestion pipeline for new documents.",
    lifespan=lifespan,
)

class AnswerOutput(BaseModel):
    answers: List[str]

class QueryInput(BaseModel):
    documents: HttpUrl
    questions: List[str]

# =====================================================================================
# API ENDPOINT WITH NEW HYBRID PROCESSING LOGIC
# =====================================================================================

@app.post("/api/v1/hackrx/run", response_model=AnswerOutput, dependencies=[Depends(validate_token)])
async def run_hybrid_rag_pipeline(payload: QueryInput):
    """
    This intelligent endpoint handles mixed-question payloads. It categorizes each
    question and processes them in efficient batches (one for the solver, one for RAG),
    then reassembles the answers in the correct order.
    
    If the document URL is new, it will trigger a parallelized, CPU-intensive
    ingestion process to chunk and embed the document before proceeding.
    """
    httpx_client = app_state.get("httpx_client")
    pinecone_client = app_state.get("pinecone_client")
    executor = app_state.get("process_pool_executor")
    
    if not all([httpx_client, pinecone_client, executor]):
        raise HTTPException(status_code=500, detail="Core application components not initialized.")

    # --- Step 1: Categorize all questions ---
    app_logger.info("Categorizing questions into 'Solver' and 'RAG' batches.")
    solver_question_indices = []
    rag_questions_with_indices = []

    for index, question in enumerate(payload.questions):
        if "flight number" in question.lower():
            solver_question_indices.append(index)
        else:
            rag_questions_with_indices.append({"question": question, "original_index": index})

    # This list will hold the final answers, in the correct order.
    final_answers = [None] * len(payload.questions)

    try:
        # --- Step 2: Process the Solver batch (if any) ---
        if solver_question_indices:
            app_logger.info(f"Found {len(solver_question_indices)} solver question(s). Running solver once.")
            # Run the solver logic only ONCE.
            solver_result = await flight_solver_logic(httpx_client)
            flight_number = solver_result.answers[0]
            
            # Place the answer in all the slots that asked for it.
            for index in solver_question_indices:
                final_answers[index] = flight_number

        # --- Step 3: Process the RAG batch (if any) ---
        if rag_questions_with_indices:
            app_logger.info(f"Found {len(rag_questions_with_indices)} RAG question(s). Running RAG pipeline.")
            pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")
            if not pinecone_index_host:
                 raise HTTPException(status_code=500, detail="Pinecone is not configured correctly.")

            # A) Ingest the document if it's new. This now uses the parallel pipeline.
            await get_or_create_vectors(
                pinecone_client, 
                pinecone_index_host, 
                str(payload.documents), 
                executor
            )

            # B) Prepare the batch of questions for the RAG pipeline.
            rag_questions_batch = [{"question": item["question"]} for item in rag_questions_with_indices]
            
            # C) Call the RAG pipeline ONCE with all standard questions.
            rag_answer_list = await find_answers_with_pinecone(pinecone_client, pinecone_index_host, str(payload.documents), rag_questions_batch)

            # D) Place the RAG answers back into their original positions.
            for i, rag_answer in enumerate(rag_answer_list):
                original_index = rag_questions_with_indices[i]["original_index"]
                final_answers[original_index] = rag_answer
        
        # --- Step 4: Final Assembly and Response ---
        app_logger.info("All batches processed. Assembling final response.")
        return AnswerOutput(answers=final_answers)

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"An unexpected error occurred during hybrid processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
def read_root():
    return {"status": "API is running. Use /docs for documentation."}