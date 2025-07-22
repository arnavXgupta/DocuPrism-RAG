# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json

# Import the core logic from your main script
from main import process_claim

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Insurance Claim Adjudication API",
    description="An API that uses LLMs to process natural language insurance claims against policy documents.",
    version="1.0.0",
)

# 2. Define the input data model
# Pydantic models ensure that the data we receive is in the correct format.
class ClaimQuery(BaseModel):
    query: str

# 3. Define the API endpoint
@app.post("/process_claim/")
async def create_claim_decision(claim_query: ClaimQuery):
    """
    Receives a natural language query, processes it through the RAG chain,
    and returns a structured JSON decision.
    """
    if not claim_query.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Call the processing function from main.py
        result = process_claim(claim_query.query)

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process the claim. The RAG chain returned an error.")

        # FastAPI automatically converts dictionaries to JSON responses
        return result

    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# 4. Add a root endpoint for basic health check
@app.get("/")
def read_root():
    return {"status": "API is running. Use the /docs endpoint to see the documentation."}

# This block allows you to run the API directly for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)