# main.py

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# --- Pydantic Models for Structured Output ---
# This defines the exact schema for a single clause mapping
class ClauseMap(BaseModel):
    Decision: str = Field(description="The decision related to this specific clause (e.g., 'Rejected', 'Approved')")
    Clause: str = Field(description="The specific clause or code from the policy document, e.g., 'SECTION D) 2)'")
    Justification: str = Field(description="The reason this specific clause applies to the decision.")

# This is the main output schema the LLM must follow
class AdjudicationDecision(BaseModel):
    Decision: str = Field(description="The final decision for the claim, must be 'Approved', 'Rejected', or 'Needs Human Review'")
    Amount: str = Field(description="The amount covered. Must be '0' if rejected. Otherwise, state the coverage limit like 'Up to Sum Insured'.")
    Justification: str = Field(description="The overall justification for the final decision.")
    ClauseMapping: List[ClauseMap] = Field(description="A list of specific policy clauses that support the decision.")

# --- 1. Initialize the Language Model ---
# For entity extraction, a fast and efficient model is sufficient.
# We'll use gpt-3.5-turbo here.
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# In main.py, add this new function near the top

def expand_query_with_llm(user_query: str, llm_for_expansion: ChatOpenAI):
    """
    Uses an LLM to expand a simple user query into multiple, more specific questions.
    """
    print(f"\n--- Expanding Query: '{user_query}' ---")
    
    expansion_prompt_template = f"""
    You are an AI assistant specializing in insurance policies. Your task is to break down a user's query into a list of specific, detailed questions that need to be answered by searching a policy document.

    **User Query:** "{user_query}"

    **Generate a JSON list of 3 to 5 precise questions. For example:**
    - "What is the waiting period for [procedure]?"
    - "Does the policy cover [procedure] if it is due to an accident?"
    - "Are there any age or location-based restrictions for [procedure]?"
    - "What is the initial waiting period for any illness after the policy starts?"

    **Output only the minified JSON list of strings.**
    """
    
    messages = [SystemMessage(content=expansion_prompt_template)]
    
    try:
        response = llm_for_expansion.invoke(messages)
        response_json = response.content.strip()
        
        # Clean up potential markdown formatting
        if response_json.startswith("```json"):
            response_json = response_json.strip("```json\n").strip()
            
        expanded_queries = json.loads(response_json)
        print("✅ Expanded Queries:")
        for q in expanded_queries:
            print(f"  - {q}")
        return expanded_queries
    except Exception as e:
        print(f"❌ Error during query expansion: {e}")
        # Fallback to the original query
        return [user_query]

# --- 2. Define the Query Parsing Function ---
def parse_query_to_json(user_query: str):
    """
    Takes a natural language query and uses an LLM to extract structured
    information as a JSON object.

    Args:
        user_query: The user's query string.

    Returns:
        A dictionary containing the extracted entities.
    """
    print(f"\n--- Parsing Query: '{user_query}' ---")

    # The system prompt is our instruction manual for the LLM.
    # It clearly defines the persona, task, and desired output format.
    system_prompt = """
    You are an expert insurance analyst. Your task is to parse a user's query
    to extract key details for a claim. Analyze the query and return a clean,
    minified JSON object containing the following keys:

    - "age": (integer) The age of the person.
    - "gender": (string, "Male" or "Female") The gender of the person.
    - "procedure": (string) The medical procedure or reason for the claim.
    - "location": (string) The city or location where the procedure took place.
    - "policy_duration_months": (integer) The age of the policy in months.

    If any piece of information is not available in the query, you MUST use a value of null for that key.
    Do not make up information.
    """

    # We combine the system prompt with the user's actual query.
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    try:
        # Call the LLM with the prepared messages.
        response = llm.invoke(messages)
        # The LLM's response content should be a JSON string.
        response_json = response.content
        
        # Clean up potential markdown formatting from the response
        if response_json.startswith("```json"):
            response_json = response_json.strip("```json\n").strip()
            
        # Parse the JSON string into a Python dictionary.
        parsed_data = json.loads(response_json)
        print("✅ Parsed Data:")
        print(json.dumps(parsed_data, indent=2))
        return parsed_data

    except json.JSONDecodeError:
        print("❌ Error: Failed to decode JSON from the LLM response.")
        print(f"Raw Response: {response.content}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None
    
# --- 4. Load the Vector Database and Create Retriever ---
print("--- Loading Vector Database ---")
persist_directory = 'db'
embeddings = OpenAIEmbeddings()

# Load the persisted database from disk
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Create a retriever from the vector database
# This retriever will fetch the most relevant document chunks
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

print("✅ Vector Database Loaded.")

# --- 5. Craft the Reasoning Prompt Template ---
# This prompt is the "brain" of our system. It instructs the LLM on how to behave.
# It takes two inputs: `context` (the retrieved clauses) and `question` (the user's query).

# In main.py, REPLACE your old template variable with this advanced one

# advanced_template = """
# You are a hyper-vigilant insurance claims auditor. Your task is to adjudicate a claim based ONLY on the provided policy clauses. You must not miss any detail.

# **Policy Clauses (Context):**
# {context}

# **User's Claim Query:**
# {question}

# **Follow these steps precisely:**

# **Step 1: Deconstruct the Claim.**
# - Medical Procedure: (Identify the procedure from the query, e.g., 'cataract surgery')
# - Policy Duration: (Identify the policy age in months)
# - Claim Reason: (Is it due to an accident or an illness?)

# **Step 2: Find and Apply Relevant Rules from the Context.**
# - **CRITICAL CHECK**: First, scan the context for a "Specified disease/procedure waiting period" list. If the procedure from Step 1 is in that list, this rule takes top priority. Note the required waiting period.
# - **Other Waiting Periods**: Check for the 30-day initial waiting period and the pre-existing disease waiting period.
# - **Exclusions**: Check if the procedure is listed in any general or specific exclusion clauses.

# **Step 3: Synthesize the Final Decision.**
# - If a specific waiting period for the procedure was found and the policy duration is less than required, you MUST REJECT the claim.
# - If a specific waiting period was found and the policy duration is greater than or equal to the requirement, you MUST APPROVE the claim.
# - **SAFETY NET**: If the provided context is ambiguous, contradictory, or does NOT contain specific information about the procedure's waiting period or coverage, you MUST set the decision to "Needs Human Review" and state that the context is insufficient.

# **Step 4: Generate the Final JSON Output.**
# Based on your synthesis, create the final JSON object. Ensure the justification directly reflects the step-by-step reasoning you performed.

# **Your Detailed Analysis (Steps 1-3):**
# (You must write your detailed analysis here first)

# **Final JSON Output:**
# (Provide the clean, minified JSON here)
# """

# prompt = ChatPromptTemplate.from_template(advanced_template)

# --- 6. Initialize the Reasoning LLM ---
# For the final reasoning step, we need a powerful model.
# Let's use gpt-4o-mini as it's capable and cost-effective.
# For higher accuracy, you could swap this for "gpt-4-turbo".
reasoning_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

parser = JsonOutputParser(pydantic_object=AdjudicationDecision)

# --- Bind the desired JSON structure to the reasoning LLM ---
# This tells the LLM that its output MUST conform to the AdjudicationDecision schema.
structured_llm = reasoning_llm.with_structured_output(AdjudicationDecision)

# --- Update the prompt to be simpler, as the format is now enforced elsewhere ---
final_template = """
You are an expert insurance claims adjudicator. Your task is to evaluate a claim based ONLY on the provided context of policy clauses.

**Context from Policy Document:**
{context}

**User's Claim Query:**
{question}

**Instructions:**
1.  Analyze the user's query and the provided context.
2.  Determine if the claim should be Approved or Rejected based on waiting periods, exclusions, and other rules in the context.
3.  Provide a clear justification and map your decision to the specific clauses from the context.
4.  Generate a decision object that follows the required JSON schema.
"""

final_prompt = ChatPromptTemplate.from_template(final_template)

# --- 7. Chain Everything Together with LCEL ---
# LangChain Expression Language (LCEL) allows us to define the RAG chain declaratively.

def format_docs(docs):
    # Helper function to format the retrieved documents for the prompt
    return "\n\n".join(doc.page_content for doc in docs)



# In main.py, add this new verification function

def verify_decision(query: str, context: str, draft_json: dict, llm_for_verification: ChatOpenAI):
    """
    Uses an LLM as a verification agent to critique a draft decision.
    """
    print("\n--- Verifying Decision ---")
    
    verification_prompt_template = f"""
    You are an insurance audit supervisor. Your only job is to find errors in the following claim decision.
    
    - **Original Query**: {query}
    - **Retrieved Policy Clauses**: {context}
    - **Draft Decision JSON**: {json.dumps(draft_json, indent=2)}

    Critically analyze the draft decision. Check for the following:
    1.  **Context Sufficiency**: The most important check. Does the retrieved context contain the specific waiting period or exclusion clause for the procedure mentioned in the query (e.g., 'cataracts')? If the context seems to be missing this key information, the draft decision is unreliable regardless of its logic.
    2.  **Factual Inconsistency**: Does the justification misinterpret any of the provided policy clauses?
    3.  **Missed Clauses**: Did the decision ignore a more relevant clause that *was* present in the context?
    
    If the context is insufficient OR if there are factual/logical errors, describe the problem clearly.
    If and only if the context is sufficient AND the decision is logically sound and factually correct, respond with ONLY the word "OK".
    """
    
    messages = [SystemMessage(content=verification_prompt_template)]
    
    try:
        response = llm_for_verification.invoke(messages)
        critique = response.content.strip()
        
        if critique.upper() == "OK":
            print("✅ Verification Passed.")
            return None
        else:
            print(f"⚠️ Verification Failed. Critique: {critique}")
            return critique
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return "Verification process failed."

# This helper function will now handle the expanded retrieval
def get_retrieved_context(query: str, retriever, llm_for_expansion):
    # Expand the original query into a list of sub-queries
    expanded_queries = expand_query_with_llm(query, llm_for_expansion)
    
    # Retrieve documents for each expanded query
    all_retrieved_docs = []
    for q in expanded_queries:
        all_retrieved_docs.extend(retriever.invoke(q))
        
    # Remove duplicate documents to create a unique set of context
    unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()
    
    print(f"\n--- Retrieved {len(unique_docs)} unique context chunks ---")
    
    return format_docs(unique_docs)

# The new RAG chain definition
rag_chain = (
    {
        "context": RunnablePassthrough(lambda query: get_retrieved_context(query, retriever, llm)),
        "question": RunnablePassthrough()
    }
    | final_prompt
    | structured_llm # Use the new structured output LLM
    # The output from structured_llm is already a Pydantic object, so no string parser is needed.
)

# New process_claim function
# In main.py, replace the process_claim function with this final version

def process_claim_final(query: str):
    """
    Processes a claim and returns a guaranteed valid JSON object.
    """
    print(f"\n--- Starting Final Claim Process for Query: '{query}' ---")
    try:
        # The chain now directly outputs a Pydantic object, which acts like a dictionary.
        result_object = rag_chain.invoke(query)
        
        # We can convert the Pydantic object to a dictionary for JSON serialization if needed
        result_data = result_object.model_dump()
        
        print("\n✅ Final, Structurally-Guaranteed Decision JSON:")
        print(json.dumps(result_data, indent=2))
        return result_data

    except Exception as e:
        print(f"❌ An error occurred during the final claim process: {e}")
        return None
    
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | reasoning_llm
#     | StrOutputParser()
# )

# # --- 8. Create a Function to Run the Full Process ---
# def process_claim(query: str):
#     """
#     Processes a user's claim from raw query to final JSON decision.
#     """
#     print(f"\n--- Processing Claim for Query: '{query}' ---")
#     try:
#         # We don't need the structured JSON from Day 2 for the RAG chain itself,
#         # but it's good practice to have it. The full query provides enough context.
#         # parsed_info = parse_query_to_json(query)
#         # if not parsed_info:
#         #     return "Could not parse query."
            
#         # The rag_chain takes the raw query string and handles the rest.
#         result_json_str = rag_chain.invoke(query)
        
#         # Parse the final JSON string for clean output
#         result_data = json.loads(result_json_str)
        
#         print("\n✅ Final Decision JSON:")
#         print(json.dumps(result_data, indent=2))
#         return result_data

#     except Exception as e:
#         print(f"❌ An error occurred during claim processing: {e}")
#         return None


# Update the main execution block
if __name__ == "__main__":
    # Test Case: The cataract surgery query
    # query1 = "Claim for cataract surgery, policy is 30 months old"
    # process_claim_final(query1)

    # # Test Case 2: A query that should be approved (post 30-day wait)
    # query2 = "I need to file for a hernia operation. My policy is 1 year old."
    # process_claim(query2)
    
    query3 = "Claim for a condition I declared at purchase. The policy is now 40 months old."
    process_claim_final(query3)
    
    # query4 = "I need to claim for cosmetic plastic surgery. Policy is 5 years old."
    # process_claim(query4)
    
    # query5 = "Claim for a joint replacement surgery needed due to a road accident. The policy is 6 months old."
    # process_claim(query5)
    
    query6 = "Is my treatment covered?"
    process_claim_final(query6)
    
    # query7 = "Claim for infertility treatment. Policy is 10 years old."
    # process_claim(query7)
