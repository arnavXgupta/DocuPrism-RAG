# --- Dependencies ---
# pip install langchain-community "unstructured[docx,eml,pdf]" httpx langchain-pinecone "pinecone-client[grpc]" langchain-openai requests

import os
import io
import httpx
import json
import hashlib
import asyncio
from functools import partial
from pinecone import Pinecone as PineconeClient
from urllib.parse import urlparse
from typing import List, Dict, Any

# Langchain document object and text splitter
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Langchain embedding models and vector stores
from langchain_openai import OpenAIEmbeddings
# import spacy
# print("Initializing spaCy model for local query structuring...")
# try:
#     NLP = spacy.load("en_core_web_sm")
#     print("spaCy model initialized.")
# except OSError:
#     print("spaCy model not found. Please run 'python -m spacy download en_core_web_sm'")
#     NLP = None

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
BATCH_SIZE = 5
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"


# --- INITIALIZE EMBEDDING MODEL ---
print("Initializing OpenAI embedding model...")
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
print("OpenAI embedding model initialized.")


def _create_namespace_from_url(url: str) -> str:
    """Creates a stable, clean namespace by normalizing and hashing a URL."""
    parsed_url = urlparse(url.lower())
    normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    return hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans metadata to ensure all values are Pinecone-compatible types."""
    cleaned_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned_metadata[key] = value
        elif isinstance(value, list) and all(isinstance(i, str) for i in value):
            cleaned_metadata[key] = value
        else:
            cleaned_metadata[key] = json.dumps(value, default=str)
    return cleaned_metadata


# --- ADDED: Helper function to count words instead of characters ---
def count_words(text: str) -> int:
    """Helper function to count words in a text string."""
    return len(text.split())


# def load_and_chunk_document(url: str, document_bytes: bytes) -> List[Document]:
#     """Takes the raw bytes of a document, chunks it, and returns LangChain Document objects."""
#     print(f"Loading and chunking document from URL: {url}")
#     try:
#         content_stream = io.BytesIO(document_bytes)
#         from unstructured.partition.auto import partition
#         elements = partition(file=content_stream)

#         documents = []
#         for element in elements:
#             # We now store the raw text in the metadata for perfect retrieval
#             cleaned_meta = _clean_metadata(element.metadata.to_dict())
#             cleaned_meta['source'] = url
#             cleaned_meta['text'] = str(element) # Store original text here
#             documents.append(Document(page_content=str(element), metadata=cleaned_meta))

#         # --- MODIFIED: Text splitter now uses the word count function ---
#         # It splits text into chunks of 1000 words with a 200-word overlap.
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=0, # Reduced overlap to a more efficient value for word counts
#             length_function=count_words
#         )
        
#         chunked_documents = text_splitter.split_documents(documents)
#         print(f"Successfully chunked and cleaned document into {len(chunked_documents)} sections based on word count.")
#         return chunked_documents
#     except Exception as e:
#         print(f"An error occurred during file chunking: {e}")
#         raise e

def load_and_chunk_document(url: str, document_bytes: bytes) -> List[Document]:
    """Loads, merges, and chunks a document into ~500-word sections for Pinecone."""
    print(f"Loading and chunking document from URL: {url}")
    try:
        content_stream = io.BytesIO(document_bytes)
        from unstructured.partition.auto import partition
        elements = partition(file=content_stream)

        # Merge all elements into one large string
        full_text = "\n\n".join(str(element) for element in elements)

        # Create a single Document object
        merged_metadata = {"source": url}
        documents = [Document(page_content=full_text, metadata=merged_metadata)]

        # Text splitter: 500 words per chunk, 0 overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=0,
            length_function=count_words  # Uses word-based splitting
        )

        # Split into ~500-word chunks
        chunked_documents = text_splitter.split_documents(documents)

        print(f"Successfully chunked and cleaned document into {len(chunked_documents)} sections (~500 words each).")
        return chunked_documents

    except Exception as e:
        print(f"An error occurred during file chunking: {e}")
        raise e



async def get_or_create_vectors(pinecone_client: PineconeClient, pinecone_index_host: str, doc_url: str):
    """Checks if a document exists in a Pinecone namespace. If not, ingests it manually."""
    namespace = _create_namespace_from_url(doc_url)
    pinecone_index = pinecone_client.Index(host=pinecone_index_host)

    stats = pinecone_index.describe_index_stats()
    if namespace in stats.namespaces and stats.namespaces[namespace].vector_count > 0:
        print(f"DEBUG: Found {stats.namespaces[namespace].vector_count} vectors in existing namespace '{namespace}'. Skipping ingestion.")
        return
    
    print("Document not found in Pinecone. Starting ingestion process...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(doc_url)
            response.raise_for_status()
        
        chunked_docs = load_and_chunk_document(doc_url, response.content)
        # --- DEBUG: Check if chunks were created ---
        if not chunked_docs:
            print("DEBUG: ERROR - No document chunks were created. Ingestion cannot proceed.")
            return
        print(f"DEBUG: Created {len(chunked_docs)} chunks to be ingested.")

        texts_to_embed = [doc.page_content for doc in chunked_docs]
        metadata_to_upload = [{"source": doc.metadata.get("source", ""), "text": doc.page_content} for doc in chunked_docs]

        print("Embedding document chunks with OpenAI...")
        embeddings = EMBEDDING_MODEL.embed_documents(texts_to_embed)
        print(f"DEBUG: Successfully created {len(embeddings)} embeddings from OpenAI.")
        
        ids = [f"chunk_{i}" for i in range(len(chunked_docs))]
        vectors_to_upsert = list(zip(ids, embeddings, metadata_to_upload))

        print(f"DEBUG: Attempting to upsert {len(vectors_to_upsert)} vectors into namespace '{namespace}'.")
        for i in range(0, len(vectors_to_upsert), 100):
            batch = vectors_to_upsert[i:i+100]
            pinecone_index.upsert(vectors=batch, namespace=namespace)
        print("Ingestion complete.")
    except Exception as e:
        # --- DEBUG: Make failure obvious ---
        print("\n" + "="*50)
        print("DEBUG: CRITICAL ERROR DURING INGESTION!")
        print(f"Failed to ingest document: {e}")
        print("="*50 + "\n")
        raise e


# async def structure_queries_for_search(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Uses Gemini to convert user questions into keyword-focused search queries."""
#     print(f"Structuring {len(queries)} queries with Gemini...")
#     question_list = [q["question"] for q in queries]
#     prompt = (
#         "You are an expert at converting user questions into effective search queries. "
#         "For each question below, extract the core keywords and concepts. "
#         "Do not answer the question. Just provide a concise, keyword-rich search query. "
#         "Return a single JSON object with a key \"search_queries\" which contains a list of new search query strings, in order."
#         "\n\n--- QUESTIONS ---\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(question_list))
#     )
#     try:
#         payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
#         headers = {'Content-Type': 'application/json'}
#         async with httpx.AsyncClient() as client:
#             response = await client.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
#         response.raise_for_status()
#         result = response.json()
#         structured_data = json.loads(result['candidates'][0]['content']['parts'][0]['text'])
#         search_queries = structured_data.get("search_queries", [])

#         if len(search_queries) == len(queries):
#             for i, q_obj in enumerate(queries):
#                 q_obj["search_query"] = search_queries[i]
#             print("Successfully structured queries.")
#         else: # Fallback if list size mismatches
#             raise ValueError("Mismatched number of search queries returned.")
#         return queries
#     except Exception as e:
#         print(f"Gemini query structuring failed: {e}. Falling back to raw questions.")
#         for q_obj in queries:
#             q_obj["search_query"] = q_obj["question"]
#         return queries
    
    
    # SPACY VERSION (WITHOUT LLM) 
    
    
# def structure_queries_for_search(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Converts user questions into keyword-focused search queries using a local NLP model (spaCy).
#     This function does NOT use an LLM and is very fast.
#     """
#     if not NLP:
#         # Fallback if spaCy model isn't loaded
#         print("spaCy model not available. Falling back to using raw questions.")
#         for q_obj in queries:
#             q_obj["search_query"] = q_obj["question"]
#         return queries

#     print(f"Structuring {len(queries)} queries locally with spaCy...")
#     for q_obj in queries:
#         question = q_obj["question"]
#         doc = NLP(question)

#         # Extract meaningful tokens: Nouns, Proper Nouns, Adjectives, Verbs
#         # and also any recognized Named Entities (like dates, organizations)
#         keywords = []
#         for token in doc:
#             if not token.is_stop and not token.is_punct:
#                 if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"]:
#                     keywords.append(token.lemma_.lower()) # Use lemma for root form of word

#         # Extract named entities and add them
#         for ent in doc.ents:
#             if ent.text.lower() not in keywords:
#                  keywords.append(ent.text.lower())
        
#         # Create the structured query string
#         structured_query = " ".join(keywords)
        
#         # If no keywords were extracted, fall back to the original question
#         q_obj["search_query"] = structured_query if structured_query else question

#     print("Successfully structured queries locally.")
#     return queries


async def find_answers_with_pinecone(pinecone_client: PineconeClient, 
    pinecone_index_host: str, 
    doc_url: str, 
    queries: List[Dict[str, Any]]
) -> List[str]:
    """
    Finds answers by performing embedding, retrieval, and generation tasks in parallel.
    """
    # 1. Structure all queries in a single initial call (already efficient)
    # structured_queries = structure_queries_for_search(queries)
    # search_query_list = [q["search_query"] for q in structured_queries]

    for q in queries:
        q["search_query"] = q["question"]
    structured_queries = queries # Keep variable name for consistency downstream
    search_query_list = [q["search_query"] for q in structured_queries]
    
    namespace = _create_namespace_from_url(doc_url)
    pinecone_index = pinecone_client.Index(host=pinecone_index_host)

    # 2. Embed all search queries in a single, parallelized API call
    print(f"Embedding {len(search_query_list)} queries in a single batch...")
    query_embeddings = EMBEDDING_MODEL.embed_documents(search_query_list)
    print("All queries embedded successfully.")

    # 3. Query Pinecone for all vectors concurrently
    print("Querying Pinecone for all embeddings concurrently...")
    
    # The pinecone-client is synchronous, so we run it in a thread pool
    # to avoid blocking the asyncio event loop.
    loop = asyncio.get_running_loop()
    pinecone_query_tasks = []
    for embedding in query_embeddings:
        query_task = loop.run_in_executor(
            None,  # Use the default thread pool executor
            partial(
                pinecone_index.query,
                vector=embedding,
                top_k=4,
                include_metadata=True,
                namespace=namespace
            )
        )
        pinecone_query_tasks.append(query_task)
    
    # Wait for all Pinecone queries to complete
    query_results_list = await asyncio.gather(*pinecone_query_tasks)
    print("All Pinecone queries completed.")

    # 4. Prepare contexts and group into batches for Gemini
    contexts = []
    for query_results in query_results_list:
        matches = query_results.get('matches', [])
        retrieved_docs = [match['metadata']['text'] for match in matches if 'text' in match.get('metadata', {})]
        context = "\n\n---\n\n".join(retrieved_docs)
        contexts.append(context)

    # 5. Create and run Gemini generation tasks for all batches in parallel
    gemini_tasks = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(structured_queries), BATCH_SIZE):
            batch_queries = structured_queries[i:i + BATCH_SIZE]
            batch_contexts = contexts[i:i + BATCH_SIZE]

            batch_prompt_parts = [
                f"\nQuestion {idx+1}: {query['question']}\nContext {idx+1}:\n---\n{ctx}\n---" 
                for idx, (query, ctx) in enumerate(zip(batch_queries, batch_contexts))
            ]
            full_prompt = (
                "You are an expert analyst. Based ONLY on the provided context for each question, provide a clear and concise answer. "
                "If the answer is not in the context, explicitly state 'The answer cannot be found in the provided document context.' "
                "Your entire response must be a single JSON object with one key: \"answers\". The value of \"answers\" must be a JSON array of strings. "
                "Each string in the array must be the answer to the corresponding question in the same order."
                + "".join(batch_prompt_parts)
            )
            
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}
            }
            headers = {'Content-Type': 'application/json'}

            # Add the async task to the list
            task = client.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
            gemini_tasks.append(task)
        
        print(f"Dispatching {len(gemini_tasks)} batch-requests to Gemini concurrently...")
        # Await all Gemini calls to complete
        gemini_responses = await asyncio.gather(*gemini_tasks, return_exceptions=True)
        print("All Gemini responses received.")

    # 6. Process all results
    all_answers = []
    for i, response in enumerate(gemini_responses):
        if isinstance(response, Exception):
            print(f"An error occurred during Gemini answer generation for batch {i}: {response}")
            # Add error placeholders for each query in the failed batch
            batch_size = BATCH_SIZE if (i+1)*BATCH_SIZE <= len(queries) else len(queries) % BATCH_SIZE
            all_answers.extend(["Error during answer generation"] * batch_size)
            continue
        
        try:
            response.raise_for_status()
            result = response.json()
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            answers_obj = json.loads(json_text)
            batch_answers = answers_obj.get("answers", [])
            
            # Sanitize answers as in the original code
            sanitized_answers = [str(" ".join(map(str, answer))) if isinstance(answer, list) else str(answer) for answer in batch_answers]
            all_answers.extend(sanitized_answers)
        except Exception as e:
            print(f"Failed to parse response for batch {i}: {e}")
            batch_size = BATCH_SIZE if (i+1)*BATCH_SIZE <= len(queries) else len(queries) % BATCH_SIZE
            all_answers.extend(["Failed to parse Gemini response"] * batch_size)

    return all_answers