# # --- Dependencies ---
# # pip install langchain-community "unstructured[docx,eml,pdf]" httpx langchain-pinecone "pinecone-client[grpc]" langchain-openai requests

# import os
# import io
# import httpx
# import json
# import hashlib
# import asyncio
# from functools import partial
# from pinecone import Pinecone as PineconeClient
# from urllib.parse import urlparse
# from typing import List, Dict, Any

# # Langchain document object and text splitter
# from langchain_community.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Langchain embedding models and vector stores
# from langchain_openai import OpenAIEmbeddings
# # import spacy
# # print("Initializing spaCy model for local query structuring...")
# # try:
# #     NLP = spacy.load("en_core_web_sm")
# #     print("spaCy model initialized.")
# # except OSError:
# #     print("spaCy model not found. Please run 'python -m spacy download en_core_web_sm'")
# #     NLP = None

# # --- CONFIGURATION ---
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# BATCH_SIZE = 5
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"


# # --- INITIALIZE EMBEDDING MODEL ---
# print("Initializing OpenAI embedding model...")
# EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
# print("OpenAI embedding model initialized.")


# def _create_namespace_from_url(url: str) -> str:
#     """Creates a stable, clean namespace by normalizing and hashing a URL."""
#     parsed_url = urlparse(url.lower())
#     normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
#     return hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()


# def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
#     """Cleans metadata to ensure all values are Pinecone-compatible types."""
#     cleaned_metadata = {}
#     for key, value in metadata.items():
#         if isinstance(value, (str, int, float, bool)):
#             cleaned_metadata[key] = value
#         elif isinstance(value, list) and all(isinstance(i, str) for i in value):
#             cleaned_metadata[key] = value
#         else:
#             cleaned_metadata[key] = json.dumps(value, default=str)
#     return cleaned_metadata


# # --- ADDED: Helper function to count words instead of characters ---
# def count_words(text: str) -> int:
#     """Helper function to count words in a text string."""
#     return len(text.split())


# # def load_and_chunk_document(url: str, document_bytes: bytes) -> List[Document]:
# #     """Takes the raw bytes of a document, chunks it, and returns LangChain Document objects."""
# #     print(f"Loading and chunking document from URL: {url}")
# #     try:
# #         content_stream = io.BytesIO(document_bytes)
# #         from unstructured.partition.auto import partition
# #         elements = partition(file=content_stream)

# #         documents = []
# #         for element in elements:
# #             # We now store the raw text in the metadata for perfect retrieval
# #             cleaned_meta = _clean_metadata(element.metadata.to_dict())
# #             cleaned_meta['source'] = url
# #             cleaned_meta['text'] = str(element) # Store original text here
# #             documents.append(Document(page_content=str(element), metadata=cleaned_meta))

# #         # --- MODIFIED: Text splitter now uses the word count function ---
# #         # It splits text into chunks of 1000 words with a 200-word overlap.
# #         text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=1000,
# #             chunk_overlap=0, # Reduced overlap to a more efficient value for word counts
# #             length_function=count_words
# #         )
        
# #         chunked_documents = text_splitter.split_documents(documents)
# #         print(f"Successfully chunked and cleaned document into {len(chunked_documents)} sections based on word count.")
# #         return chunked_documents
# #     except Exception as e:
# #         print(f"An error occurred during file chunking: {e}")
# #         raise e

# def load_and_chunk_document(url: str, document_bytes: bytes) -> List[Document]:
#     """Loads, merges, and chunks a document into ~500-word sections for Pinecone."""
#     print(f"Loading and chunking document from URL: {url}")
#     try:
#         content_stream = io.BytesIO(document_bytes)
#         from unstructured.partition.auto import partition
#         elements = partition(file=content_stream)

#         # Merge all elements into one large string
#         full_text = "\n\n".join(str(element) for element in elements)

#         # Create a single Document object
#         merged_metadata = {"source": url}
#         documents = [Document(page_content=full_text, metadata=merged_metadata)]

#         # Text splitter: 500 words per chunk, 0 overlap
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=250,
#             chunk_overlap=0,
#             length_function=count_words  # Uses word-based splitting
#         )

#         # Split into ~500-word chunks
#         chunked_documents = text_splitter.split_documents(documents)

#         print(f"Successfully chunked and cleaned document into {len(chunked_documents)} sections (~500 words each).")
#         return chunked_documents

#     except Exception as e:
#         print(f"An error occurred during file chunking: {e}")
#         raise e



# async def get_or_create_vectors(pinecone_client: PineconeClient, pinecone_index_host: str, doc_url: str):
#     """Checks if a document exists in a Pinecone namespace. If not, ingests it manually."""
#     namespace = _create_namespace_from_url(doc_url)
#     pinecone_index = pinecone_client.Index(host=pinecone_index_host)

#     stats = pinecone_index.describe_index_stats()
#     if namespace in stats.namespaces and stats.namespaces[namespace].vector_count > 0:
#         print(f"DEBUG: Found {stats.namespaces[namespace].vector_count} vectors in existing namespace '{namespace}'. Skipping ingestion.")
#         return
    
#     print("Document not found in Pinecone. Starting ingestion process...")
#     try:
#         async with httpx.AsyncClient(timeout=120.0) as client:
#             response = await client.get(doc_url)
#             response.raise_for_status()
        
#         chunked_docs = load_and_chunk_document(doc_url, response.content)
#         # --- DEBUG: Check if chunks were created ---
#         if not chunked_docs:
#             print("DEBUG: ERROR - No document chunks were created. Ingestion cannot proceed.")
#             return
#         print(f"DEBUG: Created {len(chunked_docs)} chunks to be ingested.")

#         texts_to_embed = [doc.page_content for doc in chunked_docs]
#         metadata_to_upload = [{"source": doc.metadata.get("source", ""), "text": doc.page_content} for doc in chunked_docs]

#         print("Embedding document chunks with OpenAI...")
#         embeddings = EMBEDDING_MODEL.embed_documents(texts_to_embed)
#         print(f"DEBUG: Successfully created {len(embeddings)} embeddings from OpenAI.")
        
#         ids = [f"chunk_{i}" for i in range(len(chunked_docs))]
#         vectors_to_upsert = list(zip(ids, embeddings, metadata_to_upload))

#         print(f"DEBUG: Attempting to upsert {len(vectors_to_upsert)} vectors into namespace '{namespace}'.")
#         for i in range(0, len(vectors_to_upsert), 100):
#             batch = vectors_to_upsert[i:i+100]
#             pinecone_index.upsert(vectors=batch, namespace=namespace)
#         print("Ingestion complete.")
#     except Exception as e:
#         # --- DEBUG: Make failure obvious ---
#         print("\n" + "="*50)
#         print("DEBUG: CRITICAL ERROR DURING INGESTION!")
#         print(f"Failed to ingest document: {e}")
#         print("="*50 + "\n")
#         raise e


# # async def structure_queries_for_search(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
# #     """Uses Gemini to convert user questions into keyword-focused search queries."""
# #     print(f"Structuring {len(queries)} queries with Gemini...")
# #     question_list = [q["question"] for q in queries]
# #     prompt = (
# #         "You are an expert at converting user questions into effective search queries. "
# #         "For each question below, extract the core keywords and concepts. "
# #         "Do not answer the question. Just provide a concise, keyword-rich search query. "
# #         "Return a single JSON object with a key \"search_queries\" which contains a list of new search query strings, in order."
# #         "\n\n--- QUESTIONS ---\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(question_list))
# #     )
# #     try:
# #         payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
# #         headers = {'Content-Type': 'application/json'}
# #         async with httpx.AsyncClient() as client:
# #             response = await client.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
# #         response.raise_for_status()
# #         result = response.json()
# #         structured_data = json.loads(result['candidates'][0]['content']['parts'][0]['text'])
# #         search_queries = structured_data.get("search_queries", [])

# #         if len(search_queries) == len(queries):
# #             for i, q_obj in enumerate(queries):
# #                 q_obj["search_query"] = search_queries[i]
# #             print("Successfully structured queries.")
# #         else: # Fallback if list size mismatches
# #             raise ValueError("Mismatched number of search queries returned.")
# #         return queries
# #     except Exception as e:
# #         print(f"Gemini query structuring failed: {e}. Falling back to raw questions.")
# #         for q_obj in queries:
# #             q_obj["search_query"] = q_obj["question"]
# #         return queries
    
    
#     # SPACY VERSION (WITHOUT LLM) 
    
    
# # def structure_queries_for_search(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
# #     """
# #     Converts user questions into keyword-focused search queries using a local NLP model (spaCy).
# #     This function does NOT use an LLM and is very fast.
# #     """
# #     if not NLP:
# #         # Fallback if spaCy model isn't loaded
# #         print("spaCy model not available. Falling back to using raw questions.")
# #         for q_obj in queries:
# #             q_obj["search_query"] = q_obj["question"]
# #         return queries

# #     print(f"Structuring {len(queries)} queries locally with spaCy...")
# #     for q_obj in queries:
# #         question = q_obj["question"]
# #         doc = NLP(question)

# #         # Extract meaningful tokens: Nouns, Proper Nouns, Adjectives, Verbs
# #         # and also any recognized Named Entities (like dates, organizations)
# #         keywords = []
# #         for token in doc:
# #             if not token.is_stop and not token.is_punct:
# #                 if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"]:
# #                     keywords.append(token.lemma_.lower()) # Use lemma for root form of word

# #         # Extract named entities and add them
# #         for ent in doc.ents:
# #             if ent.text.lower() not in keywords:
# #                  keywords.append(ent.text.lower())
        
# #         # Create the structured query string
# #         structured_query = " ".join(keywords)
        
# #         # If no keywords were extracted, fall back to the original question
# #         q_obj["search_query"] = structured_query if structured_query else question

# #     print("Successfully structured queries locally.")
# #     return queries


# async def find_answers_with_pinecone(pinecone_client: PineconeClient, 
#     pinecone_index_host: str, 
#     doc_url: str, 
#     queries: List[Dict[str, Any]]
# ) -> List[str]:
#     """
#     Finds answers by performing embedding, retrieval, and generation tasks in parallel.
#     """
#     # 1. Structure all queries in a single initial call (already efficient)
#     # structured_queries = structure_queries_for_search(queries)
#     # search_query_list = [q["search_query"] for q in structured_queries]

#     for q in queries:
#         q["search_query"] = q["question"]
#     structured_queries = queries # Keep variable name for consistency downstream
#     search_query_list = [q["search_query"] for q in structured_queries]
    
#     namespace = _create_namespace_from_url(doc_url)
#     pinecone_index = pinecone_client.Index(host=pinecone_index_host)

#     # 2. Embed all search queries in a single, parallelized API call
#     print(f"Embedding {len(search_query_list)} queries in a single batch...")
#     query_embeddings = EMBEDDING_MODEL.embed_documents(search_query_list)
#     print("All queries embedded successfully.")

#     # 3. Query Pinecone for all vectors concurrently
#     print("Querying Pinecone for all embeddings concurrently...")
    
#     # The pinecone-client is synchronous, so we run it in a thread pool
#     # to avoid blocking the asyncio event loop.
#     loop = asyncio.get_running_loop()
#     pinecone_query_tasks = []
#     for embedding in query_embeddings:
#         query_task = loop.run_in_executor(
#             None,  # Use the default thread pool executor
#             partial(
#                 pinecone_index.query,
#                 vector=embedding,
#                 top_k=4,
#                 include_metadata=True,
#                 namespace=namespace
#             )
#         )
#         pinecone_query_tasks.append(query_task)
    
#     # Wait for all Pinecone queries to complete
#     query_results_list = await asyncio.gather(*pinecone_query_tasks)
#     print("All Pinecone queries completed.")

#     # 4. Prepare contexts and group into batches for Gemini
#     contexts = []
#     for query_results in query_results_list:
#         matches = query_results.get('matches', [])
#         retrieved_docs = [match['metadata']['text'] for match in matches if 'text' in match.get('metadata', {})]
#         context = "\n\n---\n\n".join(retrieved_docs)
#         contexts.append(context)

#     # 5. Create and run Gemini generation tasks for all batches in parallel
#     gemini_tasks = []
#     async with httpx.AsyncClient() as client:
#         for i in range(0, len(structured_queries), BATCH_SIZE):
#             batch_queries = structured_queries[i:i + BATCH_SIZE]
#             batch_contexts = contexts[i:i + BATCH_SIZE]

#             batch_prompt_parts = [
#                 f"\nQuestion {idx+1}: {query['question']}\nContext {idx+1}:\n---\n{ctx}\n---" 
#                 for idx, (query, ctx) in enumerate(zip(batch_queries, batch_contexts))
#             ]
#             full_prompt = (
#                 "You are an expert analyst. Based ONLY on the provided context for each question, provide a clear and concise answer. "
#                 "If the answer is not in the context, explicitly state 'The answer cannot be found in the provided document context.' "
#                 "Your entire response must be a single JSON object with one key: \"answers\". The value of \"answers\" must be a JSON array of strings. "
#                 "Each string in the array must be the answer to the corresponding question in the same order."
#                 + "".join(batch_prompt_parts)
#             )
            
#             payload = {
#                 "contents": [{"parts": [{"text": full_prompt}]}],
#                 "generationConfig": {"responseMimeType": "application/json"}
#             }
#             headers = {'Content-Type': 'application/json'}

#             # Add the async task to the list
#             task = client.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
#             gemini_tasks.append(task)
        
#         print(f"Dispatching {len(gemini_tasks)} batch-requests to Gemini concurrently...")
#         # Await all Gemini calls to complete
#         gemini_responses = await asyncio.gather(*gemini_tasks, return_exceptions=True)
#         print("All Gemini responses received.")

#     # 6. Process all results
#     all_answers = []
#     for i, response in enumerate(gemini_responses):
#         if isinstance(response, Exception):
#             print(f"An error occurred during Gemini answer generation for batch {i}: {response}")
#             # Add error placeholders for each query in the failed batch
#             batch_size = BATCH_SIZE if (i+1)*BATCH_SIZE <= len(queries) else len(queries) % BATCH_SIZE
#             all_answers.extend(["Error during answer generation"] * batch_size)
#             continue
        
#         try:
#             response.raise_for_status()
#             result = response.json()
#             json_text = result['candidates'][0]['content']['parts'][0]['text']
#             answers_obj = json.loads(json_text)
#             batch_answers = answers_obj.get("answers", [])
            
#             # Sanitize answers as in the original code
#             sanitized_answers = [str(" ".join(map(str, answer))) if isinstance(answer, list) else str(answer) for answer in batch_answers]
#             all_answers.extend(sanitized_answers)
#         except Exception as e:
#             print(f"Failed to parse response for batch {i}: {e}")
#             batch_size = BATCH_SIZE if (i+1)*BATCH_SIZE <= len(queries) else len(queries) % BATCH_SIZE
#             all_answers.extend(["Failed to parse Gemini response"] * batch_size)

#     return all_answers

# --- Dependencies ---
# pip install langchain-community "unstructured[docx,eml,pdf]" httpx langchain-pinecone "pinecone-client[grpc]" langchain-openai requests "PyMuPDF"

# import os
# import io
# import httpx
# import json
# import hashlib
# import asyncio
# import logging
# import sys
# import fitz # PyMuPDF
# from functools import partial
# from pinecone import Pinecone as PineconeClient
# from urllib.parse import urlparse
# from typing import List, Dict, Any, Generator

# # Langchain document object and text splitter
# from langchain_community.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Langchain embedding models and vector stores
# from langchain_openai import OpenAIEmbeddings

# # Unstructured elements for hierarchical chunking
# from unstructured.partition.auto import partition
# from unstructured.documents.elements import Element, Title, ListItem

# # --- LOGGING SETUP ---
# # Create a logger to provide detailed, step-by-step output
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)  # Set the lowest level to capture everything

# # Create a file handler to write all logs to a file
# # 'w' mode overwrites the log file on each run, use 'a' to append.
# try:
#     file_handler = logging.FileHandler('rag_pipeline.log', mode='w')
#     file_handler.setLevel(logging.DEBUG) # Log everything to the file
# except Exception as e:
#     print(f"Error setting up file handler for logging: {e}")
#     file_handler = None

# # Create a console handler to show less verbose logs in the terminal
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.INFO)  # Only show INFO and higher level messages on console

# # Create a formatter to define the log message structure
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# if file_handler:
#     file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)

# # Add the configured handlers to the logger
# if not logger.handlers:
#     if file_handler:
#         logger.addHandler(file_handler)
#     logger.addHandler(console_handler)


# # --- CONFIGURATION ---
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# BATCH_SIZE = 5
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"


# # --- INITIALIZE EMBEDDING MODEL ---
# logger.info("Initializing OpenAI embedding model...")
# EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
# logger.info("OpenAI embedding model initialized.")


# def _create_namespace_from_url(url: str) -> str:
#     """Creates a stable, clean namespace by normalizing and hashing a URL."""
#     logger.debug(f"Creating namespace for URL: {url}")
#     parsed_url = urlparse(url.lower())
#     normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
#     namespace = hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()
#     logger.debug(f"Generated namespace '{namespace}' for URL '{url}'")
#     return namespace


# def count_words(text: str) -> int:
#     """Helper function to count words in a text string."""
#     return len(text.split())


# def stream_pdf_pages(url: str, document_stream: io.BytesIO) -> Generator[Element, None, None]:
#     """
#     Tries to stream a PDF page-by-page using PyMuPDF. Falls back to Unstructured's partition for the whole document if it fails.
#     """
#     try:
#         logger.debug(f"Attempting to process PDF page-by-page with PyMuPDF for {url}")
#         doc = fitz.open(stream=document_stream, filetype="pdf")
#         for i, page in enumerate(doc):
#             new_doc = fitz.open()
#             new_doc.insert_pdf(doc, from_page=i, to_page=i)
#             page_stream = io.BytesIO(new_doc.tobytes())
#             new_doc.close()
#             yield from partition(file=page_stream, strategy="hi_res")
#         doc.close()
#         logger.debug(f"Successfully processed {len(doc)} pages with PyMuPDF.")
#     except Exception as e:
#         logger.warning(f"PyMuPDF failed to process '{url}' with error: {e}. Falling back to whole-document partitioning.")
#         document_stream.seek(0)
#         yield from partition(file=document_stream, strategy="hi_res")


# def load_and_chunk_document(url: str, document_stream: io.BytesIO) -> List[Document]:
#     """
#     Loads a document and applies an advanced hierarchical and dynamic chunking strategy.
#     """
#     logger.info(f"Starting ADVANCED hierarchical chunking for document from URL: {url}")
    
#     # --- DYNAMIC CHUNKING PARAMETERS ---
#     MIN_CHUNK_WORDS = 80
#     TARGET_CHUNK_WORDS_NARRATIVE = 250 
#     MAX_CHUNK_WORDS = 400
#     MAX_CHUNK_WORDS_LIST = 500

#     chunked_documents = []
#     current_chunk_text = ""
#     hierarchy_stack = ["Introduction"] 
#     current_content_type = 'narrative' 

#     fallback_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=MAX_CHUNK_WORDS,
#         chunk_overlap=int(MAX_CHUNK_WORDS * 0.1),
#         length_function=count_words
#     )

#     def create_and_save_chunk(text: str):
#         nonlocal chunked_documents
#         text = text.strip()
#         if not text:
#             return
        
#         metadata = {
#             "source": url,
#             "title": hierarchy_stack[-1].strip(),
#             "hierarchy": " > ".join(h.strip() for h in hierarchy_stack),
#             "text": text,
#             "word_count": count_words(text)
#         }
#         doc = Document(page_content=text, metadata=metadata)
#         chunked_documents.append(doc)
#         logger.debug(f"Created chunk with title '{metadata['title']}' and word count {metadata['word_count']}")

#     try:
#         for el in stream_pdf_pages(url, document_stream):
#             if not hasattr(el, 'text') or not el.text:
#                 continue
#             element_text = el.text.strip()
#             element_word_count = count_words(element_text)
#             element_type = type(el)

#             if element_type is Title:
#                 if count_words(current_chunk_text) >= MIN_CHUNK_WORDS:
#                     create_and_save_chunk(current_chunk_text)
#                     current_chunk_text = ""
                
#                 hierarchy_stack.append(element_text)
#                 current_content_type = 'narrative'
#                 logger.debug(f"New title detected. Hierarchy updated: {' > '.join(hierarchy_stack)}")
#                 continue
            
#             if element_word_count > MAX_CHUNK_WORDS:
#                 logger.debug(f"Element with {element_word_count} words exceeds max size, using fallback splitter.")
#                 if count_words(current_chunk_text) > 0:
#                     create_and_save_chunk(current_chunk_text)
#                     current_chunk_text = ""
                
#                 sub_chunks = fallback_splitter.split_text(element_text)
#                 for sub_chunk in sub_chunks:
#                     create_and_save_chunk(sub_chunk)
#                 continue

#             new_content_type = 'list' if element_type is ListItem else 'narrative'

#             split_before_adding = False
#             if current_content_type != new_content_type and count_words(current_chunk_text) > 0:
#                 split_before_adding = True
#             elif current_content_type == 'narrative' and count_words(current_chunk_text) >= TARGET_CHUNK_WORDS_NARRATIVE:
#                 split_before_adding = True
#             elif current_content_type == 'list' and count_words(current_chunk_text) >= MAX_CHUNK_WORDS_LIST:
#                  split_before_adding = True
            
#             if split_before_adding:
#                 create_and_save_chunk(current_chunk_text)
#                 current_chunk_text = ""

#             current_chunk_text += f"\n\n{element_text}"
#             current_content_type = new_content_type

#         if current_chunk_text.strip():
#             create_and_save_chunk(current_chunk_text)

#         logger.info(f"Successfully chunked document into {len(chunked_documents)} sections using advanced logic.")
#         return chunked_documents
#     except Exception as e:
#         logger.error(f"An error occurred during advanced file chunking for URL {url}: {e}", exc_info=True)
#         raise e


# async def get_or_create_vectors(pinecone_client: PineconeClient, pinecone_index_host: str, doc_url: str):
#     """Checks if a document exists in a Pinecone namespace. If not, ingests it using advanced chunking."""
#     logger.info(f"Ensuring vectors exist for document: {doc_url}")
#     namespace = _create_namespace_from_url(doc_url)
#     pinecone_index = pinecone_client.Index(host=pinecone_index_host)

#     stats = pinecone_index.describe_index_stats()
#     if namespace in stats.namespaces and stats.namespaces[namespace].vector_count > 0:
#         logger.info(f"Found {stats.namespaces[namespace].vector_count} vectors in existing namespace '{namespace}'. Skipping ingestion.")
#         return
    
#     logger.info(f"Document not found in Pinecone namespace '{namespace}'. Starting ingestion process...")
#     try:
#         async with httpx.AsyncClient(timeout=120.0) as client:
#             logger.debug(f"Fetching document content from URL: {doc_url}")
#             response = await client.get(doc_url)
#             response.raise_for_status()
#             logger.debug("Successfully fetched document content.")
        
#         document_stream = io.BytesIO(response.content)
#         chunked_docs = load_and_chunk_document(doc_url, document_stream)

#         if not chunked_docs:
#             logger.error("No document chunks were created. Ingestion cannot proceed.")
#             return
#         logger.debug(f"Created {len(chunked_docs)} chunks to be ingested.")

#         texts_to_embed = [doc.page_content for doc in chunked_docs]
#         # The new chunking logic produces rich metadata, which we will now pass entirely to Pinecone.
#         metadata_to_upload = [doc.metadata for doc in chunked_docs]

#         logger.info("Embedding document chunks with OpenAI...")
#         embeddings = EMBEDDING_MODEL.embed_documents(texts_to_embed)
#         logger.info(f"Successfully created {len(embeddings)} embeddings from OpenAI.")
        
#         ids = [f"chunk_{i}" for i in range(len(chunked_docs))]
#         vectors_to_upsert = list(zip(ids, embeddings, metadata_to_upload))

#         logger.info(f"Upserting {len(vectors_to_upsert)} vectors into namespace '{namespace}'.")
#         for i in range(0, len(vectors_to_upsert), 100):
#             batch = vectors_to_upsert[i:i+100]
#             logger.debug(f"Upserting batch {i//100 + 1} of size {len(batch)}.")
#             pinecone_index.upsert(vectors=batch, namespace=namespace)
#         logger.info("Ingestion complete.")
#     except Exception as e:
#         logger.critical(f"CRITICAL ERROR DURING INGESTION for doc_url '{doc_url}': {e}", exc_info=True)
#         raise e


# async def find_answers_with_pinecone(pinecone_client: PineconeClient, 
#     pinecone_index_host: str, 
#     doc_url: str, 
#     queries: List[Dict[str, Any]]
# ) -> List[str]:
#     """
#     Finds answers by performing embedding, retrieval, and generation tasks in parallel.
#     """
#     logger.info(f"Starting answer finding process for {len(queries)} queries on document: {doc_url}")
    
#     for q in queries:
#         q["search_query"] = q["question"]
#     structured_queries = queries
#     search_query_list = [q["search_query"] for q in structured_queries]
#     logger.debug(f"Input questions to be processed: {[q['question'] for q in queries]}")
    
#     namespace = _create_namespace_from_url(doc_url)
#     pinecone_index = pinecone_client.Index(host=pinecone_index_host)

#     logger.info(f"Embedding {len(search_query_list)} queries in a single batch...")
#     query_embeddings = EMBEDDING_MODEL.embed_documents(search_query_list)
#     logger.info("All queries embedded successfully.")

#     logger.info("Querying Pinecone for all embeddings concurrently...")
#     loop = asyncio.get_running_loop()
#     pinecone_query_tasks = []
#     for embedding in query_embeddings:
#         query_task = loop.run_in_executor(
#             None,
#             partial(
#                 pinecone_index.query,
#                 vector=embedding,
#                 top_k=4,
#                 include_metadata=True,
#                 namespace=namespace
#             )
#         )
#         pinecone_query_tasks.append(query_task)
    
#     query_results_list = await asyncio.gather(*pinecone_query_tasks)
#     logger.info("All Pinecone queries completed.")

#     contexts = []
#     logger.debug("--- Retrieved Contexts from Pinecone ---")
#     for i, query_results in enumerate(query_results_list):
#         matches = query_results.get('matches', [])
#         retrieved_docs = [match['metadata']['text'] for match in matches if 'text' in match.get('metadata', {})]
#         context = "\n\n---\n\n".join(retrieved_docs)
#         contexts.append(context)
#         logger.debug(f"Context for Query #{i+1} ('{queries[i]['question']}'):\n{context}\n")
#     logger.debug("--- End of Retrieved Contexts ---")

#     gemini_tasks = []
#     async with httpx.AsyncClient() as client:
#         for i in range(0, len(structured_queries), BATCH_SIZE):
#             batch_queries = structured_queries[i:i + BATCH_SIZE]
#             batch_contexts = contexts[i:i + BATCH_SIZE]

#             batch_prompt_parts = [
#                 f"\nQuestion {idx+1}: {query['question']}\nContext {idx+1}:\n---\n{ctx}\n---" 
#                 for idx, (query, ctx) in enumerate(zip(batch_queries, batch_contexts))
#             ]
#             full_prompt = (
#                 "You are an expert analyst. Based ONLY on the provided context for each question, provide a clear and concise answer. "
#                 "If the answer is not in the context, explicitly state 'The answer cannot be found in the provided document context.' "
#                 "Your entire response must be a single JSON object with one key: \"answers\". The value of \"answers\" must be a JSON array of strings. "
#                 "Each string in the array must be the answer to the corresponding question in the same order."
#                 + "".join(batch_prompt_parts)
#             )
#             logger.debug(f"Full prompt for Gemini batch starting at index {i}:\n{full_prompt}")
            
#             payload = {
#                 "contents": [{"parts": [{"text": full_prompt}]}],
#                 "generationConfig": {"responseMimeType": "application/json"}
#             }
#             headers = {'Content-Type': 'application/json'}

#             task = client.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
#             gemini_tasks.append(task)
        
#         logger.info(f"Dispatching {len(gemini_tasks)} batch-requests to Gemini concurrently...")
#         gemini_responses = await asyncio.gather(*gemini_tasks, return_exceptions=True)
#         logger.info("All Gemini responses received.")

#     all_answers = []
#     logger.debug("--- Processing Gemini Responses ---")
#     for i, response in enumerate(gemini_responses):
#         if isinstance(response, Exception):
#             logger.error(f"An error occurred during Gemini answer generation for batch {i}: {response}", exc_info=True)
#             batch_size = BATCH_SIZE if (i+1)*BATCH_SIZE <= len(queries) else len(queries) % BATCH_SIZE
#             all_answers.extend(["Error during answer generation"] * batch_size)
#             continue
        
#         try:
#             response.raise_for_status()
#             result = response.json()
#             json_text = result['candidates'][0]['content']['parts'][0]['text']
#             logger.debug(f"Raw Gemini JSON response for batch {i}: {json_text}")

#             answers_obj = json.loads(json_text)
#             batch_answers = answers_obj.get("answers", [])
            
#             sanitized_answers = [str(" ".join(map(str, answer))) if isinstance(answer, list) else str(answer) for answer in batch_answers]
#             logger.debug(f"Sanitized answers for batch {i}: {sanitized_answers}")
#             all_answers.extend(sanitized_answers)
#         except Exception as e:
#             logger.error(f"Failed to parse response for batch {i}: {e}", exc_info=True)
#             batch_size = BATCH_SIZE if (i+1)*BATCH_SIZE <= len(queries) else len(queries) % BATCH_SIZE
#             all_answers.extend(["Failed to parse Gemini response"] * batch_size)
#     logger.debug("--- End of Gemini Response Processing ---")
    
#     logger.info("--- Final Q&A Pairs ---")
#     for i, query in enumerate(queries):
#         if i < len(all_answers):
#             logger.info(f"Q: {query['question']}")
#             logger.info(f"A: {all_answers[i]}")
#         else:
#             logger.warning(f"Q: {query['question']} - A: [No answer generated]")
#     logger.info("-----------------------")
    
#     logger.info("Answer finding process complete.")
#     return all_answers


# --- Dependencies ---
# pip install langchain-community "unstructured[docx,eml,pdf]" httpx langchain-pinecone "pinecone-client[grpc]" langchain-openai requests "PyMuPDF"
# --- Dependencies ---
# pip install langchain-community "unstructured[pdf]" httpx langchain-pinecone "pinecone-client[grpc]" langchain-openai "PyMuPDF" langdetect tiktoken googletrans==4.0.0-rc1

import os
import io
import httpx
import json
import hashlib
import asyncio
import logging
import sys
import fitz  # PyMuPDF
import tiktoken
from functools import partial
from pinecone import Pinecone as PineconeClient
from urllib.parse import urlparse
from typing import List, Dict, Any, Generator
from concurrent.futures import ProcessPoolExecutor

# Imports for Multilingual RAG
from langdetect import detect, LangDetectException
from googletrans import Translator

# Langchain and Unstructured imports
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element, Title, ListItem

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
try:
    file_handler = logging.FileHandler('rag_pipeline.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
except Exception as e:
    print(f"Error setting up file handler for logging: {e}")
    file_handler = None
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
if file_handler:
    file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
if not logger.handlers:
    if file_handler:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
BATCH_SIZE = 5
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

# --- INITIALIZE EMBEDDING MODEL ---
logger.info("Initializing OpenAI embedding model...")
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
logger.info("OpenAI embedding model initialized.")


# ==============================================================================
# 1. SHARED UTILITIES
# ==============================================================================

def _create_namespace_from_url(url: str) -> str:
    """Creates a stable, clean namespace by normalizing and hashing a URL."""
    logger.debug(f"Creating namespace for URL: {url}")
    parsed_url = urlparse(url.lower())
    normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    namespace = hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()
    logger.debug(f"Generated namespace '{namespace}' for URL '{url}'")
    return namespace

def count_tokens(text: str) -> int:
    """Counts tokens using tiktoken for multilingual support, falling back to word count."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        return len(text.split())

# ==============================================================================
# 2. ADVANCED, MULTILINGUAL CHUNKING LOGIC
# This section is designed to be run in a separate process via ProcessPoolExecutor
# ==============================================================================

def stream_pdf_pages(url: str, document_stream: io.BytesIO) -> Generator[Element, None, None]:
    """Streams a PDF page-by-page to handle large documents efficiently."""
    try:
        doc = fitz.open(stream=document_stream, filetype="pdf")
        for i, page in enumerate(doc):
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=i, to_page=i)
            page_stream = io.BytesIO(new_doc.tobytes())
            new_doc.close()
            yield from partition(file=page_stream, strategy="fast", languages=[])
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF failed to process '{url}' page-by-page: {e}. Falling back to whole-document parsing.")
        document_stream.seek(0)
        yield from partition(file=document_stream, strategy="fast", languages=[])

def load_and_chunk_document(url: str, document_stream: io.BytesIO) -> List[Document]:
    """
    CPU-bound function for advanced, token-based, multilingual document chunking.
    """
    logger.info(f"[CPU-Bound Process] Starting MULTILINGUAL TOKEN-BASED chunking for: {url}")
    
    MIN_CHUNK_TOKENS = 100
    TARGET_CHUNK_TOKENS = 350
    MAX_CHUNK_TOKENS = 500

    chunked_documents = []
    current_chunk_text = ""
    hierarchy_stack = ["Introduction"]

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_TOKENS,
        chunk_overlap=int(MAX_CHUNK_TOKENS * 0.1),
        length_function=count_tokens
    )

    def create_and_save_chunk(text: str):
        nonlocal chunked_documents
        text = text.strip()
        if not text: return

        try:
            lang = detect(text)
        except LangDetectException:
            lang = 'unknown'

        metadata = {
            "source": url,
            "title": hierarchy_stack[-1].strip(),
            "hierarchy": " > ".join(h.strip() for h in hierarchy_stack),
            "text": text,
            "language": lang,
            "token_count": count_tokens(text)
        }
        chunked_documents.append(Document(page_content=text, metadata=metadata))

    for el in stream_pdf_pages(url, document_stream):
        if not hasattr(el, 'text') or not el.text: continue
        element_text = el.text.strip()
        element_token_count = count_tokens(element_text)

        if type(el) is Title:
            if count_tokens(current_chunk_text) >= MIN_CHUNK_TOKENS:
                create_and_save_chunk(current_chunk_text)
                current_chunk_text = ""
            hierarchy_stack.append(element_text)
            continue

        if element_token_count > MAX_CHUNK_TOKENS:
            if count_tokens(current_chunk_text) > 0:
                create_and_save_chunk(current_chunk_text)
                current_chunk_text = ""
            for sub_chunk in fallback_splitter.split_text(element_text):
                create_and_save_chunk(sub_chunk)
            continue
        
        if count_tokens(current_chunk_text) + element_token_count > TARGET_CHUNK_TOKENS:
            create_and_save_chunk(current_chunk_text)
            current_chunk_text = ""

        current_chunk_text += f"\n\n{element_text}"

    if current_chunk_text.strip():
        create_and_save_chunk(current_chunk_text)

    logger.info(f"[CPU-Bound Process] Finished chunking for {url}. Created {len(chunked_documents)} chunks.")
    return chunked_documents


# ==============================================================================
# 3. ON-DEMAND PARALLEL INGESTION ORCHESTRATOR
# ==============================================================================

async def get_or_create_vectors(pinecone_client: PineconeClient, pinecone_index_host: str, doc_url: str, executor: ProcessPoolExecutor):
    """
    Checks if a document exists. If not, ingests it based on its content type.
    """
    logger.info(f"Ensuring vectors exist for document: {doc_url}")
    namespace = _create_namespace_from_url(doc_url)
    pinecone_index = pinecone_client.Index(host=pinecone_index_host)

    try:
        stats = pinecone_index.describe_index_stats()
        if namespace in stats.namespaces and stats.namespaces[namespace].vector_count > 0:
            logger.info(f"Found {stats.namespaces[namespace].vector_count} vectors in existing namespace '{namespace}'. Skipping ingestion.")
            return
    except Exception as e:
        logger.error(f"Could not connect to Pinecone to check namespace stats: {e}")
        raise

    logger.info(f"Document not found in Pinecone. Starting parallel ingestion process for namespace '{namespace}'...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(doc_url)
            response.raise_for_status()
            document_bytes = response.content
            # --- FIX: Get content-type from headers ---
            content_type = response.headers.get("content-type", "").lower()
            logger.info(f"Downloaded document from {doc_url} with content-type: {content_type}")

        loop = asyncio.get_running_loop()
        
        # --- FIX: Decide chunking strategy based on content-type ---
        chunked_docs = []
        known_document_types = ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]

        if any(doc_type in content_type for doc_type in known_document_types):
            logger.info("Using advanced document chunking for complex file type.")
            document_stream = io.BytesIO(document_bytes)
            chunked_docs = await loop.run_in_executor(executor, load_and_chunk_document, doc_url, document_stream)
        else:
            logger.info("Using simple text chunking for plain text or unknown content type.")
            full_text = document_bytes.decode('utf-8', errors='replace')
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=350, # Using TARGET_CHUNK_TOKENS
                chunk_overlap=35, # 10% overlap
                length_function=count_tokens
            )
            texts = text_splitter.split_text(full_text)
            for text_chunk in texts:
                metadata = {"source": doc_url, "text": text_chunk, "language": "en", "token_count": count_tokens(text_chunk)}
                chunked_docs.append(Document(page_content=text_chunk, metadata=metadata))

        if not chunked_docs:
            logger.error("No document chunks were created. Ingestion cannot proceed.")
            return

        EMBEDDING_BATCH_SIZE = 100
        logger.info(f"Total chunks to process: {len(chunked_docs)}. Processing in batches of {EMBEDDING_BATCH_SIZE}.")

        for i in range(0, len(chunked_docs), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_docs = chunked_docs[i:i + EMBEDDING_BATCH_SIZE]
            
            texts_to_embed = [doc.page_content for doc in batch_docs]
            logger.info(f"  - Creating embeddings for batch {batch_num} ({len(texts_to_embed)} chunks)...")
            
            embeddings = await loop.run_in_executor(None, EMBEDDING_MODEL.embed_documents, texts_to_embed)

            ids = [f"chunk_{i + j}" for j in range(len(batch_docs))]
            metadata_to_upload = [doc.metadata for doc in batch_docs]
            vectors_to_upsert = list(zip(ids, embeddings, metadata_to_upload))

            logger.info(f"  - Uploading {len(vectors_to_upsert)} vectors for batch {batch_num}...")
            await loop.run_in_executor(None, partial(pinecone_index.upsert, vectors=vectors_to_upsert, namespace=namespace))

        logger.info(f"Successfully completed parallel ingestion for '{doc_url}'.")

    except Exception as e:
        logger.critical(f"CRITICAL ERROR during ingestion for '{doc_url}': {e}", exc_info=True)
        raise e

# ==============================================================================
# 4. RETRIEVAL AND GENERATION (RAG CORE) WITH TRANSLATOR
# ==============================================================================

def normalize_text(text: str) -> str:
    """Cleans text by replacing common problematic characters."""
    text = text.replace("�", "'") # Fix for Unicode replacement character
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text

async def structure_queries_for_search(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Uses Gemini to convert user questions into keyword-focused search queries."""
    logger.info(f"Structuring {len(queries)} queries with Gemini...")
    question_list = [q["question"] for q in queries]
    prompt = (
        "You are an expert at converting user questions into effective search queries. "
        "For each question below, extract the core keywords and concepts. "
        "Do not answer the question. Just provide a concise, keyword-rich search query. "
        "Return a single JSON object with a key \"search_queries\" which contains a list of new search query strings, in order."
        "\n\n--- QUESTIONS ---\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(question_list))
    )
    try:
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
        headers = {'Content-Type': 'application/json'}
        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        json_text = result['candidates'][0]['content']['parts'][0]['text'].strip().replace("```json", "").replace("```", "")
        structured_data = json.loads(json_text)
        search_queries = structured_data.get("search_queries", [])

        if len(search_queries) == len(queries):
            for i, q_obj in enumerate(queries):
                q_obj["search_query"] = search_queries[i]
            logger.info("Successfully structured queries.")
        else: # Fallback if list size mismatches
            raise ValueError("Mismatched number of search queries returned.")
        return queries
    except Exception as e:
        logger.error(f"Gemini query structuring failed: {e}. Falling back to raw questions.", exc_info=True)
        for q_obj in queries:
            q_obj["search_query"] = q_obj["question"]
        return queries

async def find_answers_with_pinecone(pinecone_client: PineconeClient, 
    pinecone_index_host: str, 
    doc_url: str, 
    queries: List[Dict[str, Any]]
) -> List[str]:
    """
    Finds answers by performing multilingual embedding, retrieval, translation, and generation.
    """
    logger.info(f"Starting answer finding process for {len(queries)} queries on document: {doc_url}")
    
    # Initialize the translator
    translator = Translator()
    
    # Normalize questions and detect their language
    for q in queries:
        q["question"] = normalize_text(q["question"])
        try:
            q["question_lang"] = detect(q["question"])
        except Exception:
            q["question_lang"] = "en" # Default to English on failure

    structured_queries = await structure_queries_for_search(queries)
    
    namespace = _create_namespace_from_url(doc_url)
    pinecone_index = pinecone_client.Index(host=pinecone_index_host)

    search_query_list = [q["search_query"] for q in structured_queries]
    logger.info(f"Embedding {len(search_query_list)} structured queries in a single batch...")
    query_embeddings = EMBEDDING_MODEL.embed_documents(search_query_list)
    logger.info("All queries embedded successfully.")

    logger.info("Querying Pinecone for all embeddings concurrently...")
    loop = asyncio.get_running_loop()
    pinecone_query_tasks = [
        loop.run_in_executor(None, partial(pinecone_index.query, vector=embedding, top_k=4, include_metadata=True, namespace=namespace))
        for embedding in query_embeddings
    ]
    query_results_list = await asyncio.gather(*pinecone_query_tasks)
    logger.info("All Pinecone queries completed.")

    # Prepare contexts and detect their languages
    contexts = []
    context_langs = []
    for i, query_results in enumerate(query_results_list):
        matches = query_results.get('matches', [])
        retrieved_docs = [match['metadata']['text'] for match in matches if 'text' in match.get('metadata', {})]
        context = "\n\n---\n\n".join(retrieved_docs)
        contexts.append(context)
        try:
            context_lang = detect(context) if context.strip() else "en"
        except Exception:
            context_lang = "en"
        context_langs.append(context_lang)
        logger.debug(f"Context for Query #{i+1} ('{queries[i]['question']}') is in language: {context_lang}")

    gemini_tasks = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(structured_queries), BATCH_SIZE):
            batch_queries = structured_queries[i:i + BATCH_SIZE]
            batch_contexts = contexts[i:i + BATCH_SIZE]
            batch_context_langs = context_langs[i:i + BATCH_SIZE]
            batch_question_langs = [q.get("question_lang", "en") for q in batch_queries]
            
            # Translate questions to match context language if they differ
            translated_questions = []
            for q, q_lang, ctx_lang in zip(batch_queries, batch_question_langs, batch_context_langs):
                question_text = q["question"]
                if q_lang != ctx_lang and ctx_lang != "en":
                    try:
                        translated = translator.translate(question_text, dest=ctx_lang).text
                        translated_questions.append(translated)
                        logger.debug(f"Translated question from '{q_lang}' to '{ctx_lang}': '{question_text}' -> '{translated}'")
                    except Exception as e:
                        logger.warning(f"Translation failed for question: {question_text}. Using original. Error: {e}")
                        translated_questions.append(question_text)
                else:
                    translated_questions.append(question_text)

            batch_prompt_parts = [
                f"\nQuestion {idx+1}: {translated_questions[idx]}\nContext {idx+1}:\n---\n{ctx}\n---"
                for idx, ctx in enumerate(batch_contexts)
            ]
            
            lang_note = (
                "The following questions and contexts may be in different languages. "
                "Answer each question in the language of its provided context. "
            )

            full_prompt = (
                f"{lang_note}"
                "You are an expert analyst. For each question below, use ONLY the provided context to answer. "
                "If the answer is directly present, quote or paraphrase it. "
                "If the answer can be reasonably inferred from the context, do so and explain briefly. "
                "Always provide your answers in English, regardless of the question or context language."
                "Only if there is truly no relevant information, reply: 'The answer cannot be found in the provided document context.' "
                "Your response must be a single JSON object with one key: \"answers\". The value of \"answers\" must be a JSON array of strings, in order to."
                + "".join(batch_prompt_parts)
            )
            
            payload = {"contents": [{"parts": [{"text": full_prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
            headers = {'Content-Type': 'application/json'}
            task = client.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
            gemini_tasks.append(task)
        
        logger.info(f"Dispatching {len(gemini_tasks)} batch-requests to Gemini concurrently...")
        gemini_responses = await asyncio.gather(*gemini_tasks, return_exceptions=True)
        logger.info("All Gemini responses received.")

    all_answers = []
    for i, response in enumerate(gemini_responses):
        if isinstance(response, Exception):
            logger.error(f"An error occurred during Gemini answer generation for batch {i}: {response}", exc_info=True)
            batch_size = min(BATCH_SIZE, len(queries) - i * BATCH_SIZE)
            all_answers.extend(["Error during answer generation"] * batch_size)
            continue
        try:
            response.raise_for_status()
            result = response.json()
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            answers_obj = json.loads(json_text)
            batch_answers = answers_obj.get("answers", [])
            sanitized_answers = [str(answer) for answer in batch_answers]
            all_answers.extend(sanitized_answers)
        except Exception as e:
            logger.error(f"Failed to parse response for batch {i}: {e}", exc_info=True)
            batch_size = min(BATCH_SIZE, len(queries) - i * BATCH_SIZE)
            all_answers.extend(["Failed to parse Gemini response"] * batch_size)
    
    logger.info("Answer finding process complete.")
    return all_answers
