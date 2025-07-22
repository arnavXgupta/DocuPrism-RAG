import os
from dotenv import load_dotenv 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv() 

# --- 1. Load Documents ---
# We'll use a DirectoryLoader to load all PDF files from a specified folder.
print("Loading documents...")
loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

if not documents:
    print("No documents found. Please ensure your PDF files are in the 'data' directory.")
    exit()

print(f"Loaded {len(documents)} documents.")

# --- 2. Chunk Documents ---
# We split the loaded documents into smaller chunks to fit into the model's context window.
# RecursiveCharacterTextSplitter is a good general-purpose choice.
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")


# --- 3. Create Embeddings & Store in ChromaDB ---
# We use OpenAI's embedding model to convert text chunks into numerical vectors.
# These vectors are then stored in ChromaDB, a local vector store.
# This allows us to perform semantic searches later.

# IMPORTANT: This step requires your OPENAI_API_KEY to be set in your environment.
print("Generating embeddings and storing in ChromaDB...")

# The directory where the vector database will be saved.
persist_directory = 'db'

# Initialize the OpenAI embeddings model.
# Make sure your OPENAI_API_KEY is set as an environment variable.
embeddings = OpenAIEmbeddings()

# Create the Chroma vector store from the document chunks.
# This will process all chunks, generate embeddings, and save them to disk.
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Persist the database to disk.
vectordb.persist()
vectordb = None  # Clear the object from memory

print("\n--- Ingestion Complete ---")
print(f"The vector database is saved in the '{persist_directory}' directory.")