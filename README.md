# Advanced RAG Pipeline for Document Q&A

A high-performance, scalable Retrieval-Augmented Generation (RAG) pipeline designed to answer questions about complex documents provided via URL. Built with FastAPI, this system offers production-ready document processing with advanced semantic understanding and multilingual capabilities.

## 🚀 Key Features

- **Just-in-Time, Persistent Ingestion**: Documents are processed only once with deterministic namespacing
- **Advanced Hierarchical Chunking**: Semantic structure understanding using unstructured and PyMuPDF
- **Intelligent Query Structuring**: AI-powered query optimization using OpenAI LLM
- **High-Performance Async Architecture**: Fully asynchronous FastAPI with concurrent processing
- **Multilingual Capabilities**: Automatic language detection and translation
- **Production-Ready**: Bearer token authentication and comprehensive logging

## 🏗️ Architecture

```
API Request (URL, Questions)
      ↓
[FastAPI Endpoint: main.py]
  - Authenticate Request
  - Start Logging
      ↓
[get_or_create_vectors in rag_pipeline.py]
  - Create unique Namespace from URL
  - Check if Namespace exists in Pinecone
      ↓
      ├── (Yes) → Skip Ingestion
      ↓
      └── (No) → [Ingestion Pipeline]
                   - Download Document
                   - Hierarchical Chunking
                   - Embed Chunks (OpenAI)
                   - Upsert to Pinecone Namespace
      ↓
[find_answers_with_pinecone in rag_pipeline.py]
  - Structure queries using OpenAI
  - Embed search queries
  - Query Pinecone in parallel
  - Generate answers using Gemini
      ↓
[FastAPI Endpoint: main.py]
  - Log final answers
  - Return JSON Response
```

## 📋 Prerequisites

- Python 3.9+
- [Pinecone](https://www.pinecone.io/) account (free tier sufficient)
- [OpenAI API](https://openai.com/api/) key
- [Google AI (Gemini)](https://ai.google.dev/) API key

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Pinecone Credentials
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your-pinecone-index-name
PINECONE_API_HOST=your-pinecone-index-host

# OpenAI Credentials (for embedding)
OPENAI_API_KEY=sk-your_openai_key

# Google Gemini Credentials
GEMINI_API_KEY=your_gemini_api_key

# API Security Token
BEARER_TOKEN=your_super_secret_token
```

> **Important**: Your Pinecone index must be created with **vector dimension 1536** to match the OpenAI `text-embedding-3-small` model.

## 🚀 Usage

### Start the API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000/api/v1/hackrx/run`

### API Documentation

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### Example Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/hackrx/run' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your_super_secret_token' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": "https://arxiv.org/pdf/1706.03762",
    "questions": [
      "What is the title of this document?",
      "Explain the concept of self-attention in two sentences."
    ]
  }'
```

### Response Format

```json
{
  "answers": [
    "Attention Is All You Need"
    "Self-attention is a mechanism that allows..."
  ]
}
```

## 📁 Project Structure

```
├── main.py                 # FastAPI application and endpoints
├── rag_pipeline.py         # Core RAG logic and processing
├── .env                    # Environment variables (create this)
├── requirements.txt        # Python dependencies
├── rag_pipeline.log        # Application logs
└── README.md              # This file
```

## 📊 Logging

The system provides comprehensive logging:

- **`rag_pipeline.log`**: Main application flow, requests, and responses, contexts

## 🔮 Future Enhancements

- [ ] **Re-ranking Step**: Cross-encoder model for improved retrieval accuracy
- [ ] **Source Citations**: Automatic citation generation with document references
- [ ] **Conversational Memory**: Multi-turn conversation support with chat history
- [ ] **Async Ingestion**: Background processing with webhook notifications for large documents
- [ ] **Caching Layer**: Redis integration for faster response times
- [ ] **Document Format Support**: Extended support for various document formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🆘 Support

If you encounter any issues or have questions:

1. Check the logs (`rag_api.log` and `raw_text.log`)
2. Verify your environment variables are correctly set
3. Ensure your Pinecone index has the correct vector dimension (1536)
4. Open an issue in the repository

Built with ❤️ using FastAPI, Pinecone, OpenAI, and Google Gemini
