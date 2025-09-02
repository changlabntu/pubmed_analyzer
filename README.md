# LlamaIndex Exploration Project

This project demonstrates the capabilities of LlamaIndex, a data framework for LLM applications that helps you ingest, structure, and access private or domain-specific data.

## Setup

### 1. Environment Setup
```bash
# Activate the conda environment
conda activate llamaindex

# Install dependencies (already done)
pip install -r requirements.txt
```

### 2. API Key Configuration
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

## Examples

### Basic Example (`basic_example.py`)
Demonstrates core LlamaIndex functionality:
- Creating documents from text
- Building a vector index
- Querying the index with natural language

```bash
python basic_example.py
```

### Document Loader Example (`document_loader_example.py`)
Shows how to:
- Load documents from files
- Create an interactive query interface
- Process multiple document types

```bash
python document_loader_example.py
```

## Key Concepts

- **Documents**: Text data that gets indexed
- **Vector Index**: Stores embeddings for semantic search
- **Query Engine**: Interface for asking questions about your data
- **RAG**: Retrieval-Augmented Generation combines search with LLM responses

## Next Steps

Try these advanced features:
- Custom document parsers
- Different vector stores (Chroma, Pinecone)
- Chat engines for conversational interfaces
- Document agents for complex reasoning
