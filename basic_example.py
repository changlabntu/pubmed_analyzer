"""
Basic LlamaIndex Example
This script demonstrates how to:
1. Create documents from text
2. Build an index
3. Query the index
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

def setup_llama_index():
    """Configure LlamaIndex settings"""
    # Set up OpenAI LLM
    Settings.llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Set up OpenAI embeddings
    Settings.embed_model = OpenAIEmbedding(
        api_key=os.getenv("OPENAI_API_KEY")
    )

def create_sample_documents():
    """Create sample documents for indexing"""
    documents = [
        Document(text="LlamaIndex is a data framework for LLM applications. It provides tools to ingest, structure, and access private or domain-specific data."),
        Document(text="Vector databases store high-dimensional vectors and enable similarity search. They are essential for RAG (Retrieval Augmented Generation) applications."),
        Document(text="Embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings in vector space."),
        Document(text="RAG combines retrieval and generation. It first retrieves relevant documents, then uses them as context for generating responses."),
        Document(text="Python is a popular programming language for AI and machine learning applications due to its simplicity and rich ecosystem.")
    ]
    return documents

def main():
    """Main function to demonstrate LlamaIndex functionality"""
    print("🦙 LlamaIndex Basic Example")
    print("=" * 40)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        print("See .env.example for the format.")
        return
    
    try:
        # Setup LlamaIndex
        print("🔧 Setting up LlamaIndex...")
        setup_llama_index()
        
        # Create documents
        print("📄 Creating sample documents...")
        documents = create_sample_documents()
        print(f"Created {len(documents)} documents")
        
        # Build index
        print("🏗️  Building vector index...")
        index = VectorStoreIndex.from_documents(documents)
        print("✅ Index built successfully!")
        
        # Create query engine
        query_engine = index.as_query_engine()
        
        # Example queries
        queries = [
            "What is LlamaIndex?",
            "How do vector databases work?",
            "What is RAG?",
            "Tell me about embeddings"
        ]
        
        print("\n🔍 Running example queries:")
        print("-" * 40)
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: {query}")
            response = query_engine.query(query)
            print(f"   Answer: {response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have set up your OpenAI API key correctly.")

if __name__ == "__main__":
    main()
