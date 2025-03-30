import os
from typing import List
from pathlib import Path
import weaviate
from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

def ingest_pdfs(pdf_dir: str, weaviate_url: str, weaviate_api_key: str, index_name: str) -> None:
    # Initialize Weaviate client
    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key)
    )

    # Initialize vector store
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=index_name
    )

    # Set up storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Initialize PDF reader
    loader = PyMuPDFReader()

    # Load PDF documents
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    documents = []
    for pdf_file in pdf_files:
        documents.extend(loader.load(file_path=str(pdf_file)))

    # Create and persist index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    print(f"Ingested {len(pdf_files)} PDF files into Weaviate index: {index_name}")

def retrieve_documents(query: str, weaviate_url: str, weaviate_api_key: str, index_name: str, top_k: int = 5) -> List[str]:
    # Initialize Weaviate client
    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key)
    )

    # Initialize vector store
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=index_name
    )

    # Set up storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the existing index
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Create query engine
    query_engine = index.as_query_engine()

    # Perform the query
    response = query_engine.query(query)

    # Extract and return the top k results
    return [node.text for node in response.source_nodes[:top_k]]

if __name__ == "__main__":
    # Configuration
    PDF_DIR = "pdf_dir"
    WEAVIATE_URL = "http://localhost:8080"
    WEAVIATE_API_KEY = "your-api-key"
    INDEX_NAME = "PDFDocuments"
    OPENAI_API_KEY = "your-openai-api-key"

    # Set up OpenAI embedding
    Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

    # Ingest PDFs
    ingest_pdfs(PDF_DIR, WEAVIATE_URL, WEAVIATE_API_KEY, INDEX_NAME)

    # Example retrieval
    query = "What is the main topic of the documents?"
    results = retrieve_documents(query, WEAVIATE_URL, WEAVIATE_API_KEY, INDEX_NAME)
    
    print("Query:", query)
    print("Top results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result[:100]}...")
