import os
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
import weaviate

def ingest_pdfs(pdf_dir: str, weaviate_url: str, index_name: str) -> VectorStoreIndex:
    # Initialize Weaviate client
    client = weaviate.Client(weaviate_url)

    # Initialize vector store
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Configure LlamaIndex settings
    Settings.embed_model = OpenAIEmbedding()

    # Load PDF documents
    pdf_reader = PDFReader()
    documents = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, file)
            documents.extend(pdf_reader.load_data(file=file_path))

    # Create index from documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    return index

def retrieve_documents(index: VectorStoreIndex, query: str, top_k: int = 5) -> List[str]:
    # Create query engine
    query_engine = index.as_query_engine()

    # Perform query
    response = query_engine.query(query)

    # Extract and return top k source nodes
    source_nodes = response.source_nodes
    return [node.node.text for node in source_nodes[:top_k]]

if __name__ == "__main__":
    # Configuration
    PDF_DIR = "./pdf"
    WEAVIATE_URL = "http://localhost:8080"
    INDEX_NAME = "PDFDocuments"
    QUERY = "What is the main topic of the documents?"

    # Ingest PDFs
    index = ingest_pdfs(PDF_DIR, WEAVIATE_URL, INDEX_NAME)

    # Retrieve documents
    results = retrieve_documents(index, QUERY)

    # Print results
    print(f"Top {len(results)} relevant document snippets for query: '{QUERY}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result[:200]}...")