import argparse
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.readers import PDFReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import weaviate

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PDF to Weaviate RAG pipeline")
    parser.add_argument("--pdf-dir", type=str, default="pdf", 
                       help="Directory containing PDF files")
    parser.add_argument("--weaviate-url", type=str, required=True,
                       help="Weaviate cluster URL")
    parser.add_argument("--weaviate-api-key", type=str, required=True,
                       help="Weaviate API key")
    parser.add_argument("--index-name", type=str, default="rag_index",
                       help="Name for Weaviate index")
    args = parser.parse_args()

    # Load PDF documents
    reader = PDFReader()
    documents = reader.load_data(args.pdf_dir)

    # Configure Weaviate connection
    client = weaviate.connect_to_wcs(
        cluster_url=args.weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key=args.weaviate_api_key)
    )

    # Initialize vector store and index
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=args.index_name
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    # Save index for later use
    index.save_to_file("rag_index.pkl")

    # Example query function
    query_engine = index.as_query_engine()
    def query(question):
        response = query_engine.query(question)
        return response

if __name__ == "__main__":
    main()
