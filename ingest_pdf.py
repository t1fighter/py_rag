#!/usr/bin/env python3
# Created by AI: Claude Sonnet 3.7 Thinking
# Tuned / Debugged by t1fighter@github
"""
PDF to Qdrant Vector Database Ingestion
---------------------------------------
This script loads a PDF file from disk, processes it using LlamaIndex,
and stores the embedded chunks in a Qdrant vector database.
Uses a vLLM instance via OpenAI-compatible API for embeddings.
"""

import os
import logging
import argparse
from typing import List

# LlamaIndex core imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

# Qdrant client
import qdrant_client
# Vector store and embedding imports
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# PDF reader - attempting to load the best available option
try:
    from llama_index.readers.file import PyMuPDFReader as PDFReader
    pdf_reader_name = "PyMuPDFReader"
except ImportError:
    try:
        from llama_index.readers.file import PyPDFReader as PDFReader
        pdf_reader_name = "PyPDFReader"
    except ImportError:
        from llama_index.readers.file import PDFReader
        pdf_reader_name = "PDFReader"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and convert it to LlamaIndex Documents.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of LlamaIndex Documents
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    logger.info(f"Loading PDF from {pdf_path} using {pdf_reader_name}")
    documents = PDFReader.load(file_path=pdf_path)
    
    logger.info(f"Loaded {len(documents)} document(s) from PDF")
    return documents

def setup_qdrant(collection_name: str, host: str = "localhost", port: int = 6333) -> QdrantVectorStore:
    """
    Set up a Qdrant vector store.
    
    Args:
        collection_name: Name of the collection to create or use
        host: Qdrant server host
        port: Qdrant server port
        
    Returns:
        QdrantVectorStore instance
    """
    logger.info(f"Connecting to Qdrant at {host}:{port}")
    client = qdrant_client.QdrantClient(host=host, port=port)
    
    # Create vector store
    logger.info(f"Setting up vector store for collection: {collection_name}")
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=client,
    )
    
    return vector_store

def ingest_pdf_to_qdrant(
    pdf_path: str,
    collection_name: str,
    vllm_api_base: str,
    vllm_embed_model: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> VectorStoreIndex:
    """
    Ingest a PDF file into a Qdrant vector store.
    
    Args:
        pdf_path: Path to the PDF file
        collection_name: Name of the Qdrant collection
        vllm_api_base: Base URL for the vLLM instance (with /v1 endpoint)
        vllm_embed_model: Name of the Embedding model 
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        chunk_size: Size of text chunks for indexing
        chunk_overlap: Overlap between chunks
        
    Returns:
        VectorStoreIndex instance
    """
    # Load PDF
    documents = load_pdf(pdf_path)
    
    # Setup Qdrant vector store
    vector_store = setup_qdrant(
        collection_name=collection_name,
        host=qdrant_host,
        port=qdrant_port
    )
    
    # Create embedding model - using OpenAI compatible API to vLLM
    logger.info(f"Configuring embedding model with vLLM API at {vllm_api_base}")
    embed_model = OpenAIEmbedding(
        api_base=vllm_api_base,
        api_key="dummy-key",  # Can be dummy for local vLLM
        model=vllm_embed_model
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create ingestion pipeline
    logger.info(f"Setting up ingestion pipeline with chunk size {chunk_size} and overlap {chunk_overlap}")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            TitleExtractor(),  # Extract titles from content where possible
            embed_model,       # Generate embeddings
        ],
    )
    
    # Process documents
    logger.info("Processing documents through ingestion pipeline")
    nodes = pipeline.run(documents=documents)
    logger.info(f"Generated {len(nodes)} nodes from documents")
    
    # Create index
    logger.info("Creating vector index")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    return index

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description="Ingest PDF file to Qdrant vector database")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--collection", required=True, help="Name of the Qdrant collection")
    parser.add_argument("--vllm-api", required=True, help="Base URL for the vLLM API (with /v1 endpoint)")
    parser.add_argument("--vllm-model", required=True, help="Name of the Embedding model")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--chunk-size", type=int, default=512, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    try:
        # Ingest PDF to Qdrant
        index = ingest_pdf_to_qdrant(
            pdf_path=args.pdf,
            collection_name=args.collection,
            vllm_api_base=args.vllm_api,
            vllm_embed_model=args.vllm_model,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        
        logger.info(f"Successfully ingested PDF from {args.pdf} into Qdrant collection '{args.collection}'")
        
        # Example query
        logger.info("Running example query...")
        query_engine = index.as_query_engine()
        response = query_engine.query("What is this document about?")
        print("\nExample query result:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
