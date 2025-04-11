#!/usr/bin/env python3

"""
PDF Directory to Qdrant Vector Database Ingestion
------------------------------------------------
This script recursively loads all PDF files from a directory,
processes them using LlamaIndex, and stores the embedded
chunks in a Qdrant vector database.
Uses a vLLM instance via OpenAI-compatible API for embeddings.

DETAILED EXPLANATION:
---------------------
This script performs the following high-level operations:
1. Recursively searches a directory for PDF files
2. Loads each PDF and converts it to LlamaIndex Document objects
3. Splits the documents into smaller chunks (text segments)
4. Extracts potential titles from these chunks
5. Generates vector embeddings for each chunk using a vLLM model
6. Stores these embeddings in a Qdrant vector database
7. Creates a searchable index from these embeddings
8. Runs a test query against the indexed documents

The script uses an OpenAI-compatible API provided by a vLLM instance,
which allows using open-source models for embedding generation.
"""

import os
import logging
import argparse
from typing import List

# LlamaIndex core imports
# - Document: Represents a document in LlamaIndex
# - VectorStoreIndex: Creates a searchable index from document nodes
# - StorageContext: Manages storage for indices and their components
from llama_index.core import Document, VectorStoreIndex, StorageContext

# - SentenceSplitter: Splits documents into manageable chunks
from llama_index.core.node_parser import SentenceSplitter

# - TitleExtractor: Attempts to extract meaningful titles from chunks
from llama_index.core.extractors import TitleExtractor

# - IngestionPipeline: Manages the document processing workflow
from llama_index.core.ingestion import IngestionPipeline

# Qdrant client - connects to the Qdrant vector database service
import qdrant_client

# Vector store and embedding imports
# - QdrantVectorStore: Adapter for using Qdrant with LlamaIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

# - OpenAIEmbedding: Uses OpenAI-compatible API for generating embeddings
from llama_index.embeddings.openai import  OpenAIEmbedding
# from openai import OpenAI

# PDF reader - attempting to load the best available option
# The script tries to use faster/better PDF readers first and falls back
# to simpler ones if the preferred ones aren't available
try:
    from llama_index.readers.file import PyMuPDFReader as PDFReader
    pdf_reader_name = "PyMuPDFReader"  # Fastest and most feature-rich PDF reader
except ImportError:
    try:
        from llama_index.readers.file import PyPDFReader as PDFReader
        pdf_reader_name = "PyPDFReader"  # Decent alternative PDF reader
    except ImportError:
        from llama_index.readers.file import PDFReader
        pdf_reader_name = "PDFReader"  # Basic fallback PDF reader

# Configure logging - Sets up informative console output
# This helps track the progress of the script and diagnose issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_pdf_files(directory_path: str) -> List[str]:
    """
    Recursively find all PDF files in a directory.
    
    This function:
    1. Verifies that the provided directory exists
    2. Uses os.walk to recursively iterate through all subdirectories
    3. Checks each file to see if it has a .pdf extension (case-insensitive)
    4. Builds a list of absolute paths to all PDF files found
    
    Dependencies:
    - os module for filesystem operations
    - logging for reporting progress
    
    Args:
        directory_path: Path to the directory to search
        
    Returns:
        List of absolute paths to PDF files
        
    Raises:
        FileNotFoundError: If the specified directory doesn't exist
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    pdf_files = []
    
    # os.walk yields (dirpath, dirnames, filenames) for each directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Case-insensitive check for .pdf extension
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
    return pdf_files

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and convert it to LlamaIndex Documents.
    
    This function:
    1. Verifies that the PDF file exists
    2. Uses the selected PDF reader to extract text from the PDF
    3. Converts the extracted text into LlamaIndex Document objects
    
    Dependencies:
    - os module for file existence check
    - PDFReader (determined earlier in the script)
    - logging for reporting progress
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of LlamaIndex Documents
        
    Raises:
        FileNotFoundError: If the specified PDF file doesn't exist
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    logger.info(f"Loading PDF from {pdf_path} using {pdf_reader_name}")
    
    # Note: load() is a static method - doesn't require instantiation
    documents = PDFReader.load(file_path=pdf_path)
    
    logger.info(f"Loaded {len(documents)} document(s) from PDF")
    return documents

def setup_qdrant(collection_name: str, host: str = "localhost", port: int = 6333) -> QdrantVectorStore:
    """
    Set up a Qdrant vector store.
    
    This function:
    1. Connects to a Qdrant server using the specified host and port
    2. Creates (or uses an existing) collection with the specified name
    
    Dependencies:
    - qdrant_client for connecting to the Qdrant server
    - QdrantVectorStore from llama_index for integrating with LlamaIndex
    - logging for reporting progress
    
    Args:
        collection_name: Name of the collection to create or use
        host: Qdrant server host address
        port: Qdrant server port number
        
    Returns:
        QdrantVectorStore instance configured for the specified collection
        
    Note:
        The collection will be created automatically if it doesn't exist,
        using default settings appropriate for the embedding model.
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

def ingest_pdfs_to_qdrant(
    pdf_dir: str,
    collection_name: str,
    vllm_api_base: str,
    # vllm_embed_model: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> VectorStoreIndex:
    """
    Ingest all PDF files from a directory into a Qdrant vector store.
    
    This function:
    1. Finds all PDF files in the specified directory
    2. Sets up the Qdrant vector store
    3. Configures the embedding model using the vLLM API
    4. Creates an ingestion pipeline for processing documents
    5. Processes each PDF file, handling errors for individual files
    6. Creates a vector index from all processed nodes
    
    Dependencies:
    - find_pdf_files function to locate PDF files
    - load_pdf function to read PDFs
    - setup_qdrant function to prepare the vector store
    - OpenAIEmbedding for generating vector embeddings
    - SentenceSplitter, TitleExtractor for document processing
    - IngestionPipeline for orchestrating the processing workflow
    - VectorStoreIndex for creating the searchable index
    - StorageContext for managing storage components
    - logging for reporting progress
    
    Args:
        pdf_dir: Directory containing PDF files
        collection_name: Name of the Qdrant collection
        vllm_api_base: Base URL for the vLLM instance (with /v1 endpoint)
        vllm_embed_model: Name of the embedding model to use
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        chunk_size: Size of text chunks for indexing
        chunk_overlap: Overlap between chunks
        
    Returns:
        VectorStoreIndex instance or None if no PDFs were found
        
    Note:
        - chunk_size controls how large each text segment will be
        - chunk_overlap determines how much text is shared between chunks
        - Higher chunk_size means fewer chunks but may lose context
        - Higher chunk_overlap may preserve more context but creates more chunks
    """
    # Find all PDF files
    pdf_files = find_pdf_files(pdf_dir)
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return None
        
    # Setup Qdrant vector store
    vector_store = setup_qdrant(
        collection_name=collection_name,
        host=qdrant_host,
        port=qdrant_port
    )
    
    # Create embedding model - using OpenAI compatible API to vLLM
    # This connects to a local vLLM server that provides OpenAI-compatible API
    logger.info(f"Configuring embedding model with vLLM API at {vllm_api_base}")
    embed_model = OpenAIEmbedding(
        api_base=vllm_api_base,
        api_key="dummy-key",  # Can be dummy for local vLLM
        # model=vllm_embed_model
    )
    
    # models = embed_model.models.list()
    # model = models.data[0].id
    
    # Create storage context - manages where and how data is stored
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create ingestion pipeline - defines the document processing workflow
    logger.info(f"Setting up ingestion pipeline with chunk size {chunk_size} and overlap {chunk_overlap}")
    os.environ["OPENAI_API_KEY"] = "aaaaa"
    pipeline = IngestionPipeline(
        transformations=[
            # Split documents into smaller, manageable chunks
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            # Extract titles from content where possible
            TitleExtractor(),
            # Generate embeddings for each chunk
            # embed_model.embeddings.create(model=model),
            embed_model,
        ],
    )
    
    # Process each PDF file
    all_nodes = []
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing {pdf_path}")
            # Load the PDF
            documents = load_pdf(pdf_path)
            
            # Process documents through the pipeline
            nodes = pipeline.run(documents=documents)
            logger.info(f"Generated {len(nodes)} nodes from {pdf_path}")
            
            # Add these nodes to our collection
            all_nodes.extend(nodes)
        except Exception as e:
            # Log error but continue processing other PDFs
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            continue
    
    logger.info(f"Total nodes generated: {len(all_nodes)}")
    
    # Create index with all nodes - this makes the documents searchable
    logger.info("Creating vector index")
    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    return index

def main():
    """
    Main function to run the script.
    
    This function:
    1. Parses command-line arguments
    2. Calls ingest_pdfs_to_qdrant with the provided parameters
    3. Runs a test query if ingestion is successful
    4. Handles exceptions with detailed error messages
    
    Dependencies:
    - argparse for command-line argument parsing
    - ingest_pdfs_to_qdrant for the main functionality
    - logging for reporting progress and errors
    """
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description="Ingest PDF files from directory to Qdrant vector database")
    
    # Required arguments:
    parser.add_argument("--pdf-dir", required=True, 
                      help="Path to the directory containing PDF files")
    parser.add_argument("--collection", required=True, 
                      help="Name of the Qdrant collection")
    parser.add_argument("--vllm-api", required=True, 
                      help="Base URL for the vLLM API (with /v1 endpoint)")
    # parser.add_argument("--vllm-model", required=True, 
                    #   help="Name of the embedding model")
    
    # Optional arguments with defaults:
    parser.add_argument("--qdrant-host", default="localhost", 
                      help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, 
                      help="Qdrant server port")
    parser.add_argument("--chunk-size", type=int, default=512, 
                      help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=50, 
                      help="Overlap between chunks")
    
    args = parser.parse_args()
    
    try:
        # Ingest PDFs to Qdrant
        index = ingest_pdfs_to_qdrant(
            pdf_dir=args.pdf_dir,
            collection_name=args.collection,
            vllm_api_base=args.vllm_api,
            # vllm_embed_model=args.vllm_model,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        
        if not index:
            return
            
        logger.info(f"Successfully ingested PDF files from {args.pdf_dir} into Qdrant collection '{args.collection}'")
        
        # Example query - tests the functionality of the index
        logger.info("Running example query...")
        query_engine = index.as_query_engine()
        response = query_engine.query("What are these documents about?")
        print("\nExample query result:")
        print(response)
        
    except Exception as e:
        # Detailed error logging with traceback
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
        raise

# CHANGEABLE VARIABLES:
# --------------------
# When running this script, the following variables can be modified:
#
# 1. --pdf-dir: Directory containing PDF files
#    - Controls which PDFs are processed
#    - Can be any valid directory path containing PDFs
#
# 2. --collection: Name of the Qdrant collection
#    - Controls where embeddings are stored in Qdrant
#    - Choose a meaningful name related to your documents
#
# 3. --vllm-api: Base URL for the vLLM API
#    - Points to your vLLM instance with OpenAI API compatibility
#    - Example: "http://localhost:8000/v1"
#
# 4. --vllm-model: Name of the embedding model
#    - Specifies which model to use for creating embeddings
#    - Example: "text-embedding-ada-002" or another compatible model
#
# 5. --qdrant-host: Qdrant server host (default: "localhost")
#    - Change if Qdrant is running on a different machine
#
# 6. --qdrant-port: Qdrant server port (default: 6333)
#    - Change if Qdrant is running on a non-standard port
#
# 7. --chunk-size: Size of text chunks (default: 512)
#    - Larger chunks capture more context but may be less specific
#    - Smaller chunks are more granular but may lose broader context
#    - Value represents approximately the number of characters per chunk
#
# 8. --chunk-overlap: Overlap between chunks (default: 50)
#    - Higher overlap helps maintain context between chunks
#    - Too high may cause redundancy and increase storage requirements
#    - Value represents approximately the number of overlapping characters

if __name__ == "__main__":
    main()
