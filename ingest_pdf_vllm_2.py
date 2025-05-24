#!/usr/bin/env python3
"""
PDF Directory to Qdrant Vector Database Ingestion
------------------------------------------------
This script recursively loads all PDF files from a directory,
processes them using LlamaIndex, and stores the embedded
chunks in a Qdrant vector database.
Uses a vLLM instance via OpenAI-compatible API for embeddings.
"""

import os
import logging
import argparse
from typing import List, Optional, Any
import asyncio

# LlamaIndex core imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

# Qdrant client
import qdrant_client

# Vector store imports
from llama_index.vector_stores.qdrant import QdrantVectorStore

# OpenAI client for vLLM embeddings
from openai import OpenAI

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

from dataclasses import dataclass, field

@dataclass
class VLLMEmbedding(TransformComponent):
    """Custom embedding class using vLLM via OpenAI-compatible API."""
    
    api_base: str
    api_key: str = "EMPTY"
    model: Optional[str] = None
    # embed_batch_size: int = 10
    # Declare client as a field with default_factory to initialize it after other fields
    #client: Any = field(init=False)
    # embed_dim: Optional[int] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize the client after other fields are set."""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        
        # Get list of available models and use the first one if none specified
        if self.model is None:
            models = self.client.models.list()
            self.model = models.data[0].id
            
    # Rest of your methods remain the same
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        # Your implementation
        pass
    
    def get_text_embedding(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Your implementation
        pass


def find_pdf_files(directory_path: str) -> List[str]:
    """
    Recursively find all PDF files in a directory.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
        
    pdf_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
    return pdf_files


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and convert it to LlamaIndex Documents.
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
    vllm_embed_model: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> VectorStoreIndex:
    """
    Ingest all PDF files from a directory into a Qdrant vector store.
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
    
    # Create custom embedding model using vLLM via OpenAI API
    logger.info(f"Configuring embedding model with vLLM API at {vllm_api_base}")
    embed_model = VLLMEmbedding(
        api_base=vllm_api_base,
        api_key="dummy-key",  # Can be dummy for local vLLM
        model=vllm_embed_model
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create ingestion pipeline
    logger.info(f"Setting up ingestion pipeline with chunk size {chunk_size} and overlap {chunk_overlap}")
    os.environ["OPENAI_API_KEY"] = "aaaaa"
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            TitleExtractor(),  # Extract titles from content where possible
            embed_model,  # Generate embeddings
        ],
    )
    
    # Process each PDF file
    all_nodes = []
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing {pdf_path}")
            documents = load_pdf(pdf_path)
            
            # Process documents
            nodes = pipeline.run(documents=documents)
            logger.info(f"Generated {len(nodes)} nodes from {pdf_path}")
            all_nodes.extend(nodes)
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            continue
    
    logger.info(f"Total nodes generated: {len(all_nodes)}")
    
    # Create index with all nodes
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
    """
    parser = argparse.ArgumentParser(description="Ingest PDF files from directory to Qdrant vector database")
    parser.add_argument("--pdf-dir", required=True, help="Path to the directory containing PDF files")
    parser.add_argument("--collection", required=True, help="Name of the Qdrant collection")
    parser.add_argument("--vllm-api", required=True, help="Base URL for the vLLM API (with /v1 endpoint)")
    parser.add_argument("--vllm-model", required=True, help="Name of the Embedding model")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--chunk-size", type=int, default=512, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    try:
        # Ingest PDFs to Qdrant
        index = ingest_pdfs_to_qdrant(
            pdf_dir=args.pdf_dir,
            collection_name=args.collection,
            vllm_api_base=args.vllm_api,
            vllm_embed_model=args.vllm_model,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        
        if not index:
            return
            
        logger.info(f"Successfully ingested PDF files from {args.pdf_dir} into Qdrant collection '{args.collection}'")
        
        # Example query
        logger.info("Running example query...")
        query_engine = index.as_query_engine()
        response = query_engine.query("What are these documents about?")
        print("\nExample query result:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
