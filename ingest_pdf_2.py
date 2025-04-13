#!/usr/bin/env python3
# Created by AI: Claude Sonnet 3.7 Thinking
# Tuned / Debugged by t1fighter@github
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
from typing import List

# LlamaIndex core imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline


# Qdrant client
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import VectorParams, Distance
# Vector store and embedding imports
from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI

# PDF reader - attempting to load the best available option
from llama_index.readers.file import PyMuPDFReader

from typing import List
from pydantic import BaseModel
from llama_index.core.schema import Node, TransformComponent

from typing import List
from pydantic import BaseModel
from llama_index.core.schema import TransformComponent, Node
from llama_index.core.bridge.pydantic import Field

class OpenAILikeTransform(TransformComponent, BaseModel):
    llm: any = Field(..., description="OpenAI-like LLM to use for transformations")
    
    def __call__(self, nodes: List[Node], **kwargs) -> List[Node]:
        # Validate that nodes is not None
        if nodes is None:
            print("Warning: Received None instead of a list of nodes")
            return []
            
        try:
            for node in nodes:
                if node.text is None:
                    print(f"Warning: Node {node.id} has None text, skipping")
                    continue
                    
                # Call the remote LLM API to process node text
                response = self.llm.embed(node.text)
                
                # Verify response structure
                if response is None:
                    print(f"Warning: LLM returned None for node {node.id}")
                    continue
                
                # Update node text
                if hasattr(response, 'text'):
                    node.text = response.text
                elif isinstance(response, str):
                    node.text = response
                else:
                    print(f"Warning: Unexpected response type: {type(response)}")
                    
            return nodes
            
        except Exception as e:
            print(f"Error in OpenAILikeTransform: {e}")
            # Return the original nodes unchanged rather than crashing
            return nodes
    
import requests
from typing import List
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import Field

# Minimal wrapper to call your vLLM embeddings endpoint.
class VLLMEmbedding(BaseEmbedding):
    # Declare fields as class attributes so Pydantic can track them.
    api_base: str = Field(..., description="The API base URL for the vLLM server")
    model: str = Field(..., description="The model name used for embeddings")

    def _get_text_embedding(self, text: str) -> List[float]:
        payload = {"model": self.model, "input": text}
        response = requests.post(f"{self.api_base}/embeddings", json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def _get_query_embedding(self, text: str) -> List[float]:
        # Use the same logic for query embedding.
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, text: str) -> List[float]:
        # Synchronously call the method; for real async, use an async HTTP client.
        return self._get_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_pdf_files(directory_path: str) -> List[str]:
    """
    Recursively find all PDF files in a directory.
    
    Args:
        directory_path: Path to the directory to search
        
    Returns:
        List of paths to PDF files
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
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of LlamaIndex Documents
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    logger.info(f"Loading PDF from {pdf_path}")
    PDFReader=PyMuPDFReader()
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
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(collection_name=collection_name,vectors_config=VectorParams(size=384, distance=Distance.COSINE))

    # Create vector store
    logger.info(f"Setting up vector store for collection: {collection_name}")
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=client,
    )
    
    return vector_store, client

def remove_already_ingested_pdf(qdrant_client: qdrant_client, collection_name: str, pdf_files: list) -> List[str]:
    pdf_files_clean=[]
    print("Cleaning up PDF files which are already ingested")
    for file in pdf_files:
        n=qdrant_client.count(collection_name,count_filter=Filter(must=[FieldCondition(key="file_path",match=MatchValue(value=file))]),exact=True)
        if n.count == 0:
            pdf_files_clean.append(file)
        else:
            print(f"{file} ist already in collection {collection_name}! Skipping...")
    
    return pdf_files_clean

def ingest_pdfs_to_qdrant(
    pdf_dir: str,
    collection_name: str,
    vllm_embed_api_base: str,
    vllm_embed_model: str,
    vllm_llm_api_base: str,
    vllm_llm_model: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> VectorStoreIndex:
    """
    Ingest all PDF files from a directory into a Qdrant vector store.
    
    Args:
        pdf_dir: Directory containing PDF files
        collection_name: Name of the Qdrant collection
        vllm_embed_api_base: Base URL for the vLLM instance (with /v1 endpoint)
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        chunk_size: Size of text chunks for indexing
        chunk_overlap: Overlap between chunks
        
    Returns:
        VectorStoreIndex instance
    """
    # Setup Qdrant vector store
    vector_store, client = setup_qdrant(
        collection_name=collection_name,
        host=qdrant_host,
        port=qdrant_port
    )
    
    # Find all PDF files
    pdf_files = find_pdf_files(pdf_dir)
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return None
    
    # Clean out Files that are already stored in the vectordb
    pdf_files_clean = remove_already_ingested_pdf(qdrant_client=client, collection_name=collection_name, pdf_files=pdf_files)
    
    # Create embedding model - using OpenAI compatible API to vLLM
    logger.info(f"Configuring embedding model with vLLM API at {vllm_embed_api_base}")
    os.environ["OPENAI_API_KEY"] = "aaaaa"
    os.environ["OPENAI_API_BASE"] = "http://vllm2:8000/v1"
    # embed_model = OpenAIEmbedding(
    #     api_base=vllm_embed_api_base,
    #     api_key="dummy-key"  # Can be dummy for local vLLM
    #     model=vllm_embed_model
    # )
    embedding_model = VLLMEmbedding(
        api_base=vllm_embed_api_base,
        model=vllm_embed_model  # adjust as needed
    )
    # openai_like_llm = OpenAILike(
    #     model=vllm_embed_model,  # or your custom model name
    #     api_base=vllm_embed_api_base,
    #     api_key="dummy"         # you can set this to any dummy string if not required
    # )
    
    # embed_vllm = OpenAI(
    # base_url=vllm_embed_api_base,
    # )
    # models = embed_vllm.models.list()
    # model = models.data[0].id
    
    # custom_transform = OpenAILikeTransform(llm=openai_like_llm)
    # custom_transform = OpenAILikeTransform(llm=embed_vllm, model=model)
    # external_llm = OpenAILike(
    #     api_key="dummy_api_key",  # Replace if your server requires a key
    #     endpoint=vllm_embed_api_base,
    #     model=vllm_embed_model,
    #     is_chat_model=False,
    # )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create ingestion pipeline
    logger.info(f"Setting up ingestion pipeline with chunk size {chunk_size} and overlap {chunk_overlap}")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            #TitleExtractor(),  # Extract titles from content where possible
            #custom_transform      # Generate embeddings
            VLLMEmbedding(api_base=vllm_embed_api_base,model=vllm_embed_model)
        ],
        vector_store=vector_store
    )
    
    # Process each PDF file
    all_nodes = []
    for pdf_path in pdf_files_clean:
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
    #print(all_nodes)
    
    # Create index with all nodes
    logger.info("Creating vector index")
    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        embed_model=embedding_model
    )
    
    #index = VectorStoreIndex.from_vector_store(vector_store)
    # VectorStoreIndex.from_vector_store(vector_store)
    #print(index)
    
    #return index
    return index

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description="Ingest PDF files from directory to Qdrant vector database")
    parser.add_argument("--pdf-dir", required=True, help="Path to the directory containing PDF files")
    parser.add_argument("--collection", required=True, help="Name of the Qdrant collection")
    parser.add_argument("--vllm-embed-api", required=True, help="Base URL for the vLLM API (with /v1 endpoint)")
    parser.add_argument("--vllm-embed-model", required=True, help="Embedding Model to use from vLLM Base")
    parser.add_argument("--vllm-llm-api", required=True, help="Base URL for the vLLM API (with /v1 endpoint)")
    parser.add_argument("--vllm-llm-model", required=True, help="Embedding Model to use from vLLM Base")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--chunk-size", type=int, default=180, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=25, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    try:
        # Ingest PDFs to Qdrant
        index = ingest_pdfs_to_qdrant(
            pdf_dir=args.pdf_dir,
            collection_name=args.collection,
            vllm_embed_api_base=args.vllm_embed_api,
            vllm_embed_model=args.vllm_embed_model,
            vllm_llm_api_base=args.vllm_llm_api,
            vllm_llm_model=args.vllm_llm_model,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        
        if not index:
            return

        logger.info(f"Successfully ingested PDF files from {args.pdf_dir} into Qdrant collection '{args.collection}'")
        
        # Example query
        # logger.info("Running example query...")
        # query_engine = index.as_query_engine()
        # response = query_engine.query("What are these documents about?")
        # print("\nExample query result:")
        # print(response)
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
