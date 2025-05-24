#!/usr/bin/env python3

"""
Command Line RAG Query Testing Script

Simple script to test queries against your Qdrant vector database
from the command line using the same vLLM servers.
"""

import os
import logging
import argparse
from typing import List
import requests

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

# Qdrant client
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

# LLM imports
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI

# Custom embedding class (same as ingest script)
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import Field

class VLLMEmbedding(BaseEmbedding):
    """Custom embedding class for vLLM server."""
    
    api_base: str = Field(..., description="The API base URL for the vLLM server")
    model: str = Field(..., description="The model name used for embeddings")

    def _get_text_embedding(self, text: str) -> List[float]:
        payload = {"model": self.model, "input": text}
        response = requests.post(f"{self.api_base}/embeddings", json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def _get_query_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_query_engine(
    collection_name: str,
    vllm_embed_api_base: str,
    vllm_llm_api_base: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    similarity_top_k: int = 5
):
    """Setup query engine using the same configuration as ingest script."""
    
    # Setup environment (same as ingest script)
    os.environ["OPENAI_API_KEY"] = "aaaaa"
    
    # System prompt (same as ingest script)
    system_prompt = """
<|im_start|>system_prompt
This is your system prompt.
- Keep your answers short and accurate.
- Only do what is asked of you, nothing else.
- Do not create additional explanations.
- State from which files of the delivered context the information is you used to answer the query. Use the file paths.
<|im_end|>
"""
    
    # Setup LLM (same as ingest script)
    llm_client = OpenAI(
        base_url=vllm_llm_api_base,
        api_key="aaaaa",
    )
    
    llm_model = llm_client.models.list().data[0].id
    
    llm = OpenAILike(
        api_key="dummy_api_key",
        api_base=vllm_llm_api_base,
        model=llm_model,
        max_tokens=6000,
        context_window=11000,
        system_prompt=system_prompt,
        temperature=0.1,
    )
    
    # Setup embedding model (same as ingest script)
    embed_client = OpenAI(
        base_url=vllm_embed_api_base,
        api_key="aaaaa",
    )
    
    embed_model_name = embed_client.models.list().data[0].id
    
    embedding_model = VLLMEmbedding(
        api_base=vllm_embed_api_base,
        model=embed_model_name
    )
    
    # Setup Qdrant connection
    client = qdrant_client.QdrantClient(host=qdrant_host, port=qdrant_port)
    
    if not client.collection_exists(collection_name=collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    # Setup vector store
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=client,
    )
    
    # Create index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embedding_model
    )
    
    # Create query engine
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k
    )
    
    return query_engine

def main():
    """Main function to run a single query from command line."""
    parser = argparse.ArgumentParser(description="Test a single RAG query against Qdrant vector database")
    parser.add_argument("--collection", required=True, help="Name of the Qdrant collection to query")
    parser.add_argument("--vllm-embed-api", required=True, help="Base URL for the vLLM embedding API")
    parser.add_argument("--vllm-llm-api", required=True, help="Base URL for the vLLM LLM API")
    parser.add_argument("--query", required=True, help="The query to test")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--similarity-top-k", type=int, default=5, help="Number of similar documents to retrieve")
    parser.add_argument("--show-sources", action="store_true", help="Show source documents")
    
    args = parser.parse_args()
    
    try:
        # Setup query engine
        query_engine = setup_query_engine(
            collection_name=args.collection,
            vllm_embed_api_base=args.vllm_embed_api,
            vllm_llm_api_base=args.vllm_llm_api,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            similarity_top_k=args.similarity_top_k
        )
        
        # Run the query
        print(f"Query: {args.query}")
        print("-" * 60)
        
        response = query_engine.query(args.query)
        print(f"Response: {response}")
        
        # Show sources if requested
        if args.show_sources and hasattr(response, 'source_nodes') and response.source_nodes:
            print(f"\nSources ({len(response.source_nodes)} documents):")
            for i, node in enumerate(response.source_nodes, 1):
                score = getattr(node, 'score', 'N/A')
                file_path = node.node.metadata.get('file_path', 'Unknown')
                print(f"  {i}. Score: {score:.3f} | File: {file_path}")
                if args.show_sources:
                    # Show a snippet of the source text
                    text_snippet = node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text
                    print(f"     Text: {text_snippet}")
                    print()
        
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise

if __name__ == "__main__":
    main()
