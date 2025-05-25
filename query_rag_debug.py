#!/usr/bin/env python3

"""
Command Line RAG Query Testing Script - DEBUG VERSION

Enhanced version with comprehensive debug output showing:
- All API calls and responses
- Vector search operations
- Document retrieval details
- LLM interactions
- Formatted output for human readability
"""

import os
import logging
import argparse
import json
import time
from typing import List, Dict, Any, Optional
import requests
from datetime import datetime

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore

# Qdrant client
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

# LLM imports
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI

# Custom embedding class with debug output
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import Field

# Enhanced logging configuration for debug mode
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_header(title: str, char: str = "=", width: int = 80) -> None:
    """Print a formatted header for visual separation"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_section(title: str, char: str = "-", width: int = 60) -> None:
    """Print a formatted section header"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")

def print_json(data: Any, title: str = "JSON Data") -> None:
    """Pretty print JSON data with formatting"""
    print(f"\n📄 {title}:")
    print("┌" + "─" * 78 + "┐")
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    for line in json_str.split('\n'):
        print(f"│ {line:<76} │")
    print("└" + "─" * 78 + "┘")

def print_api_call(method: str, url: str, payload: Dict = None) -> None:
    """Print formatted API call information"""
    print(f"\n🌐 API CALL: {method} {url}")
    if payload:
        print_json(payload, "Request Payload")

def print_api_response(response: requests.Response, show_full: bool = True) -> None:
    """Print formatted API response information"""
    print(f"\n📥 API RESPONSE:")
    print(f"   Status: {response.status_code}")
    print(f"   Headers: {dict(response.headers)}")
    
    if show_full:
        try:
            response_data = response.json()
            print_json(response_data, "Response Data")
        except:
            print(f"   Raw Response: {response.text[:500]}...")

class DebugVLLMEmbedding(BaseEmbedding):
    """Enhanced embedding class with comprehensive debug output."""
    
    api_base: str = Field(..., description="The API base URL for the vLLM server")
    model: str = Field(..., description="The model name used for embeddings")

    def _get_text_embedding(self, text: str) -> List[float]:
        print_section("EMBEDDING REQUEST", "🔤")
        
        payload = {"model": self.model, "input": text}
        url = f"{self.api_base}/embeddings"
        
        print(f"📝 Input Text (length: {len(text)} chars):")
        print(f"   \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        
        print_api_call("POST", url, payload)
        
        start_time = time.time()
        response = requests.post(url, json=payload)
        duration = time.time() - start_time
        
        print(f"⏱️  Request Duration: {duration:.3f}s")
        
        response.raise_for_status()
        result = response.json()
        
        embedding = result["data"][0]["embedding"]
        
        print(f"✅ Embedding Generated:")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Last 5 values: {embedding[-5:]}")
        print(f"   Vector norm: {sum(x*x for x in embedding)**0.5:.6f}")
        
        return embedding

    def _get_query_embedding(self, text: str) -> List[float]:
        print_header("QUERY EMBEDDING GENERATION", "🔍")
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

class DebugOpenAILike(OpenAILike):
    """Enhanced OpenAI client with debug output"""
    
    def complete(self, prompt, **kwargs):
        print_section("LLM COMPLETION REQUEST", "🤖")
        
        print(f"📝 Prompt (length: {len(prompt)} chars):")
        print("┌" + "─" * 78 + "┐")
        for line in prompt.split('\n'):
            print(f"│ {line[:76]:<76} │")
        print("└" + "─" * 78 + "┘")
        
        print(f"🔧 Parameters:")
        for key, value in kwargs.items():
            print(f"   {key}: {value}")
        
        start_time = time.time()
        result = super().complete(prompt, **kwargs)
        duration = time.time() - start_time
        
        print(f"⏱️  LLM Request Duration: {duration:.3f}s")
        print(f"📤 LLM Response:")
        print("┌" + "─" * 78 + "┐")
        response_text = str(result)
        for line in response_text.split('\n'):
            print(f"│ {line[:76]:<76} │")
        print("└" + "─" * 78 + "┘")
        
        return result   

class DebugLLMRerank(LLMRerank):
    """Enhanced LLMRerank with comprehensive debug output."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print_section("DEBUG RERANKER INITIALIZED", "🔄")
        print(f"📊 Configuration:")
        print(f"  Choice batch size: {self.choice_batch_size}")
        print(f"  Top N: {self.top_n}")
        print(f"  LLM model: {self.llm.model if hasattr(self.llm, 'model') else 'Unknown'}")
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[Any] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes with comprehensive debug output."""
        
        print_header("RERANKING PROCESS", "🔄")
        
        # Debug input information
        print_section("RERANK INPUT", "📥")
        print(f"🔍 Query: {query_bundle.query_str if query_bundle else 'No query'}")
        print(f"📄 Input nodes count: {len(nodes)}")
        print(f"⚙️ Target top_n: {self.top_n}")
        
        # Show input nodes details
        print_section("INPUT NODES DETAILS", "📚")
        for i, node in enumerate(nodes, 1):
            score = getattr(node, 'score', 'N/A')
            file_path = node.node.metadata.get('file_path', 'Unknown')
            text_preview = node.node.text[:100] + "..." if len(node.node.text) > 100 else node.node.text
            
            print(f"\n📄 Node {i}:")
            print(f"  📊 Initial Score: {score:.4f}" if isinstance(score, (int, float)) else f"  📊 Score: {score}")
            print(f"  📁 File: {file_path}")
            print(f"  📝 Text Preview: {text_preview}")
        
        # Start reranking process
        print_section("RERANKING EXECUTION", "⚡")
        start_time = time.time()
        
        try:
            # Call parent method with debug wrapping
            reranked_nodes = super()._postprocess_nodes(nodes, query_bundle)
            
            duration = time.time() - start_time
            print(f"⏱️ Reranking Duration: {duration:.3f}s")
            
            # Debug output results
            self._debug_rerank_results(nodes, reranked_nodes, query_bundle)
            
            return reranked_nodes
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Reranking Error after {duration:.3f}s: {e}")
            print_section("ERROR DETAILS", "🐛")
            import traceback
            traceback.print_exc()
            raise
    
    def _debug_rerank_results(
        self, 
        original_nodes: List[NodeWithScore], 
        reranked_nodes: List[NodeWithScore],
        query_bundle: Optional[Any] = None
    ):
        """Debug output for reranking results."""
        
        print_section("RERANKING RESULTS", "🎯")
        print(f"📥 Original nodes: {len(original_nodes)}")
        print(f"📤 Reranked nodes: {len(reranked_nodes)}")
        print(f"🔢 Requested top_n: {self.top_n}")
        
        # Show reranked results
        print_section("RERANKED NODE DETAILS", "🏆")
        for i, node in enumerate(reranked_nodes, 1):
            score = getattr(node, 'score', 'N/A')
            file_path = node.node.metadata.get('file_path', 'Unknown')
            text_preview = node.node.text[:100] + "..." if len(node.node.text) > 100 else node.node.text
            
            # Find original position
            original_pos = "Not found"
            for j, orig_node in enumerate(original_nodes):
                if orig_node.node.node_id == node.node.node_id:
                    original_pos = j + 1
                    break
            
            print(f"\n🥇 Rank {i} (was position {original_pos}):")
            print(f"  📊 Rerank Score: {score:.4f}" if isinstance(score, (int, float)) else f"  📊 Score: {score}")
            print(f"  📁 File: {file_path}")
            print(f"  📝 Text Preview: {text_preview}")
        
        # Show ranking changes
        if len(reranked_nodes) > 0:
            print_section("RANKING CHANGES ANALYSIS", "📈")
            changes = []
            for i, reranked_node in enumerate(reranked_nodes):
                for j, original_node in enumerate(original_nodes):
                    if original_node.node.node_id == reranked_node.node.node_id:
                        change = j - i  # Positive means moved up
                        changes.append((i+1, j+1, change, reranked_node.node.metadata.get('file_path', 'Unknown')))
                        break
            
            for new_pos, old_pos, change, file_path in changes:
                direction = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
                print(f"  {direction} Position {old_pos} → {new_pos} (change: {change:+d}) - {file_path}")

# Custom LLM wrapper for reranking debug
class DebugRerankerLLM(DebugOpenAILike):
    """Enhanced LLM specifically for reranking with detailed debug output."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print_section("RERANKER LLM INITIALIZED", "🤖")
        print(f"🔧 Reranker LLM Configuration:")
        print(f"  Model: {self.model}")
        print(f"  API Base: {self.api_base}")
        print(f"  Max Tokens: {self.max_tokens}")
        print(f"  Temperature: {self.temperature}")
    
    def complete(self, prompt, **kwargs):
        """Override complete method with reranker-specific debug output."""
        print_section("RERANKER LLM CALL", "🔄")
        print(f"📝 Reranking Prompt (length: {len(str(prompt))} chars):")
        print("┌" + "─" * 78 + "┐")
        prompt_str = str(prompt)
        for line in prompt_str.split('\n'):
            print(f"│ {line[:76]:<76} │")
        print("└" + "─" * 78 + "┘")
        
        print(f"🔧 Reranking Parameters:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
        
        start_time = time.time()
        result = super().complete(prompt, **kwargs)
        duration = time.time() - start_time
        
        print(f"⏱️ Reranking LLM Duration: {duration:.3f}s")
        print(f"📤 Reranking LLM Response (length: {len(str(result))} chars):")
        print("┌" + "─" * 78 + "┐")
        response_text = str(result)
        for line in response_text.split('\n'):
            print(f"│ {line[:76]:<76} │")
        print("└" + "─" * 78 + "┘")
        
        # Additional reranking-specific analysis
        if "1." in response_text or "2." in response_text:
            print("🎯 Detected numbered ranking format in response")
        
        return result


def debug_qdrant_collection_info(client: qdrant_client.QdrantClient, collection_name: str):
    """Print detailed information about the Qdrant collection"""
    print_section("QDRANT COLLECTION INFO", "🗄️")
    
    try:
        collection_info = client.get_collection(collection_name)
        print(f"📊 Collection: {collection_name}")
        print(f"   Status: {collection_info.status}")
        print(f"   Vectors count: {collection_info.vectors_count}")
        print(f"   Points count: {collection_info.points_count}")
        print(f"   Vector size: {collection_info.config.params.vectors.size}")
        print(f"   Distance metric: {collection_info.config.params.vectors.distance}")
        
    except Exception as e:
        print(f"❌ Error getting collection info: {e}")

def setup_query_engine(
    collection_name: str,
    vllm_embed_api_base: str,
    vllm_llm_api_base: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    similarity_top_k: int = 5
):
    """Setup query engine with comprehensive debug output."""
    
    print_header("RAG SYSTEM INITIALIZATION", "🚀")
    
    # Setup environment
    os.environ["OPENAI_API_KEY"] = "aaaaa"
    print("✅ Environment variables set")

    # System prompt
    system_prompt = """
<|im_start|>system_prompt
This is your system prompt.
- Keep your answers short and accurate.
- Only do what is asked of you, nothing else.
- Do not create additional explanations.
- State from which files of the delivered context the information is you used to answer the query. Use the file paths.
<|im_end|>
"""
    
    print_section("SYSTEM PROMPT", "💬")
    print(system_prompt)

    # Setup LLM
    print_section("LLM SETUP", "🤖")
    llm_client = OpenAI(
        base_url=vllm_llm_api_base,
        api_key="aaaaa",
    )
    
    print(f"🔗 LLM API Base: {vllm_llm_api_base}")
    
    try:
        models = llm_client.models.list()
        llm_model = models.data[0].id
        print(f"✅ Available LLM Model: {llm_model}")
    except Exception as e:
        print(f"❌ Error getting LLM models: {e}")
        raise

    llm = DebugOpenAILike(
        api_key="dummy_api_key",
        api_base=vllm_llm_api_base,
        model=llm_model,
        max_tokens=4000,
        context_window=20000,
        system_prompt=system_prompt,
        temperature=0.2,
    )
    print("✅ LLM client configured")

    # Setup embedding model
    print_section("EMBEDDING MODEL SETUP", "🔤")
    embed_client = OpenAI(
        base_url=vllm_embed_api_base,
        api_key="aaaaa",
    )
    
    print(f"🔗 Embedding API Base: {vllm_embed_api_base}")
    
    try:
        embed_models = embed_client.models.list()
        embed_model_name = embed_models.data[0].id
        print(f"✅ Available Embedding Model: {embed_model_name}")
    except Exception as e:
        print(f"❌ Error getting embedding models: {e}")
        raise

    embedding_model = DebugVLLMEmbedding(
        api_base=vllm_embed_api_base,
        model=embed_model_name
    )
    print("✅ Embedding model configured")

    # Initialize rerank model    
    # rerank_model = OpenAILike(
    #     api_key="dummy_api_key",
    #     api_base=vllm_llm_api_base,
    #     model=llm_model,
    #     max_tokens=6000,
    #     context_window=11000,
    #     system_prompt=system_prompt,
    #     temperature=0.1,
    # )
    
    # Replace this section in setup_query_engine function:
    # reranker = LLMRerank(
    #     choice_batch_size=5, # How many choices to rank at once
    #     top_n=3, # Number of top documents to return after reranking
    #     llm=llm # The LLM to use for reranking
    # )

    # With this enhanced debug version:
    reranker_llm = DebugRerankerLLM(
        api_key="dummy_api_key",
        api_base=vllm_llm_api_base,
        model=llm_model,
        max_tokens=1000,
        context_window=20000,
        system_prompt=system_prompt,
        temperature=0.1,
    )
    reranker = DebugLLMRerank(
        choice_batch_size=5,  # How many choices to rank at once
        top_n=3,  # Number of top documents to return after reranking
        llm=reranker_llm  # The debug-wrapped LLM for reranking
    )

    # Setup Qdrant connection
    print_section("QDRANT CONNECTION", "🗄️")
    print(f"🔗 Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
    
    client = qdrant_client.QdrantClient(host=qdrant_host, port=qdrant_port)
    
    if not client.collection_exists(collection_name=collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    print(f"✅ Connected to collection: {collection_name}")
    debug_qdrant_collection_info(client, collection_name)

    # Setup vector store
    print_section("VECTOR STORE SETUP", "📊")
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=client,
    )
    print("✅ Vector store configured")

    # Create index
    print_section("INDEX CREATION", "📚")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embedding_model
    )
    print("✅ Vector index created")

    # Create query engine
    print_section("QUERY ENGINE SETUP", "⚙️")
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,
        node_postprocessors=[reranker]
    )
    print(f"✅ Query engine created with top_k={similarity_top_k}")

    print_header("RAG SYSTEM READY", "✨")
    return query_engine

def debug_query_execution(query_engine, query: str, show_sources: bool = False):
    """Execute query with detailed debug output"""
    
    print_header("QUERY EXECUTION", "🔍")
    
    print(f"❓ Query: {query}")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    print_section("EXECUTING QUERY", "⚡")
    response = query_engine.query(query)
    
    total_duration = time.time() - start_time
    
    print_header("QUERY RESULTS", "📋")
    print(f"⏱️  Total Execution Time: {total_duration:.3f}s")
    
    print_section("FINAL RESPONSE", "💬")
    print("┌" + "─" * 78 + "┐")
    response_text = str(response)
    for line in response_text.split('\n'):
        print(f"│ {line[:76]:<76} │")
    print("└" + "─" * 78 + "┘")

    # Show sources if requested
    if show_sources and hasattr(response, 'source_nodes') and response.source_nodes:
        print_section(f"SOURCE DOCUMENTS ({len(response.source_nodes)} found)", "📚")
        
        for i, node in enumerate(response.source_nodes, 1):
            score = getattr(node, 'score', 'N/A')
            file_path = node.node.metadata.get('file_path', 'Unknown')
            
            print(f"\n📄 Source {i}:")
            print(f"   📊 Similarity Score: {score:.4f}" if score != 'N/A' else f"   📊 Score: {score}")
            print(f"   📁 File: {file_path}")
            
            if hasattr(node.node, 'text'):
                text_snippet = node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text
                print(f"   📝 Text Preview:")
                print("   ┌" + "─" * 74 + "┐")
                for line in text_snippet.split('\n'):
                    print(f"   │ {line[:72]:<72} │")
                print("   └" + "─" * 74 + "┘")
            
            if hasattr(node.node, 'metadata') and node.node.metadata:
                print(f"   🏷️  Metadata:")
                for key, value in node.node.metadata.items():
                    if key != 'file_path':  # Already shown above
                        print(f"      {key}: {value}")

    return response

def main():
    """Main function with enhanced argument parsing and error handling."""
    
    parser = argparse.ArgumentParser(
        description="Debug RAG Query Testing Script - Enhanced version with comprehensive debug output"
    )
    parser.add_argument("--collection", required=True, help="Name of the Qdrant collection to query")
    parser.add_argument("--vllm-embed-api", required=True, help="Base URL for the vLLM embedding API")
    parser.add_argument("--vllm-llm-api", required=True, help="Base URL for the vLLM LLM API")
    parser.add_argument("--query", required=True, help="The query to test")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--similarity-top-k", type=int, default=5, help="Number of similar documents to retrieve")
    parser.add_argument("--show-sources", action="store_true", help="Show detailed source documents")
    parser.add_argument("--no-debug", action="store_true", help="Disable detailed debug output")

    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.no_debug:
        logging.getLogger().setLevel(logging.WARNING)
    
    print_header("DEBUG RAG QUERY TESTING SCRIPT", "🚀")
    print(f"🔧 Configuration:")
    print(f"   Collection: {args.collection}")
    print(f"   Embedding API: {args.vllm_embed_api}")
    print(f"   LLM API: {args.vllm_llm_api}")
    print(f"   Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    print(f"   Top-K: {args.similarity_top_k}")
    print(f"   Show Sources: {args.show_sources}")

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

        # Execute query with debug output
        response = debug_query_execution(
            query_engine, 
            args.query, 
            show_sources=args.show_sources
        )
        
        print_header("EXECUTION COMPLETED SUCCESSFULLY", "✅")

    except Exception as e:
        print_header("ERROR OCCURRED", "❌")
        logger.error(f"Error during query execution: {str(e)}")
        import traceback
        print("\n🐛 Full Traceback:")
        print("┌" + "─" * 78 + "┐")
        for line in traceback.format_exc().split('\n'):
            print(f"│ {line[:76]:<76} │")
        print("└" + "─" * 78 + "┘")
        raise

if __name__ == "__main__":
    main()