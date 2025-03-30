#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) system using LlamaIndex and Weaviate.
This script ingests PDF files into a vector store and provides retrieval capabilities.
"""

import os
import argparse
from typing import List, Optional
import weaviate
from pathlib import Path

# LlamaIndex imports - adjust based on your LlamaIndex version
from llama_index import VectorStoreIndex, Settings
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.readers.file import PDFReader


class RAGSystem:
    """RAG System for ingesting PDFs into Weaviate and retrieving them."""
    
    def __init__(
        self,
        pdf_dir: str,
        weaviate_url: str,
        weaviate_api_key: Optional[str] = None,
        class_name: str = "Document",
        embedding_model: str = "local:BAAI/bge-small-en-v1.5",
    ):
        """
        Initialize the RAG system.
        
        Args:
            pdf_dir: Directory containing PDF files to ingest
            weaviate_url: URL of the Weaviate instance
            weaviate_api_key: API key for Weaviate (optional)
            class_name: Class name in Weaviate
            embedding_model: Model to use for embeddings
        """
        self.pdf_dir = pdf_dir
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.class_name = class_name
        self.embedding_model = embedding_model
        self.vector_store = None
        self.index = None
        
        # Configure embedding model
        Settings.embed_model = embedding_model
        
    def connect_to_weaviate(self):
        """
        Establish connection to Weaviate vector store.
        
        Returns:
            weaviate.Client: Weaviate client instance
        """
        # Set up authentication if API key is provided
        auth_config = weaviate.auth.AuthApiKey(api_key=self.weaviate_api_key) if self.weaviate_api_key else None
        
        # Connect to Weaviate
        client = weaviate.Client(
            url=self.weaviate_url,
            auth_client_secret=auth_config
        )
        
        # Check if class exists, if not create it
        if not client.schema.exists(self.class_name):
            class_obj = {
                "class": self.class_name,
                "description": "Document class for RAG system",
                "vectorizer": "none",  # We'll use our own embeddings
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "The content of the document chunk"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "JSON metadata about the document"
                    }
                ]
            }
            client.schema.create_class(class_obj)
            print(f"Created Weaviate class: {self.class_name}")
        
        # Initialize vector store
        self.vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=self.class_name,
            text_key="content",
            metadata_key="metadata"
        )
        
        return client
    
    def load_pdf_documents(self) -> List:
        """
        Load all PDF documents from the specified directory.
        
        Returns:
            List: List of document objects
        """
        # Create PDF reader
        pdf_reader = PDFReader()
        
        # Get all PDF files in the directory
        pdf_files = list(Path(self.pdf_dir).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return []
        
        # Load each PDF file
        documents = []
        for pdf_file in pdf_files:
            print(f"Loading {pdf_file}...")
            docs = pdf_reader.load_data(file=pdf_file)
            documents.extend(docs)
            
        print(f"Loaded {len(documents)} document chunks from {len(pdf_files)} PDF files")
        return documents
    
    def ingest_documents(self, documents):
        """
        Ingest documents into vector store.
        
        Args:
            documents: List of document objects to ingest
        """
        if not documents:
            print("No documents to ingest")
            return
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Create and initialize index
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        
        print(f"Successfully ingested {len(documents)} document chunks into Weaviate")
    
    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List: List of retrieved nodes
        """
        if not self.index:
            raise ValueError("Index not initialized. Please ingest documents first.")
        
        # Create retriever
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        # Retrieve documents
        retrieved_nodes = retriever.retrieve(query)
        
        return retrieved_nodes


def main():
    """Main entry point of the script."""
    parser = argparse.ArgumentParser(
        description="RAG System for PDF documents using LlamaIndex and Weaviate"
    )
    
    # Required arguments
    parser.add_argument(
        "--pdf_dir", 
        type=str, 
        required=True, 
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--weaviate_url", 
        type=str, 
        required=True, 
        help="Weaviate URL (e.g., http://localhost:8080)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--weaviate_api_key", 
        type=str,
        help="Weaviate API key"
    )
    parser.add_argument(
        "--class_name", 
        type=str, 
        default="Document", 
        help="Class name in Weaviate"
    )
    parser.add_argument(
        "--embedding_model", 
        type=str, 
        default="local:BAAI/bge-small-en-v1.5", 
        help="Embedding model to use"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Query to search for (if provided, will perform retrieval)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=5, 
        help="Number of results to return for retrieval"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag_system = RAGSystem(
        pdf_dir=args.pdf_dir,
        weaviate_url=args.weaviate_url,
        weaviate_api_key=args.weaviate_api_key,
        class_name=args.class_name,
        embedding_model=args.embedding_model
    )
    
    # Connect to Weaviate
    rag_system.connect_to_weaviate()
    
    # Load documents
    documents = rag_system.load_pdf_documents()
    
    # Ingest documents
    rag_system.ingest_documents(documents)
    
    # Perform retrieval if query is provided
    if args.query:
        results = rag_system.retrieve(args.query, top_k=args.top_k)
        print(f"\nQuery: {args.query}")
        print(f"Retrieved {len(results)} results:\n")
        
        for i, node in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Score: {node.score}")
            print(f"Content: {node.node.text[:500]}...")
            print("-" * 50)


if __name__ == "__main__":
    main()