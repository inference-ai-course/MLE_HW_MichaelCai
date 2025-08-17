from typing import List, Optional, Dict, Any
import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv")
    print("Environment variables will be loaded from system only.")

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
except ImportError:
    print("Warning: langchain packages not installed. Run: pip install langchain-openai langchain-community langchain-core chromadb")
    OpenAIEmbeddings = None
    Chroma = None
    Document = None


class EmbeddingService:
    """
    Service for generating embeddings from text chunks using OpenAI embeddings and Chroma vector store.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the embedding service with OpenAI embeddings and Chroma vector store.
        
        Args:
            persist_directory: Directory to persist the Chroma database. If None, will use PERSIST_DIRECTORY from .env
        """
        if OpenAIEmbeddings is None or Chroma is None:
            raise ImportError("Required packages not installed. Run: pip install langchain-openai langchain-community langchain-core chromadb python-dotenv")
        
        # Check if OpenAI API key is available from .env file or system environment
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Please:\n"
                "1. Add OPENAI_API_KEY=your-key to .env file, OR\n"
                "2. Set OPENAI_API_KEY environment variable"
            )
        
        # Set up persist directory (priority: parameter > .env file > default)
        if persist_directory:
            self.persist_directory = persist_directory
        else:
            self.persist_directory = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        
        print(f"Embedding service initialized with persist directory: {self.persist_directory}")
        print(f"OpenAI API key loaded: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
        
    def create_vectorstore(self, chunks: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> Chroma:
        """
        Create a Chroma vector store from text chunks.
        
        Args:
            chunks: List of text chunks to embed and store
            metadatas: Optional metadata for each chunk
            
        Returns:
            Chroma vector store instance
        """
        if not chunks:
            raise ValueError("No chunks provided to create vector store")
        
        # Filter out empty chunks
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        if not valid_chunks:
            raise ValueError("No valid chunks after filtering empty ones")
        
        # Create documents
        documents = [Document(page_content=chunk) for chunk in valid_chunks]
        
        # Add metadata if provided
        if metadatas:
            for i, doc in enumerate(documents):
                if i < len(metadatas):
                    doc.metadata = metadatas[i]
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        return self.vectorstore
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing vector store from disk.
        
        Returns:
            Chroma vector store instance or None if not found
        """
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return self.vectorstore
        except Exception as e:
            logging.warning(f"Could not load existing vector store: {e}")
            return None
    
    def add_chunks(self, chunks: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Add new chunks to existing vector store.
        
        Args:
            chunks: List of text chunks to add
            metadatas: Optional metadata for each chunk
        """
        if not self.vectorstore:
            raise ValueError("Vector store not created. Call create_vectorstore() first.")
        
        # Filter out empty chunks
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        if not valid_chunks:
            return
        
        # Create documents
        documents = [Document(page_content=chunk) for chunk in valid_chunks]
        
        # Add metadata if provided
        if metadatas:
            for i, doc in enumerate(documents):
                if i < len(metadatas):
                    doc.metadata = metadatas[i]
        
        # Add to vector store
        self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar chunks using a text query.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not created. Call create_vectorstore() or load_vectorstore() first.")
        
        if filter_dict:
            return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5, min_score: float = 0.0) -> List[tuple]:
        """
        Search for similar chunks with similarity scores and optional filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            min_score: Minimum similarity score threshold (0.0 to 1.0)
            
        Returns:
            List of (document, score) tuples filtered by minimum score
        """
        if not self.vectorstore:
            raise ValueError("Vector store not created. Call create_vectorstore() or load_vectorstore() first.")
        
        # Get all results with scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Filter by minimum score threshold
        filtered_results = [(doc, score) for doc, score in results if score >= min_score]
        
        return filtered_results
    
    def persist(self):
        """
        Persist the vector store to disk.
        Note: Modern Chroma versions auto-persist when persist_directory is specified.
        """
        # Modern Chroma versions automatically persist when persist_directory is set
        # No explicit persist() call needed
        print(f"Vector store auto-persisted to: {self.persist_directory}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of OpenAI embeddings.
        
        Returns:
            Embedding dimension (1536 for OpenAI)
        """
        return 1536  # OpenAI embeddings are 1536 dimensions


# Convenience functions for quick usage
def create_embedding_service(openai_api_key: Optional[str] = None, persist_directory: str = "./chroma_db") -> EmbeddingService:
    """
    Create an embedding service instance with OpenAI embeddings.
    
    Args:
        openai_api_key: OpenAI API key
        persist_directory: Directory to persist vector store
        
    Returns:
        EmbeddingService instance
    """
    return EmbeddingService(openai_api_key, persist_directory)


def create_vectorstore_from_chunks(chunks: List[str], openai_api_key: Optional[str] = None, 
                                 persist_directory: str = "./chroma_db") -> Chroma:
    """
    Quick function to create a vector store from text chunks.
    
    Args:
        chunks: List of text chunks
        openai_api_key: OpenAI API key
        persist_directory: Directory to persist vector store
        
    Returns:
        Chroma vector store
    """
    service = EmbeddingService(openai_api_key, persist_directory)
    return service.create_vectorstore(chunks)