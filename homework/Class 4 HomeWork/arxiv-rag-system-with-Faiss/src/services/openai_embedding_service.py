from typing import List
import numpy as np
import os
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class OpenAIEmbeddingService:
    """
    Embedding service using OpenAI's text-embedding-ada-002 model.
    Fast and high-quality embeddings via API.
    """
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        """
        Initialize OpenAI embedding service.
        
        Args:
            model: OpenAI embedding model to use
        """
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.model = model
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        print(f"Initialized OpenAI embedding service with model: {model}")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of documents using OpenAI API."""
        print(f"Generating OpenAI embeddings for {len(texts)} documents...")
        
        try:
            # OpenAI API can handle multiple texts in one request
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            # Extract embeddings from response
            embeddings = []
            for data in response.data:
                embeddings.append(data.embedding)
            
            embeddings_array = np.array(embeddings, dtype='float32')
            print(f"Generated embeddings with shape: {embeddings_array.shape}")
            
            return embeddings_array
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise e
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed_documents([query])[0]