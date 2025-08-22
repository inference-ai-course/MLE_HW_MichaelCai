from typing import List, Tuple, Dict, Any, Optional
import faiss
import numpy as np
import json
import os

class FaissService:
    """
    Service for managing Faiss index for vector similarity search.
    """
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """
        Initialize FaissService.
        
        Args:
            dimension: Dimension of the embeddings
            index_type: Type of Faiss index ("flat", "ivf", "hnsw")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.document_store = []
        
        # Create appropriate index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        elif index_type == "l2":
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        else:
            self.index = faiss.IndexFlatIP(dimension)  # Default to cosine similarity
            
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Add embeddings and corresponding documents to the index.
        
        Args:
            embeddings: Document embeddings
            documents: Original document texts
            metadata: Optional metadata for each document
        """
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to Faiss index
        self.index.add(embeddings)
        
        # Store documents and metadata for retrieval
        for i, doc in enumerate(documents):
            doc_data = {
                'text': doc,
                'metadata': metadata[i] if metadata and i < len(metadata) else {},
                'id': len(self.document_store)
            }
            self.document_store.append(doc_data)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples (document_text, score, metadata)
        """
        # Ensure query embedding is the right shape and type
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index (not -1)
                doc_data = self.document_store[idx]
                results.append((doc_data['text'], float(score), doc_data['metadata']))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            'total_documents': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.index.is_trained
        }
    
    def save_index(self, index_filepath: str, documents_filepath: Optional[str] = None):
        """
        Save the Faiss index and document store to disk.
        
        Args:
            index_filepath: Path to save the Faiss index
            documents_filepath: Path to save document store (optional)
        """
        # Save Faiss index
        faiss.write_index(self.index, index_filepath)
        
        # Save document store if filepath provided
        if documents_filepath:
            with open(documents_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.document_store, f, ensure_ascii=False, indent=2)
    
    def load_or_create_index(self, index_filepath: str, documents_filepath: Optional[str] = None):
        """
        Load a Faiss index and document store from disk. If files don't exist, create new ones.
        
        Args:
            index_filepath: Path to the Faiss index file
            documents_filepath: Path to the document store file (optional)
        
        Returns:
            bool: True if loaded from existing files, False if created new
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(index_filepath), exist_ok=True)
        if documents_filepath:
            os.makedirs(os.path.dirname(documents_filepath), exist_ok=True)
        
        # Load Faiss index if exists, otherwise create new
        if os.path.exists(index_filepath):
            self.index = faiss.read_index(index_filepath)
            print(f"Loaded existing index from {index_filepath}")
            loaded_index = True
        else:
            # Create new index (already done in __init__)
            print(f"Creating new index at {index_filepath}")
            loaded_index = False
        
        # Load document store if exists, otherwise create new
        if documents_filepath and os.path.exists(documents_filepath):
            with open(documents_filepath, 'r', encoding='utf-8') as f:
                self.document_store = json.load(f)
            print(f"Loaded existing documents from {documents_filepath}")
        else:
            # Initialize empty document store (already done in __init__)
            if documents_filepath:
                print(f"Creating new document store at {documents_filepath}")
        
        return loaded_index
    
    def load_index(self, index_filepath: str, documents_filepath: Optional[str] = None):
        """
        Load a Faiss index and document store from disk.
        
        Args:
            index_filepath: Path to the Faiss index file
            documents_filepath: Path to the document store file (optional)
        """
        # Load Faiss index
        if os.path.exists(index_filepath):
            self.index = faiss.read_index(index_filepath)
        else:
            raise FileNotFoundError(f"Index file not found: {index_filepath}")
        
        # Load document store if filepath provided
        if documents_filepath and os.path.exists(documents_filepath):
            with open(documents_filepath, 'r', encoding='utf-8') as f:
                self.document_store = json.load(f)
        
    def clear(self):
        """Clear the index and document store."""
        self.index.reset()
        self.document_store = []