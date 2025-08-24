import os
import numpy as np
from src.services.text_extraction_service import extract_text_from_folder, extract_text_from_pdf
from src.services.openai_embedding_service import OpenAIEmbeddingService
from src.services.chunking_service import chunk_files, chunk_text
from src.services.faiss_service import FaissService
from src.services.database_service import DatabaseService
from src.services.hybrid_search_service import HybridSearchService
from src.utils.filter_duplicate import filter_duplicate_chunks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException
from typing import Dict, List
import logging

app = FastAPI()

#step 1, load env file 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError: 
    logging.warning("Warning: python-dotenv not installed. Run: pip install python-dotenv")
    logging.warning("Environment variables will be loaded from system only.")



#use Faiss to indexing and search  
def load_faiss_service()-> tuple[FaissService, OpenAIEmbeddingService, DatabaseService]:
    try:
        faiss_service = FaissService(dimension=1536)  # OpenAI embeddings are 1536-dimensional
        embedding_service = OpenAIEmbeddingService()
        database_service = DatabaseService()
        
        index_path = "indexes/faiss.index"
        documents_path = "indexes/documents.json"
        
        # Get existing chunks for similarity comparison
        existing_chunks = []
        
        # Try to load existing index and documents
        try:
            faiss_service.load_index(index_path, documents_path)
            print(f"Loaded existing index with {faiss_service.get_stats()['total_documents']} documents")
            # Get existing chunks for comparison
            if hasattr(faiss_service, 'document_store') and faiss_service.document_store:
                existing_chunks = [doc['text'] for doc in faiss_service.document_store]
        except (FileNotFoundError, Exception) as load_error:
            print(f"No existing index found: {load_error}. Creating new index...")

        #load pdf files
        folder_path = os.getenv("PDF_FOLDER_PATH")

        if not folder_path:
            raise ValueError("PDF_FOLDER_PATH environment variable not set")
        
        print(f"Extracting PDF files from folder {folder_path}")
        import sys
        sys.stdout.flush()
        
        pdf_texts = extract_text_from_folder(folder_path, max_files=3)
        if not pdf_texts:
            logging.warning(f"No PDF found in directory {folder_path}")
            raise Exception("Please make sure PDFs are added in the folder")
        
        print(f"Extracted {len(pdf_texts)} PDF files. Starting chunking...")
        sys.stdout.flush()
        
        #chunk 
        all_chunks, chunk_meta_data = chunk_files(pdf_texts)
        print(f"Created {len(all_chunks)} chunks. Checking for duplicates...")
        sys.stdout.flush()
        
        # Filter out duplicate chunks if we have existing ones
        if existing_chunks:
            unique_chunks = filter_duplicate_chunks(all_chunks, existing_chunks)
            if not unique_chunks:
                print("All chunks already exist. No new content to add.")
                return (faiss_service, embedding_service, database_service)
            
            # Update metadata for unique chunks only
            unique_meta_data = chunk_meta_data[:len(unique_chunks)]
            all_chunks = unique_chunks
            chunk_meta_data = unique_meta_data
        
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        sys.stdout.flush()
        
        # OpenAI can handle all chunks at once efficiently
        import time
        start_time = time.time()
        embedding = embedding_service.embed_documents(all_chunks)
        elapsed = time.time() - start_time
        print(f"Generated embeddings in {elapsed:.2f}s")
        sys.stdout.flush()
        print("Adding documents to Faiss index...")
        sys.stdout.flush()
        
        faiss_service.add_documents(embedding, all_chunks, chunk_meta_data)
        database_service.insert_document_list(chunk_meta_data)
        database_service.insert_chunk_list(all_chunks, chunk_meta_data)
        
        # Save the updated index
        try:
            faiss_service.save_index(index_path, documents_path)
            print(f"Saved index with {faiss_service.get_stats()['total_documents']} total documents")
        except Exception as save_error:
            logging.warning(f"Could not save index: {save_error}")

        return (faiss_service, embedding_service, database_service)
    
    except Exception as e:
        logging.error(f"Unable to load Faiss Service: {e}")
        raise e
    
#step 2 load embedding - initialize as None, load on first request
faiss_service = None
embedding_service = None
database_service = None
hybrid_search_service = None

def get_services():
    """Lazy initialization of services on first request."""
    global faiss_service, embedding_service, database_service, hybrid_search_service
    
    if faiss_service is None or embedding_service is None or database_service is None:
        try:
            print("Initializing Faiss and embedding services...")
            import sys
            sys.stdout.flush()  # Force output to show immediately
            
            faiss_service, embedding_service, database_service = load_faiss_service()
            
            # Initialize hybrid search service
            hybrid_search_service = HybridSearchService(
                faiss_service, database_service, embedding_service
            )
            
            print("Services initialized successfully!")
            sys.stdout.flush()
        except Exception as e:
            logging.error(f"Failed to initialize services: {e}")
            import traceback
            print(f"Full error traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=503, detail=f"Service initialization failed: {str(e)}")
    
    return faiss_service, embedding_service, database_service, hybrid_search_service



@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Server is running"}

@app.post("/initialize")
async def initialize_services():
    """Initialize services manually."""
    try:
        faiss_service, embedding_service, database_service, hybrid_search_service = get_services()
        return {
            "status": "success", 
            "message": "Services initialized successfully",
            "documents": faiss_service.get_stats()['total_documents'],
            "database_stats": database_service.get_stats(),
            "hybrid_search_available": hybrid_search_service is not None
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/search")
async def search(q: str) -> dict:
    """
    receive query q, embed it, retrieve top 3 passages, and return them
    """
    try:
        # Get services (initialize if needed)
        faiss_service, embedding_service, database_service, hybrid_search_service = get_services()
        
        #step 1: Retrieve relevant documents
        docs = faiss_service.search(embedding_service.embed_query(q))
        if not docs:
            raise HTTPException(status_code=404, detail=f"No documents found for query: {q}")
        
        # Format and return the results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            })
        
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"DEBUG ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/search_db")
async def search_database(q: str, limit: int = 10) -> dict:
    """
    Search documents using SQLite FTS5 full-text search
    """
    try:
        # Get services (initialize if needed)
        faiss_service, embedding_service, database_service, hybrid_search_service = get_services()
        
        # Search using SQLite FTS5
        results = database_service.search_documents(q, limit)
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No documents found for query: {q}")
        
        return {
            "query": q,
            "results": results,
            "count": len(results),
            "search_type": "sqlite_fts5"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"DEBUG ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/search_hybrid")
async def search_hybrid(q: str, limit: int = 10, alpha: float = 0.6) -> dict:
    """
    Hybrid search combining FAISS vector search, BM25, and FTS5 keyword search
    
    Args:
        q: Search query
        limit: Number of results to return (default: 10)
        alpha: Weight for vector vs keyword search (0-1, default: 0.6)
               Higher values favor semantic similarity, lower values favor keyword matching
    """
    try:
        # Get services (initialize if needed)
        faiss_service, embedding_service, database_service, hybrid_search_service = get_services()
        
        if hybrid_search_service is None:
            raise HTTPException(status_code=503, detail="Hybrid search service not available")
        
        # Perform hybrid search
        results = hybrid_search_service.search(q, limit=limit, alpha=alpha)
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No documents found for query: {q}")
        
        return {
            "query": q,
            "results": results,
            "count": len(results),
            "search_type": "hybrid",
            "alpha": alpha,
            "description": f"Hybrid search with {alpha:.1%} vector weight, {1-alpha:.1%} keyword weight"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"DEBUG ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/documents")
async def get_all_documents() -> dict:
    """
    Get all documents from the database
    """
    try:
        # Get services (initialize if needed)
        faiss_service, embedding_service, database_service, hybrid_search_service = get_services()
        
        documents = database_service.get_all_documents()
        
        return {
            "documents": documents,
            "count": len(documents)
        }
        
    except Exception as e:
        import traceback
        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"DEBUG ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats")
async def get_stats() -> dict:
    """
    Get system statistics
    """
    try:
        # Get services (initialize if needed)
        faiss_service, embedding_service, database_service, hybrid_search_service = get_services()
        
        return {
            "faiss_stats": faiss_service.get_stats(),
            "database_stats": database_service.get_stats()
        }
        
    except Exception as e:
        import traceback
        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"DEBUG ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")