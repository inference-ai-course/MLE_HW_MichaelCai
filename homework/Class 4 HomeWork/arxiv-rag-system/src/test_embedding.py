import os
from services.text_extraction_service import extract_text_from_folder
from services.chunking_service import chunk_text
from services.embedding_service import EmbeddingService

def main():
    # Step 1: Load configuration from .env file
    print("Loading configuration from .env file...")
    
    # The embedding service will automatically load from .env file
    # No need to manually handle the API key here!
    
    # Step 2: Extract text from PDFs
    folder_path = "../data/pdfs"
    print("Extracting text from PDFs...")
    pdf_texts = extract_text_from_folder(folder_path, max_files=2)
    
    if not pdf_texts:
        print("No PDFs found or extracted.")
        return
    
    # Step 3: Chunk the extracted text
    print("Chunking text...")
    all_chunks = []
    chunk_metadata = []
    
    for i, text in enumerate(pdf_texts):
        chunks = chunk_text(text, max_tokens=512, overlap=50)
        all_chunks.extend(chunks)
        
        # Add metadata for each chunk
        for j, chunk in enumerate(chunks):
            chunk_metadata.append({
                "document_id": i,
                "chunk_id": j,
                "source": f"pdf_{i}",
                "chunk_length": len(chunk)
            })
    
    print(f"Created {len(all_chunks)} chunks from {len(pdf_texts)} documents")
    
    # Step 4: Create vector store with OpenAI embeddings
    print("Creating vector store with OpenAI embeddings...")
    try:
        # No need to pass API key - it's loaded from .env file automatically!
        embedding_service = EmbeddingService(
            persist_directory="./arxiv_vectorstore"
        )
        
        # Create vector store
        vectorstore = embedding_service.create_vectorstore(all_chunks, chunk_metadata)
        print(f"Vector store created with {len(all_chunks)} chunks")
        print(f"Embedding dimension: {embedding_service.get_embedding_dimension()}")
        
        # Step 5: Test similarity search
        print("\n" + "="*50)
        print("Testing similarity search...")
        print("="*50)
        
        test_queries = [
            "machine learning algorithms",
            "neural networks",
            "deep learning models",
            "artificial intelligence"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 40)
            
            # Search for similar chunks
            results = embedding_service.similarity_search_with_score(query, k=3)
            
            for i, (doc, score) in enumerate(results):
                print(f"{i+1}. Score: {score:.4f}")
                print(f"   Content: {doc.page_content[:150]}...")
                print(f"   Metadata: {doc.metadata}")
                print()
        
        # Step 6: Persist the vector store
        print("Persisting vector store to disk...")
        embedding_service.persist()
        print("Vector store saved successfully!")
        
        # Step 7: Demo loading existing vector store
        print("\nDemo: Loading existing vector store...")
        new_service = EmbeddingService(persist_directory="./arxiv_vectorstore")
        loaded_store = new_service.load_vectorstore()
        if loaded_store:
            print("Vector store loaded successfully!")
            
            # Test search on loaded store
            results = new_service.similarity_search("machine learning", k=2)
            print(f"Found {len(results)} results from loaded store")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("To install required packages, run:")
        print("pip install langchain chromadb openai")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your OpenAI API key is valid and you have sufficient credits")

if __name__ == "__main__":
    main()