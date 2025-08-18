import os
from services.text_extraction_service import extract_text_from_folder
from services.embedding_service import EmbeddingService
from services.chunking_service import chunk_files
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from fastapi import FastAPI, HTTPException
from typing import Dict
import logging

app = FastAPI()

#step 1, load env file 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError: 
    logging.warning("Warning: python-dotenv not installed. Run: pip install python-dotenv")
    logging.warning("Environment variables will be loaded from system only.")

def load_embedding_service() -> EmbeddingService: 
    print("getting into load embedding service")
    try:
        embedding_service = EmbeddingService()

        vectorstore = embedding_service.load_vectorstore()
        
        #no vectore store existed. Create a new one
        print("trying to get vector store")
        if vectorstore._collection.count()==0:
            print("vector store don't have it")
            #load pdf files
            folder_path = os.getenv("PDF_FOLDER_PATH")
            print(f"extracting PDF files from folder {folder_path}")
            pdf_texts = extract_text_from_folder(folder_path, max_files= 3)
            if not pdf_texts:
                logging.warning(f"No PDF is found within this directory {folder_path}")
                raise Exception("Please make sure PDFs are added in the folder")
            
            #chunk 
            all_chunks, chunk_meta_data = chunk_files(pdf_texts)
            embedding_service.add_chunks(all_chunks, chunk_meta_data)
     
            
        return embedding_service
    
    except Exception as e:
        logging.error(f"Unable to load Vector store: {e}")
        return None
    
#step 2 Get Vector store 
print("About to call load_embedding_service...")
embedding_service = load_embedding_service()
print("Finished calling load_embedding_service...")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.get("/search")
async def search(q: str) -> dict:
    """
    receive query q, embed it, retrieve top 3 passages, and return them
    """
    try:
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service unavailable")
        
        #step 1: Retrieve relevant documents
        docs = embedding_service.similarity_search(q)
        if not docs:
            raise HTTPException(status_code=404, detail=f"No documents found for query: {q}")
        
        
        # Create a simple RAG prompt
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Question: {question}
        
        Context: {context}
        
        Answer:
        """)

        # Using Ollama API for LLM
        OLLAMA_URL = "http://localhost:11434"
        OLLAMA_MODEL = "llama3.1:8b"  # or whatever model name you have in Ollama
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL
        )
       
        qa_chain=(
            {
                "context": embedding_service.get_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm 
            | StrOutputParser()
        )

        result = qa_chain.invoke(q)
        return {"query": q, "answer": result, "status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"DEBUG ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

                

    