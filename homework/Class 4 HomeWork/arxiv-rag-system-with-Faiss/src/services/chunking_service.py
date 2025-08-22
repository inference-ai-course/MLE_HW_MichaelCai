import os
from typing import List, Tuple

def chunk_files(docs: List[str]) -> Tuple[List[str], List[dict]]:

    all_chunks = []
    chunk_meta_data = []

    for i, text in enumerate(docs):
        chunks = chunk_text(text, int(os.getenv("MAX_TOKENS", "512")), int(os.getenv("OVER_LAP", "50"))) 
        all_chunks.extend(chunks)
        for j, chunk in enumerate(chunks):
            chunk_meta_data.append({
                "document_id": i,
                "chunk_id": j,
                "source": f"pdf_{i}",
                "chunk_length": len(chunk)
            })
    print(f"Created {len(all_chunks)} chunks from {len(docs)} documents")
    return all_chunks, chunk_meta_data

def chunk_text(text: str, max_tokens: int=512, overlap: int=50)-> List[str]:
    tokens = text.split()
    chunks = []

    steps = max_tokens - overlap
    for i in range(0, len(tokens), steps):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    
    return chunks