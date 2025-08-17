from typing import List

def chunk_text(text: str, max_tokens: int=512, overlap: int=50)-> List[str]:
    tokens = text.split()
    chunks = []

    steps = max_tokens - overlap
    for i in range(0, len(tokens), steps):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    
    return chunks