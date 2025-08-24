from difflib import SequenceMatcher
try:
    import editdistance
except ImportError:
    editdistance = None
from typing import List

def text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text chunks using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()

def text_similarity_edit_distance(text1: str, text2: str) -> float:
    """Calculate similarity between two text chunks using edit distance."""
    if not editdistance:
        raise ImportError("editdistance package not installed. Run: pip install editdistance")
    
    text1_clean = text1.lower().strip()
    text2_clean = text2.lower().strip()
    
    distance = editdistance.eval(text1_clean, text2_clean)
    max_len = max(len(text1_clean), len(text2_clean))
    return 1.0 - (distance / max_len) if max_len > 0 else 1.0

def filter_duplicate_chunks(new_chunks: List[str], existing_chunks: List[str], threshold: float = 0.85) -> List[str]:
    """Filter out chunks that are too similar to existing ones."""
    unique_chunks = []
    duplicate_count = 0
    
    for new_chunk in new_chunks:
        is_duplicate = False
        for existing_chunk in existing_chunks:
            # Use edit distance if available, otherwise fallback to SequenceMatcher
            if editdistance:
                similarity = text_similarity_edit_distance(new_chunk, existing_chunk)
            else:
                similarity = text_similarity(new_chunk, existing_chunk)
            if similarity >= threshold:
                is_duplicate = True
                duplicate_count += 1
                break
        
        if not is_duplicate:
            unique_chunks.append(new_chunk)
    
    print(f"Filtered out {duplicate_count} duplicate chunks. Adding {len(unique_chunks)} new unique chunks.")
    return unique_chunks