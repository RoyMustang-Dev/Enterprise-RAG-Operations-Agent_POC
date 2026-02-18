from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Splits text into smaller chunks for vector embedding.
    
    This implementation uses a word-based splitting strategy.
    For production (Phase 2), we may switch to a token-based splitter (e.g., using Tiktoken) 
    to match the specific context window of the embedding model.
    
    Args:
        text (str): The full text content to be chunked.
        chunk_size (int): The target size of each chunk in words (default: 512).
        overlap (int): The number of words to overlap between chunks to preserve context (default: 50).
        
    Returns:
        List[str]: A list of text chunks.
    """
    if not text:
        return []
        
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        # Determine the end index for the current chunk
        end = min(start + chunk_size, len(words))
        
        # Create the chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Stop if we've reached the end
        if end == len(words):
            break
            
        # Move the start pointer, accounting for overlap
        start += (chunk_size - overlap)
        
    return chunks
