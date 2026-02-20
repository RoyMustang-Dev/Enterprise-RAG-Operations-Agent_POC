from sentence_transformers import SentenceTransformer
from typing import List, Union

class EmbeddingModel:
    """
    Singleton class to handle text embedding generation using sentence-transformers.
    
    Attributes:
    Handles generation of text embeddings using sentence-transformers.
    Uses BAAI/bge-large-en-v1.5 for high-quality enterprise retrieval.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        print(f"Loading embedding model: {model_name}...")
        # normalize_embeddings=True is highly recommended for BGE models for cosine similarity
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for a single text string.
        
        Args:
            text (str): Input text.
            
        Returns:
            List[float]: The vector embedding.
        """
        if not text:
            return []
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings (batch processing).
        
        Args:
            texts (List[str]): List of input texts.
            
        Returns:
            List[List[float]]: List of vector embeddings.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
