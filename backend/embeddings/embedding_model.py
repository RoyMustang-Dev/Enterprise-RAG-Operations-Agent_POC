from sentence_transformers import SentenceTransformer
from typing import List, Union

class EmbeddingModel:
    """
    Singleton class to handle text embedding generation using sentence-transformers.
    
    Attributes:
        model_name (str): The name of the model to use (default: 'all-MiniLM-L6-v2').
        model (SentenceTransformer): The loaded model instance.
    """
    _instance = None
    
    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.model_name = model_name
            print(f"Loading embedding model: {model_name}...")
            cls._instance.model = SentenceTransformer(model_name)
            print("Model loaded successfully.")
        return cls._instance

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
