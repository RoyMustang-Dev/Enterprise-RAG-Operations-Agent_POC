import sys
import os
import pytest

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.embeddings.embedding_model import EmbeddingModel

def test_embedding_generation():
    print("\n[TEST] Loading Embedding Model...")
    embedder = EmbeddingModel()
    
    text = "This is a test sentence for embedding."
    print(f"[TEST] Generating embedding for: '{text}'")
    
    vector = embedder.generate_embedding(text)
    
    assert isinstance(vector, list), "Embedding should be a list"
    assert len(vector) == 384, f"Expected dimension 384, got {len(vector)}"
    assert all(isinstance(x, float) for x in vector), "Embedding elements should be floats"
    
    print("[SUCCESS] Embedding generation verified.")

def test_batch_embedding_generation():
    embedder = EmbeddingModel()
    texts = ["Sentence one.", "Sentence two."]
    
    print(f"[TEST] Generating batch embeddings for {len(texts)} texts...")
    vectors = embedder.generate_embeddings(texts)
    
    assert len(vectors) == 2, "Should return 2 vectors"
    assert len(vectors[0]) == 384, "Vector dimension should be 384"
    
    print("[SUCCESS] Batch embedding generation verified.")

if __name__ == "__main__":
    test_embedding_generation()
    test_batch_embedding_generation()
