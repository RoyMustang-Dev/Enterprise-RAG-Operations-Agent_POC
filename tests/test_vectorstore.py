import sys
import os
import shutil
import numpy as np

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.vectorstore.qdrant_store import QdrantStore

TEST_PATH = "tests/data/qdrant_test_storage"

def setup_test_env():
    if os.path.exists("tests/data"):
        shutil.rmtree("tests/data", ignore_errors=True)
    os.makedirs(TEST_PATH)

def test_qdrant_store():
    setup_test_env()
    
    print("\n[TEST] Initializing Qdrant Store...")
    # Initialize with small dimension for testing
    store = QdrantStore(dimension=4, collection_name="test_collection", path=TEST_PATH)
    
    chunks = ["doc1", "doc2", "doc3"]
    # Create dummy embeddings (3 docs, 4 dimensions)
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]
    metadatas = [{"source": "s1"}, {"source": "s2"}, {"source": "s3"}]
    
    print(f"[TEST] Adding {len(chunks)} documents...")
    store.add_documents(chunks, embeddings, metadatas)
    
    assert store.ntotal == 3, "Should have 3 documents in index"
    
    # Test Search
    query = [1.0, 0.0, 0.0, 0.0] # Should match doc1
    print(f"[TEST] Searching for {query}...")
    results = store.search(query, k=1)
    
    assert len(results) == 1
    assert results[0]['text'] == "doc1"
    assert results[0]['source'] == "s1"
    print(f"[SUCCESS] Search result: {results[0]}")
    
    # Test Persistence (Qdrant does this natively)
    print("[TEST] Testing load persistence...")
    
    # Release the local database lock
    store.client.close()
    
    new_store = QdrantStore(dimension=4, collection_name="test_collection", path=TEST_PATH)
    
    assert new_store.ntotal == 3, "Loaded store should have 3 documents implicitly"
    print("[SUCCESS] Persistence verified.")

if __name__ == "__main__":
    test_qdrant_store()
