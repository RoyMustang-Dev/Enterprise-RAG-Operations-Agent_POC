import sys
import os
import shutil
import numpy as np

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.vectorstore.faiss_store import FAISSStore

TEST_INDEX_PATH = "tests/data/test_index.faiss"
TEST_META_PATH = "tests/data/test_meta.pkl"

def setup_test_env():
    if os.path.exists("tests/data"):
        shutil.rmtree("tests/data")
    os.makedirs("tests/data")

def test_faiss_store():
    setup_test_env()
    
    print("\n[TEST] Initializing FAISS Store...")
    # Initialize with small dimension for testing
    store = FAISSStore(dimension=4, index_path=TEST_INDEX_PATH, metadata_path=TEST_META_PATH)
    
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
    
    assert store.index.ntotal == 3, "Should have 3 documents in index"
    
    # Test Search
    query = [1.0, 0.0, 0.0, 0.0] # Should match doc1
    print(f"[TEST] Searching for {query}...")
    results = store.search(query, k=1)
    
    assert len(results) == 1
    assert results[0]['text'] == "doc1"
    assert results[0]['source'] == "s1"
    print(f"[SUCCESS] Search result: {results[0]}")
    
    # Test Persistence
    print("[TEST] Testing save/load persistence...")
    store.save()
    
    new_store = FAISSStore(dimension=4, index_path=TEST_INDEX_PATH, metadata_path=TEST_META_PATH)
    new_store.load()
    
    assert new_store.index.ntotal == 3, "Loaded store should have 3 documents"
    print("[SUCCESS] Persistence verified.")

if __name__ == "__main__":
    test_faiss_store()
