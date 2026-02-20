from backend.vectorstore.faiss_store import FAISSStore
from backend.embeddings.embedding_model import EmbeddingModel

def debug_retrieval(query):
    store = FAISSStore()
    embedder = EmbeddingModel()
    
    print(f"Total documents in store: {len(store.metadata)}")
    print(f"Unique sources: {store.get_all_documents()}")
    
    query_vec = embedder.generate_embedding(query)
    results = store.search(query_vec, k=10)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} chunks.")
    for i, res in enumerate(results):
        print(f"\nRank {i+1}:")
        print(f"Source: {res.get('source')}")
        print(f"Score: {res.get('score')}")
        print(f"Text Snippet: {res.get('text')[:200]}...")

if __name__ == "__main__":
    with open("debug_output.txt", "w", encoding="utf-8") as f:
        import sys
        sys.stdout = f
        debug_retrieval("Akobot")
        print("-" * 50)
        debug_retrieval("Updated_Resume_DS")
