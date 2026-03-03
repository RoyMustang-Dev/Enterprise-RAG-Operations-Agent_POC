from app.retrieval.vector_store import QdrantStore
import json

try:
    store = QdrantStore()
    
    # Get all points (no filter)
    res, offset = store.client.scroll(
        collection_name=store.collection_name, 
        limit=10, 
        with_payload=True,
        with_vectors=False
    )
    print("TOTAL Raw points:", len(res))
    if len(res) > 0:
        for p in res:
             print("ID:", p.id, "Payload:", p.payload)
except Exception as e:
    import traceback
    traceback.print_exc()
