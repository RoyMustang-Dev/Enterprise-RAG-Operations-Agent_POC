from app.retrieval.vector_store import QdrantStore
import json

try:
    store = QdrantStore()
    tensor = [0.1] * store.dimension
    tenant = 'aditya-ds'
    
    # Simulate exact fallback RAG retrieval filter that triggered it
    res = store.search(tensor, k=5, metadata_filters=None, tenant_id=tenant)
    print("SUCCESS count:", len(res))
    if len(res) > 0:
        print("First doc:", res[0].get('page_content'))
except Exception as e:
    import traceback
    traceback.print_exc()
