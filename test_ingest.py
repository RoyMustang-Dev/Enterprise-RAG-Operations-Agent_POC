from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.vector_store import QdrantStore

store = QdrantStore()
store.clear()  # reset the db to clear any non-tenant orphan chunks

p = IngestionPipeline(tenant_id='aditya-ds')
# Simulate a document
p.run_ingestion(['hello world', 'enterprise rag architecture is great', 'aditya ds resume'], metadatas=[{'source': 'test'}, {'source': 'test'}, {'source': 'test'}])
