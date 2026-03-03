# Qdrant Multi-Tenant RCA & Fixes

## 1. The "Data Not Found" Root Cause
**Symptom**: Querying the `aditya-ds` vector database always triggered the RAG "smart fallback" and yielded 0 context chunks, resulting in the agent returning `DATA_NOT_FOUND`.
**Root Cause**: In [app/ingestion/pipeline.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/ingestion/pipeline.py), the `tenant_id` argument was never passed down into the [add_documents](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/retrieval/vector_store.py#171-193) payloads. Qdrant correctly appended `FieldCondition(key="tenant_id")`, but no documents in the database actually had this tag.

**Fix Applied**: 
- Refactored [pipeline.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/ingestion/pipeline.py) to persist `tenant_id` on instantiation and explicitly pipe it into `vector_store.add_documents(..., tenant_id=self.tenant_id)`.
- Modified [vector_store.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/retrieval/vector_store.py) so [add_documents](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/retrieval/vector_store.py#171-193) actively binds `meta["tenant_id"] = tenant_id`.

## 2. The `500 Internal Server Error` (Internal Generation Error)
**Symptom**: During live queries against the fixed search endpoints, [rag.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/agents/rag.py) completely crashed, bubbling up a 500 error on the `/chat` route instead of answering the query.
**Root Cause**: The orchestrator router `execute_async()` expects keyword arguments `query=query, tenant_id=tenant_id`, etc. But inside [router.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/supervisor/router.py), the `query` variable was incorrectly parsed causing `query.lower()` to fail with an `AttributeError: dict object has no attribute lower`. This is because the execution graph state was mishandled during the payload extraction.  Furthermore, the [vector_store.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/retrieval/vector_store.py) singleton pattern prevented proper per-tenant filtering in memory if `QDRANT_MULTI_TENANT=true`.

**Fix Applied**:
- Removed the state-mutation singletons in [vector_store.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/retrieval/vector_store.py) (`self.tenant_id = tenant_id` global caching).
- Switched to completely dynamic keyword parameters across [rag.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/agents/rag.py) where `tenant_id` is supplied natively into `.search(..., tenant_id=tenant_id)` instead of modifying the global object schema. 
- Overhauled the `Exception` catch block in [routes.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/api/routes.py) and [router.py](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/supervisor/router.py) to correctly log Python Tracebacks so these silent semantic bugs don't get hidden under generic `500` codes in the future.

### Impact on Multi-Modality
Since the base [QdrantStore](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/retrieval/vector_store.py#26-241) handles standard indexing indiscriminately, whether an uploaded file is a PDF, Text file, or an Image description, it traverses the exact same [add_documents](file:///d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app/retrieval/vector_store.py#171-193) payload wrapper. With the dynamic `tenant_id=tenant_id` bindings fixed, multi-modal ingestion securely isolates the data per user session/tenant.
