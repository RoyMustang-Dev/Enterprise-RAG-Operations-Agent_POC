from typing import List, Dict, Tuple
from backend.embeddings.embedding_model import EmbeddingModel
from backend.vectorstore.faiss_store import FAISSStore
import os
import threading
import json
from datetime import datetime
from backend.orchestrator.graph import GalactusOrchestrator

class UserVectorMemory:
    """Persistent semantic memory of each user's conversation."""
    def __init__(self, embedding_model: EmbeddingModel, user_id: str = "default_user"):
        self.user_id = user_id
        self.embedder = embedding_model
        self.memory_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'memory')
        os.makedirs(self.memory_dir, exist_ok=True)
        self.index_path = os.path.join(self.memory_dir, f"{user_id}.faiss")
        self.meta_path = os.path.join(self.memory_dir, f"{user_id}.json")
        self.store = FAISSStore(index_file=self.index_path, meta_file=self.meta_path)

    def remember(self, text: str, metadata: dict = None):
        emb = self.embedder.generate_embedding(text)
        self.store.add_documents([text], [emb], [metadata or {}])
        self.store.save()

    def recall(self, query: str, k: int = 3) -> List[Dict]:
        emb = self.embedder.generate_embedding(query)
        if len(self.store.metadata) == 0:
            return []
        try:
            return self.store.search(emb, k=k)
        except Exception:
            return []


class RAGService:
    """
    Adapter wrapper that instantiates the Agentic Orchestrator and maintains UI compatibility.
    """

    def __init__(self, **kwargs):
        self.embedding_model = EmbeddingModel()
        self.vector_store = FAISSStore()
        self.model_provider = kwargs.get("model_provider", "ollama").lower()
        self.model_name = kwargs.get("model_name", "llama3.2:1b")
        api_key = kwargs.get("api_key", None)

        print(f"RAGService Adapter: Initializing {self.model_provider} provider...")
        if self.model_provider == "sarvam":
            from backend.generation.llm_provider import SarvamClient
            self.llm_client = SarvamClient(api_key=api_key, model=self.model_name)
        else:
            from backend.generation.llm_provider import OllamaClient
            self.llm_client = OllamaClient(model=self.model_name)
            
        self.user_memory = UserVectorMemory(self.embedding_model)
        
        # Instantiate the new Graph Architecture
        self.orchestrator = GalactusOrchestrator(
            llm_client=self.llm_client,
            faiss_store=self.vector_store,
            embedding_model=self.embedding_model
        )

    def answer_query(self, query: str, initial_k: int = 20, final_k: int = 5, status_callback=None, streaming_callback=None) -> Dict:
        """
        Executes the Agentic Graph and maintains the dictionary UI contract.
        """
        if status_callback:
            status_callback("Supervisor routing...")
            
        # Recall Memory and format it
        past_interactions = self.user_memory.recall(query, k=2)
        chat_history = []
        for p in past_interactions:
            # We just parse strings back if needed, but Graph handles chat_history natively as dicts. For now pass raw text.
            chat_history.append({"role": "system_memory", "content": p.get("text", "")})

        # Execute Graph
        result_state = self.orchestrator.execute(
            query=query,
            streaming_callback=streaming_callback,
            chat_history=chat_history
        )
        
        # Async Audit Logging
        self._audit_log({
            "query": query,
            "intent": result_state.get("intent"),
            "model_provider": self.model_provider,
            "latency_optimizations": result_state.get("optimizations", {})
        })
        
        # Update UI Context Contract
        return {
            "answer": result_state.get("answer", "No answer generated."),
            "sources": result_state.get("sources", []),
            "confidence": result_state.get("confidence", 0.0),
            "verifier_verdict": result_state.get("verifier_verdict", "UNKNOWN"),
            "is_hallucinated": result_state.get("is_hallucinated", False),
            "attributions": [],
            "search_query": result_state.get("search_query", query),
            "optimizations": result_state.get("optimizations", {})
        }

    def _audit_log(self, payload: dict):
        """Asynchronously log interactions for observability."""
        def write_log():
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'audit')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'audit_logs.jsonl')
            payload["timestamp"] = datetime.utcnow().isoformat()
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            except Exception as e:
                print(f"Audit Log Failed: {e}")
        threading.Thread(target=write_log, daemon=True).start()
