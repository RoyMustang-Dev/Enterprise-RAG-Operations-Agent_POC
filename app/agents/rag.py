"""
RAG Agent Orchestrator (LangGraph)

This module constructs the directed acyclic graph (DAG) specifically for Retrieval-Augmented Generation.
It physically hooks together Phase 4 (Retrieval) and Phase 5 (Reasoning) into a strict, fault-tolerant pipeline.
"""
import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from app.core.types import AgentState

# Phase 4 Imports
from app.retrieval.metadata_extractor import MetadataExtractor
from app.retrieval.vector_store import QdrantStore
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.reranker import SemanticReranker

# Phase 5 Imports
from app.reasoning.synthesis import SynthesisEngine
from app.reasoning.verifier import HallucinationVerifier
from app.reasoning.formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class RAGAgent:
    """
    Constructs the `rag_question` execution path using LangGraph.
    Data flows strictly: Metadata -> Retrive -> Rerank -> Synthesize -> Verify -> Format.
    """
    
    def __init__(self):
        """
        Instantiates all immutable heavy models required for the pipeline.
        These are loaded once to prevent RAM/VRAM exhaustion on every request.
        """
        logger.info("[RAG AGENT] Initializing Core Architecture Models...")
        self.metadata_extractor = MetadataExtractor()
        self.embedding_model = EmbeddingModel()
        self.vector_store = QdrantStore()
        self.reranker = SemanticReranker()
        self.synthesis_engine = SynthesisEngine()
        self.verifier = HallucinationVerifier()
        
        # Build the physical state computer
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """
        Defines the sequential nodes and edges of the LLM computational graph.
        """
        workflow = StateGraph(AgentState)
        
        # Node Definitions
        workflow.add_node("extract_metadata", self.node_extract_metadata)
        workflow.add_node("retrieve_documents", self.node_retrieve_documents)
        workflow.add_node("rerank_documents", self.node_rerank_documents)
        workflow.add_node("synthesize_answer", self.node_synthesize_answer)
        workflow.add_node("verify_answer", self.node_verify_answer)
        workflow.add_node("format_output", self.node_format_output)
        
        # Edge Definitions (Strict Sequential Pipeline)
        workflow.set_entry_point("extract_metadata")
        workflow.add_edge("extract_metadata", "retrieve_documents")
        
        # Conditional Edge: If retrieval finds nothing, short-circuit to formatting
        workflow.add_conditional_edges(
            "retrieve_documents",
            lambda state: "rerank_documents" if len(state.get("context_chunks", [])) > 0 else "format_output"
        )
        
        workflow.add_edge("rerank_documents", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "verify_answer")
        workflow.add_edge("verify_answer", "format_output")
        workflow.add_edge("format_output", END)
        
        return workflow.compile()
        
    def invoke(self, state: AgentState) -> AgentState:
        """
        Executes the compiled LangGraph pipeline.
        """
        logger.info(f"[RAG AGENT] Commencing Execution DAG for query: {state['query'][:50]}...")
        # Deep copy to ensure immutable updates structurally
        result = self.graph.invoke(state)
        return result

    # -------------------------------------------------------------------------
    # Graph Node Definitions
    # -------------------------------------------------------------------------

    def node_extract_metadata(self, state: AgentState) -> AgentState:
        """Dynamically extracts strict Qdrant bounds from natural language."""
        query = state["query"]
        extraction = self.metadata_extractor.extract_filters(query)
        
        # We store the applied filters in optimizations for audit tracking
        if "optimizations" not in state:
            state["optimizations"] = {}
        state["optimizations"]["metadata_filters"] = extraction.get("filters", {})
        
        # Normally you'd assign a rewritten query here, but we pass the raw intent down
        state["search_query"] = query 
        return state

    def node_retrieve_documents(self, state: AgentState) -> AgentState:
        """Executes the high-speed L2 distance fetch against the Vector DB."""
        query = state["search_query"] or state["query"]
        filters = state.get("optimizations", {}).get("metadata_filters", {})
        
        # 1. Generate Query Embedding
        query_tensor = self.embedding_model.generate_embedding(query)
        
        # 2. Execute L2 Distance Search (Requesting 30 chunks for high-recall)
        # Note: In a production Qdrant cloud environment, `filters` would be mapped 
        # to a `models.Filter` object here dynamically. 
        chunks = self.vector_store.search(query_tensor, k=30)
        
        state["context_chunks"] = chunks
        
        if not chunks:
            logger.warning("[RAG AGENT] Retrieval yielded 0 chunks. Bypassing synthesis.")
            state["answer"] = "I could not find any internal documents matching your request."
            state["confidence"] = 1.0
            
        return state

    def node_rerank_documents(self, state: AgentState) -> AgentState:
        """Executes the GPU semantic cross-encoder to isolate the Top 5 chunks."""
        query = state["search_query"] or state["query"]
        raw_chunks = state["context_chunks"]
        
        # Compress 30 -> 5
        refined_chunks = self.reranker.rerank(query, raw_chunks, top_k=5)
        state["context_chunks"] = refined_chunks
        return state

    def node_synthesize_answer(self, state: AgentState) -> AgentState:
        """Executes the 70B generation bounded by the strictly retrieved chunks."""
        query = state["query"]
        chunks = state["context_chunks"]
        
        synthesis_result = self.synthesis_engine.synthesize(query, chunks)
        
        state["answer"] = synthesis_result.get("answer", "")
        state["sources"] = synthesis_result.get("provenance", [])
        state["confidence"] = synthesis_result.get("confidence", 0.0)
        
        return state

    def node_verify_answer(self, state: AgentState) -> AgentState:
        """Audits the 70B output utilizing the independent Sarvam Verifier."""
        draft_answer = state["answer"]
        chunks = state["context_chunks"]
        
        verification = self.verifier.verify(draft_answer, chunks)
        
        state["verifier_verdict"] = verification.get("overall_verdict", "UNVERIFIED")
        state["is_hallucinated"] = verification.get("is_hallucinated", False)
        
        # Adjust confidence mathematically if Sarvam detects a logic flaw
        if state["is_hallucinated"]:
             state["confidence"] = state["confidence"] * 0.5
             
        return state

    def node_format_output(self, state: AgentState) -> AgentState:
        """Injects Markdown citational formatting ensuring Streamlit UI compatibility."""
        # The formatter returns the complete output dictionary mapping
        formatted_state = ResponseFormatter.construct_final_response(state)
        
        # Mutate the mutable LangGraph state safely
        state["answer"] = formatted_state["answer"]
        state["sources"] = formatted_state["sources"]
        state["confidence"] = formatted_state["confidence"]
        state["verifier_verdict"] = formatted_state["verifier_verdict"]
        state["is_hallucinated"] = formatted_state["is_hallucinated"]
        state["optimizations"] = formatted_state["optimizations"]
        
        logger.info("[RAG AGENT] Execution DAG Completed.")
        return state
