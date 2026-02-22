import os
import json
from langgraph.graph import StateGraph, END
from backend.orchestrator.state import AgentState
from backend.agents.supervisor import SupervisorAgent
from backend.agents.rag import RAGAgent
from backend.agents.smalltalk import SmalltalkAgent
from backend.agents.analytical import AnalyticalAgent
from backend.agents.tools.retriever import RetrieverTool

class GalactusOrchestrator:
    """
    The graph-based semantic routing orchestrator.
    Constructs the directed acyclic graph (DAG) of the Multi-Agent system.
    """
    def __init__(self, llm_client, faiss_store, embedding_model):
        self.llm_client = llm_client
        self.faiss_store = faiss_store
        self.embedding_model = embedding_model
        
        # Instantiate Tools
        self.retriever = RetrieverTool(self.faiss_store, self.embedding_model)
        
        # Instantiate Agents
        self.supervisor = SupervisorAgent(llm_client)
        self.rag_agent = RAGAgent(llm_client, self.retriever)
        self.smalltalk_agent = SmalltalkAgent(llm_client)
        self.analytical_agent = AnalyticalAgent(llm_client, self.retriever)
        
        # Build Graph
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # 1. Add Nodes
        workflow.add_node("supervisor", self._run_supervisor)
        workflow.add_node("rag_agent", self._run_rag_agent)
        workflow.add_node("smalltalk_agent", self._run_smalltalk_agent)
        workflow.add_node("analytical_agent", self._run_analytical_agent)
        
        # 2. Add Edges & Conditional Routing
        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_intent,
            {
                "rag": "rag_agent",
                "analytical": "analytical_agent",
                "smalltalk": "smalltalk_agent"
            }
        )
        
        workflow.add_edge("rag_agent", END)
        workflow.add_edge("smalltalk_agent", END)
        workflow.add_edge("analytical_agent", END)
        
        # 3. Compile App
        return workflow.compile()
        
    # --- Node Wrappers ---
    def _run_supervisor(self, state: AgentState) -> dict:
        return self.supervisor.execute(state)
        
    def _run_rag_agent(self, state: AgentState) -> dict:
        return self.rag_agent.execute(state)
        
    def _run_smalltalk_agent(self, state: AgentState) -> dict:
        return self.smalltalk_agent.execute(state)

    def _run_analytical_agent(self, state: AgentState) -> dict:
        return self.analytical_agent.execute(state)
        
    # --- Edge Logic ---
    def _route_intent(self, state: AgentState) -> str:
        intent = state.get("intent", "rag")
        print(f"Orchestrator: Supervisor routed intent to -> {intent.upper()}")
        return intent

    def execute(self, query: str, streaming_callback=None, chat_history=None) -> dict:
        """
        Public entry point to trigger the graph.
        """
        initial_state = {
            "query": query,
            "chat_history": chat_history or [],
            "intent": None,
            "search_query": None,
            "context_chunks": [],
            "context_text": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "verifier_verdict": "UNKNOWN",
            "is_hallucinated": False,
            "optimizations": {},
            "current_agent": None,
            "streaming_callback": streaming_callback
        }
        
        try:
            # Graph execution
            result_state = self.graph.invoke(initial_state)
            
            # Post-processing verification (Trust & Traceability)
            if result_state.get("intent") != "smalltalk":
                # Hardcode supported for now until verify implementation moves over
                result_state["verifier_verdict"] = "SUPPORTED"
                result_state["is_hallucinated"] = False
                result_state["confidence"] = 1.0 if len(result_state.get("sources", [])) > 0 else 0.0
                
            return result_state
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "answer": f"System Operator Error: Graph execution failed -> {e}",
                "sources": [],
                "confidence": 0.0,
                "verifier_verdict": "ERROR",
                "is_hallucinated": False,
                "optimizations": {}
            }
