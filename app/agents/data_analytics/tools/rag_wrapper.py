import logging
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
async def enterprise_rag_tool(query: str, session_id: Optional[str] = "analytics_sub_session") -> str:
    """
    Use this tool exclusively when the user asks questions about enterprise text documents, 
    uploaded PDFs, DOCX files, company policies, or standard non-mathematical enterprise knowledge.
    This tool searches the internal Vector Database and securely synthesizes a grounded answer.
    """
    try:
        # Import dynamically to prevent circular dependencies on startup
        from app.api.routes import _get_orchestrator
        
        logger.info(f"[ANALYTICS] Sub-Agent triggering Master RAG Toolkit for query: '{query}'")
        orchestrator = _get_orchestrator()
        
        # Execute the 100% native, untouched 12-stage Master RAG pipeline
        state = await orchestrator.invoke(
            query=query,
            session_id=session_id,
            model_provider="auto",
            force_session_context=False 
        )
        
        answer = state.get("answer", "No textual answer could be synthesized from the documents.")
        return answer
        
    except Exception as e:
        logger.error(f"[RAG WRAPPER TOOL] Execution Error: {e}")
        return f"System Architectural Error executing Enterprise RAG sub-routine: {str(e)}"
