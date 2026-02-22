from abc import ABC, abstractmethod
from backend.orchestrator.state import AgentState

class BaseAgent(ABC):
    """
    Abstract interface for all specialized agents in the Graph.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    @abstractmethod
    def execute(self, state: AgentState) -> dict:
        """
        Executes the agent's core capability on the current state.
        Returns a dictionary containing the state updates to be merged.
        """
        pass
