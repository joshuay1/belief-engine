from typing import Protocol, List, Dict, Tuple, Any

class DebatingAgent(Protocol):
    """
    Protocol for any agent participating in the Multi-Agent Simulation.
    Enforces a standard interface for 'Subject' and 'Counter' type agents.
    """
    
    @property
    def name(self) -> str:
        """The display name of the agent."""
        ...
        
    def generate_reply(self, conversation_history: List[Dict[str, str]]) -> Tuple[str, Any]:
        """
        Generates a reply based on the conversation history.
        
        Args:
            conversation_history: List of message dicts [{'role': 'user'|'assistant', 'content': '...'}]
            
        Returns:
            Tuple[str, Any]: (The response text, Metadata/Debug info)
        """
        ...
        
    def get_current_stance(self) -> float:
        """
        Returns the current stance of the agent as a float in [-1.0, 1.0].
        -1.0 = Strong Con
         0.0 = Neutral
         1.0 = Strong Pro
        """
        ...
