from typing import Protocol, List, Any, Optional, Dict
from typing import Any
from .argument_record import ArgumentRecord

class MemorySystem(Protocol):
    """
    Protocol defining the standard interface for all memory backends.
    Ensures compatibility across different agents (SubjectAgent, MoltbotAgent).
    """
    belief_engine: Any

    def add_argument(self, argument: Any, backend=None) -> Any:
        """
        Ingest new information (ArgumentRecord or text).
        Returns a status/id or tuple depending on implementation (to be standardized further).
        """
        ...

    def add_arguments_batch(self, arguments: List[Any], backend=None) -> Any:
        """
        Optional fast-path for ingesting a turn's extracted arguments together.
        Implementations may fall back to repeated add_argument calls.
        """
        ...

    def retrieve(self, query: str, k: int = 5) -> List[ArgumentRecord]:
        """
        Retrieve relevant memories.
        MUST return a list of ArgumentRecord objects (or subclasses).
        """
        ...
