from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.
    All agents should use this interface to communicate with the LLM.
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Send a chat request to the LLM.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
            system_prompt: Optional system prompt to prepend or set as system message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The content of the assistant's response as a string.
        """
        pass

    @abstractmethod
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7, max_tokens: int = None) -> str:
        raise NotImplementedError
