from typing import List, Dict, Optional
from openai import OpenAI
from .backend_openai import OpenAIBackend

class LocalLLMBackend(OpenAIBackend):
    """
    Backend implementation for a local LLM server (OpenAI compatible).
    Inherits from OpenAIBackend to get robust retry and get_completion support.
    """

    def __init__(self, base_url: str, api_key: str = "sk-dummy", model: str = "local-model"):
        """
        Initialize the local backend.

        Args:
            base_url: The base URL of the local server (e.g., "http://localhost:8000/v1").
            api_key: The API key (usually dummy for local servers).
            model: The model name to request.
        """
        super().__init__(api_key=api_key, model=model, base_url=base_url)
