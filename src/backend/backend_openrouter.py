from typing import List, Dict, Optional
from openai import OpenAI
from .backend_base import LLMBackend

class OpenRouterBackend(LLMBackend):
    """
    Backend implementation for OpenRouter API.
    """

    def __init__(self, api_key: str, model: str, site_url: str = "", site_name: str = ""):
        """
        Initialize the OpenRouter backend.

        Args:
            api_key: The OpenRouter API key.
            model: The model name to request (e.g., "meta-llama/llama-3-8b-instruct").
            site_url: Optional site URL for OpenRouter rankings.
            site_name: Optional site name for OpenRouter rankings.
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            }
        )
        self.model = model

    def chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Send a chat request to OpenRouter.
        """
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        
        final_messages.extend(messages)

        import time
        max_retries = 20
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content
                print(f"Warning: Empty response from OpenRouter (Attempt {attempt + 1}/{max_retries}). Retrying in 5s...")
                time.sleep(5)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Warning: OpenRouter API error: {e} (Attempt {attempt + 1}/{max_retries}). Retrying in 5s...")
                time.sleep(5)
        raise Exception("Failed to get valid response from OpenRouter after retries.")

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7, max_tokens: int = None) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            "response_format": response_format,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
