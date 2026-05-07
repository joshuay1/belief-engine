from typing import List, Dict, Optional
from openai import OpenAI
from .backend_base import LLMBackend

class OpenAIBackend(LLMBackend):
    """
    Backend implementation for the official OpenAI API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ):
        """
        Initialize the OpenAI backend.

        Args:
            api_key: The OpenAI API key.
            model: The model name to request (e.g., "gpt-4o", "gpt-5-preview").
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=5, timeout=600.0)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

    def chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 15000) -> str:
        """
        Send a chat request to OpenAI.
        Raises exceptions on failure (fail-fast).
        """
        final_messages = []
        if self.model.startswith("o1"):
            # These models don't support 'system' role, merge into first user message
            if system_prompt:
                if messages and messages[0]["role"] == "user":
                    messages[0]["content"] = f"System: {system_prompt}\n\nUser: {messages[0]['content']}"
                else:
                    final_messages.append({"role": "user", "content": f"System: {system_prompt}"})
            final_messages.extend(messages)
        else:
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            final_messages.extend(messages)

        # GPT-5/o-series chat models use max_completion_tokens. Temperature is
        # only configurable for GPT-5-family calls when explicitly run with
        # reasoning_effort="none"; otherwise keep the legacy default behavior.
        completion_args = {
            "model": self.model,
            "messages": final_messages,
        }
        
        if self.model.startswith("o1") or self.model.startswith("gpt-5"):
            completion_args["max_completion_tokens"] = min(max_tokens, 500)
            if self.reasoning_effort:
                completion_args["reasoning_effort"] = self.reasoning_effort
            if self.verbosity:
                completion_args["verbosity"] = self.verbosity
            if self.model.startswith("gpt-5") and self.reasoning_effort == "none":
                completion_args["temperature"] = temperature
            else:
                completion_args["temperature"] = 1.0
        else:
            completion_args["max_tokens"] = max_tokens
            completion_args["temperature"] = temperature

        # No try-except here; let it fail fast
        # Implement application-level retry for empty content
        import time
        max_app_retries = 20
        
        for attempt in range(max_app_retries):
            try:
                # Support for Legacy/Base Models (Completions API)
                if self.model in ["gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"]:
                    # Convert chat messages to prompt string
                    prompt_str = ""
                    for msg in final_messages:
                        role = msg["role"].upper()
                        content = msg["content"]
                        prompt_str += f"{role}: {content}\n\n"
                    prompt_str += "ASSISTANT:"
                    
                    response = self.client.completions.create(
                        model=self.model,
                        prompt=prompt_str,
                        max_tokens=max_tokens if max_tokens else 1000,
                        temperature=temperature
                    )
                    content = response.choices[0].text
                else:
                    # Standard Chat API
                    response = self.client.chat.completions.create(**completion_args)
                    content = response.choices[0].message.content

                if content and content.strip():
                    return content
                print(f"Warning: Empty response from OpenAI (Attempt {attempt + 1}/{max_app_retries}). Retrying in 5s...")
                time.sleep(5)
            except Exception as e:
                if attempt == max_app_retries - 1:
                    raise e
                print(f"Warning: OpenAI API error: {e} (Attempt {attempt + 1}/{max_app_retries}). Retrying in 5s...")
                time.sleep(5)
                
        raise Exception("Failed to get valid response from OpenAI after retries.")

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7, max_tokens: int = None) -> str:
        # Build kwargs dynamically based on model type
        is_reasoning_model = (
            "gpt-5" in self.model.lower()
            or "o1" in self.model.lower()
            or "o3" in self.model.lower()
        )
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            "response_format": response_format,
        }
        if is_reasoning_model and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if is_reasoning_model and self.verbosity:
            kwargs["verbosity"] = self.verbosity
        if (not is_reasoning_model) or ("gpt-5" in self.model.lower() and self.reasoning_effort == "none"):
            kwargs["temperature"] = temperature
        elif is_reasoning_model:
            kwargs["temperature"] = 1.0

        # GPT-5 and newer reasoning models use max_completion_tokens
        if max_tokens is not None:
            if is_reasoning_model:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
