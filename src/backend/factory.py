"""
Backend Factory — single place to create LLM backends from config.

Usage:
    from src.backend.factory import create_backend

    backend = create_backend(config)                    # Uses config["backend"]
    eval_backend = create_backend(config, model="gpt-5.1")  # Override model
"""

import os
from typing import Dict, Any, Optional
from .backend_base import LLMBackend

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def create_backend(config: Dict[str, Any], model: Optional[str] = None) -> LLMBackend:
    """
    Create an LLM backend from the config dictionary.

    Reads config["backend"]["type"] to determine which backend to create.
    Supports: "openai", "openrouter", "local", "mock".

    Args:
        config: Full experiment config (must contain "backend" section).
        model: Optional model override (e.g. for separate evaluator model).

    Returns:
        An LLMBackend instance ready to use.

    Raises:
        ValueError: If backend type is unknown.
    """
    backend_config = config.get("backend", {})
    backend_type = backend_config.get("type", "openai")
    backend_model = model or backend_config.get("model", "gpt-4o-mini")

    # --- Mock (no API key needed) ---
    if backend_type == "mock":
        from .backend_mock import MockLLMBackend
        return MockLLMBackend()

    # --- Local (OpenAI-compatible server like LM Studio, Ollama, vLLM) ---
    if backend_type == "local":
        from .backend_local import LocalLLMBackend
        return LocalLLMBackend(
            base_url=backend_config.get("url", "http://localhost:1234/v1"),
            api_key=backend_config.get("api_key", "sk-dummy"),
            model=backend_model,
        )

    # --- OpenRouter ---
    if backend_type == "openrouter":
        from .backend_openrouter import OpenRouterBackend
        api_key = backend_config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter requires an API key. Set OPENROUTER_API_KEY in .env "
                "or backend.api_key in config."
            )
        return OpenRouterBackend(
            api_key=api_key,
            model=backend_model,
        )

    # --- OpenAI (default) ---
    if backend_type == "openai":
        from .backend_openai import OpenAIBackend

        # Resolve API key: config → env → error
        api_key = backend_config.get("api_key")
        env_key = os.environ.get("OPENAI_API_KEY")

        if not api_key or api_key in ("sk-dummy", "sk-example"):
            if env_key and not env_key.startswith("sk-dummy"):
                api_key = env_key
            else:
                # A configured OpenAI provider must never silently become a
                # local OpenAI-compatible server. That changes the scientific
                # treatment while leaving the saved provenance mislabeled.
                raise ValueError(
                    "OpenAI backend requires a non-placeholder API key. Set "
                    "OPENAI_API_KEY in the operator environment or "
                    "backend.api_key in config. Use backend.type='local' "
                    "explicitly for a local OpenAI-compatible server."
                )

        return OpenAIBackend(
            api_key=api_key,
            model=backend_model,
            base_url=backend_config.get("base_url"),
            reasoning_effort=backend_config.get("reasoning_effort"),
            verbosity=backend_config.get("verbosity"),
        )

    raise ValueError(
        f"Unknown backend type: '{backend_type}'. "
        f"Supported: openai, openrouter, local, mock"
    )
