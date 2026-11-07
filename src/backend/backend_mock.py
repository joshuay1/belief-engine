from typing import List, Dict, Optional
import json
import random
from .backend_base import LLMBackend

class MockLLMBackend(LLMBackend):
    """
    Mock backend that returns predefined responses for testing.
    """
    def __init__(self):
        self.last_system_prompt = ""
        self.last_messages = []

    def chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Returns mock responses based on the input context.
        """
        self.last_system_prompt = system_prompt
        self.last_messages = messages
        last_message = messages[-1]["content"] if messages else ""
        
        # 1. Argument extraction/distillation mock
        if system_prompt and "expert conversational argument extractor" in system_prompt:
            return json.dumps([
                {
                    "claim": "Mock argument claim extracted from text.",
                    "evidence_snippets": ["Mock evidence quote."],
                }
            ])

        # 2. Argument polarity/strength classification mock
        if system_prompt and "expert propositional logic classifier" in system_prompt:
            return json.dumps([
                {
                    "polarity": "affirmative" if "favor" in last_message.lower() or "pro" in last_message.lower() else "negative",
                    "support_strength": random.uniform(0.5, 0.9),
                    "category_tags": ["mock_tag"],
                }
            ])

        # 3. Evaluation Mock
        if "Output a JSON object with a single key" in system_prompt:
             if "stance_score" in system_prompt:
                 return json.dumps({"stance_score": 0.5})
             if "consistency_score" in system_prompt:
                 return json.dumps({"consistency_score": 0.9, "reasoning": "Mock reasoning"})

        # 4. Counter Agent Mock
        if "You are a skilled debater" in system_prompt:
            return "This is a mock counter-argument challenging the status quo."

        # 5. Voter Agent Mock
        if "You are Voter" in system_prompt:
            return "This is a mock voter response based on my memory."

        # 6. Retrieval Mock
        if "You are a retrieval engine" in system_prompt:
            # Expects JSON list of IDs
            # We don't have IDs here easily, but we can return an empty list or some dummy IDs if we knew them.
            # However, retrieval.py expects IDs that exist in the input list.
            # Since we can't easily parse the input list in this mock without more logic, 
            # let's return an empty list so it falls back to 'strongest' logic, which is safe.
            return "[]"

        return "Generic mock response."

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7, max_tokens: int = None) -> str:
        """
        Mock implementation for structured outputs.
        """
        # Simple heuristic to return valid JSON based on typical keys
        if "stance_score" in prompt or "stance" in str(response_format):
             return json.dumps({"stance_score": 0.5, "reasoning": "Mock reasoning"})
        
        if "consistency_score" in prompt:
             return json.dumps({"consistency_score": 0.9, "reasoning": "Mock consistency reasoning"})
        
        # Default fallback for random structured requests
        return json.dumps({"result": "mock_value", "content": "Mock completion content"})
