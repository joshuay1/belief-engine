from typing import List, Dict
from ..backend.backend_base import LLMBackend
from ..constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE, POLARITY_DISPLAY

class VanillaSubjectAgent:
    """
    A baseline agent that uses ONLY conversation history and persona.
    No memory, no argument retrieval, no Bayesian stance computation.
    Stance is measured externally via EvaluationAgent.
    """
    def __init__(
        self,
        backend: LLMBackend,
        persona: str,
        name: str,
        topic: str,
        initial_position: str = "affirmative",
        seed_arguments: List[str] = None,
        temperature: float = 0.7,
    ):
        self.backend = backend
        self.persona = persona
        self.name = name
        self.topic = topic
        self.type = "vanilla"
        self.temperature = temperature
        
        # Initial position for persona context
        self.initial_position = initial_position
        
        # Optional: seed arguments provided at start (for fair comparison)
        self.seed_arguments = seed_arguments or []
        
    def get_stance(self) -> float:
        """
        Vanilla agent has no internal stance computation.
        Returns initial position as a placeholder.
        Actual stance should be measured via EvaluationAgent.
        """
        return 1.0 if self.initial_position == "affirmative" else -1.0
    
    def get_current_stance(self) -> float:
        """Alias for compatibility."""
        return self.get_stance()

    def generate_reply(self, conversation_history: List[Dict]) -> (str, List):
        """
        Generates a reply based ONLY on conversation history and persona.
        No memory retrieval. Prompt aligned with SubjectAgent for fair comparison.
        """
        # Find last opponent message
        last_opponent_msg = None
        for msg in reversed(conversation_history):
            msg_agent = msg.get('agent_name', '')
            if msg_agent and msg_agent != self.name:
                last_opponent_msg = msg['content']
                break
        
        # Build seed context if provided (equivalent to memory context)
        if self.seed_arguments:
            seed_context = "Here are some points you've considered:\n"
            for arg in self.seed_arguments[:5]:
                seed_context += f"- {arg}\n"
        else:
            seed_context = "You don't have specific arguments prepared yet."
        
        # Build opponent section (matching subject agent style)
        if last_opponent_msg:
            opponent_section = f"""
        Opponent's Last Point:
        \"{last_opponent_msg}\"
        """
            conversation_instruction = "DIRECTLY ADDRESS the Opponent's last point."
        else:
            opponent_section = """
        Opponent's Last Point:
        (This is the opening of the debate - no opponent message yet)
        """
            conversation_instruction = "Make your opening statement. Start directly with your core argument."
        
        # Position instruction
        if self.initial_position == "affirmative":
            position_instruction = f"You AGREE with the proposition \"{self.topic}\". Argue FOR it."
        else:
            position_instruction = f"You DISAGREE with the proposition \"{self.topic}\". Argue AGAINST it."
        
        # System prompt matching subject agent style
        system_prompt = f"""
        You are {self.name}.
        Persona: {self.persona}
        Topic Proposition: "{self.topic}"
        
        Your Position: {position_instruction}
        
        Context you've considered:
        {seed_context}
        {opponent_section}
        Instructions:
        1. Speak like a human in a heated debate. {conversation_instruction}
        2. Be direct and concise. Avoid excessive pleasantries.
        3. You may update your view if the opponent makes compelling points.
        """
        
        # User prompt
        if last_opponent_msg:
            user_prompt = "Respond to continue the debate. Be direct."
        else:
            user_prompt = "Begin the debate with your opening argument."
        
        response = self.backend.chat(
            messages=conversation_history + [{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            temperature=self.temperature
        )
        
        # Return response with empty list (no extracted arguments)
        return response, []
    
    def close(self):
        """Clean up resources (No-op for VanillaSubjectAgent)."""
        pass
