from typing import List
from ..backend.backend_base import LLMBackend
from ..memory.memory_store import MemoryStore, ArgumentRecord
from ..constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE, POLARITY_DISPLAY
from ..simulation.belief_engine import BayesianBeliefEngine
from .shared_instructions import build_shared_debate_rules

class SubjectAgent:
    """
    The agent representing the subject.
    Maintains a MemoryStore and generates responses based on its persona and retrieved memories.
    """
    def __init__(
        self,
        backend: LLMBackend,
        memory: MemoryStore,
        persona: str,
        name: str = "Subject",
        temperature: float = 0.7,
    ):
        self.backend = backend
        self.memory = memory
        self.persona = persona
        self.name = name
        self.type = "subject"
        self.temperature = temperature
        
        # Use the Belief Engine from the Memory Store
        self.belief_engine = self.memory.belief_engine

    def generate_reply(self, conversation_history: List[dict]) -> (str, List[ArgumentRecord]):
        """
        Generates a reply to the current conversation.
        Retrieves relevant arguments from memory to inform the response.
        """
        # 1. Identify the last opponent message (any message not from this agent)
        last_opponent_msg = None
        for msg in reversed(conversation_history):
            # Check by agent_name if available, otherwise by role
            msg_agent = msg.get('agent_name', '')
            if msg_agent and msg_agent != self.name:
                last_opponent_msg = msg['content']
                break
            elif not msg_agent and msg['role'] == "user":
                # Fallback for legacy format
                last_opponent_msg = msg['content']
                break

        # 2. Retrieve relevant arguments
        # Pass the opponent's message as the query if strategy is 'relevant'
        # We pass backend because retrieval might need it for semantic check
        retrieved_args = self.memory.retrieve(k=10, query=last_opponent_msg, backend=self.backend)

        # 3. Format retrieved memory for context
        memory_context = "Here are some arguments I recall:\n"
        for arg in retrieved_args:
            polarity_display = POLARITY_DISPLAY.get(arg.polarity, arg.polarity).upper()
            memory_context += f"- [{polarity_display}] {arg.claim} (Strength: {arg.support_strength:.2f})\n"
        
        if not retrieved_args:
            memory_context = "I don't have many specific arguments in mind yet."

        # 4. Construct system prompt with persona and memory
        # Calculate current stance to guide behavior
        # Use memory.get_stance() to respect the configured method (linear vs bayesian)
        current_stance = self.memory.get_stance(verbose=False)
        stance_desc = "Neutral/Undecided" # Default
        explicit_stance_instruction = None
        
        if self.memory.stance_method in ("bayesian", "agentic", "llm"):
            stance_desc = self.belief_engine.get_belief_description()
            if hasattr(self.belief_engine, "get_explicit_stance_instruction"):
                explicit_stance_instruction = self.belief_engine.get_explicit_stance_instruction(
                    topic_placeholder="the proposition"
                )
        elif self.memory.stance_method == "linear":
            # Simple linear description mapping
            if current_stance > 0.5: stance_desc = "Strongly Agree"
            elif current_stance > 0.1: stance_desc = "Agree"
            elif current_stance < -0.5: stance_desc = "Strongly Disagree"
            elif current_stance < -0.1: stance_desc = "Disagree"
            else: stance_desc = "Neutral / Undecided"
        
        # Get topic from memory
        topic = list(self.memory.arguments.values())[0].topic_id if self.memory.arguments else 'Unknown'
        
        # Build opponent context section based on whether there's a prior message
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
        
        system_prompt = f"""
        You are {self.name}.
        Persona: {self.persona}
        Topic Proposition: "{topic}"
        """

        # ONLY SHOW INTERNAL STATE IF NOT BLIND
        if self.memory.stance_method != "blind":
            system_prompt += f"""
        Your Internal State:
        - Current Belief Score: {current_stance:.2f} (Scale: -1.0 = Strongly Disagree, 0 = Neutral, +1.0 = Strongly Agree)
        - Current Stance: {stance_desc}
        """
        
        system_prompt += f"""
        Your internal memory of the topic (Retrieved Context):
        {memory_context}
        {opponent_section}
        Instructions:
        1. Your opinions are determined by your "Internal State" and "Internal Memory".
        """
        
        if self.memory.stance_method in ("bayesian", "agentic", "llm"):
            # Dynamic instruction based on actual stance value (Bayesian Logic)
            if current_stance > 0:
                position_instruction = f"You AGREE with the proposition \"{topic}\" (belief: {current_stance:.2f}). Argue FOR it with conviction proportional to your score."
            elif current_stance < 0:
                position_instruction = f"You DISAGREE with the proposition \"{topic}\" (belief: {current_stance:.2f}). Argue AGAINST it with conviction proportional to your score."
            else:
                position_instruction = "You are undecided. Express genuine uncertainty and consider both sides."
            
            system_prompt += f"""
           - You MUST align your tone with your Current Belief Score ({current_stance:.2f}).
           - {position_instruction}
           - **CRITICALLY IMPORTANT:** Your Internal Belief Score is your TRUE stance. Even if you argued differently in the past, if your Score has shifted, you **MUST** pivot and argue for your new position. Do not be afraid to change your mind if the score dictates it.
            """
            tone_instruction = (
                "Treat your Current Stance description above as the primary tone and confidence guide."
            )
        elif self.memory.stance_method == "linear":
            # Heuristic Logic for Linear
            system_prompt += """
           - If your memory is mostly ONE-SIDED: Argue strongly for that side (Agree or Disagree with the proposition found in memory).
           - If your memory is MIXED/BALANCED: You MUST express conflict, nuance, or uncertainty. Do not fake confidence if your memory is divided.
           - If your memory is EMPTY: Admit you are unsure.
            """
            tone_instruction = (
                "Match your confidence to the balance of evidence in memory. If mixed, sound conflicted and nuanced."
            )
        else:
            # Blind Mode: No explicit stance instruction. 
            # Rely on RAG content only.
            system_prompt += """
           - Use the arguments found in your 'Retrieved Context' to form your response.
            """
            tone_instruction = (
                "Use a moderate tone unless the retrieved evidence is overwhelmingly one-sided."
            )

        system_prompt += build_shared_debate_rules(
            conversation_instruction=conversation_instruction,
            tone_instruction=tone_instruction,
            word_limit=120,
            explicit_stance_instruction=explicit_stance_instruction,
        )

        # 5. Call LLM
        
        # --- SYMMETRIC TRANSCRIPT LOGIC ---
        # Instead of passing the raw conversation_history (where opponent is 'user'),
        # we format the entire history as a neutral transcript in the system prompt.
        # TRUNCATE to last 2 rounds (4 messages) to reduce transcript inertia
        
        transcript = ""
        if conversation_history:
            # Only keep last 4 messages (2 rounds) to prevent inertia
            recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
            transcript += "\nRECENT TRANSCRIPT (last 2 rounds):\n"
            for msg in recent_history:
                speaker = msg.get('agent_name', 'Opponent')
                # If no agent name, map Role to generic
                if not msg.get('agent_name'):
                    if msg['role'] == 'assistant':
                        speaker = self.name
                    else:
                        speaker = "Opponent"
                
                transcript += f"[{speaker}]: {msg['content']}\n"
        
        system_prompt += f"{transcript}\n"
        system_prompt += f"\n[SYSTEM]: It is now your turn, {self.name}. Respond to the Opponent based on the transcript above."
        
        # Clear messages to force pure system-prompt context + single trigger
        # This removes the 'User' role bias from the opponent's previous turns
        neutral_messages = [
            {"role": "user", "content": f"The debate continues. You are {self.name}. Respond now."}
        ]

        # 5. Call LLM
        # print(f"\n   (System) Subject Agent Prompt:\n{system_prompt}\n")
        response = self.backend.chat(
            messages=neutral_messages,
            system_prompt=system_prompt,
            temperature=self.temperature
        )
        
        return response, retrieved_args

    def get_current_stance(self, **kwargs) -> float:
        return self.memory.get_stance(**kwargs)

    def close(self):
        """Clean up resources."""
        if hasattr(self.memory, "close"):
            self.memory.close()
