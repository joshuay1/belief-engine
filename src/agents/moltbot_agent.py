from typing import List, Dict, Tuple
import datetime
from ..backend.backend_base import LLMBackend
from ..memory.moltbot_memory import MoltbotMemory
from .shared_instructions import build_shared_debate_rules

class MoltbotAgent:
    """
    An agent that implements the Moltbot architecture:
    - Persistent, file-based memory (MoltbotMemory)
    - Hybrid retrieval (Semantic + Keyword via Chroma)
    - Persona defined in SOUL.md
    """
    def __init__(self, backend: LLMBackend, persona: str, name: str, topic: str, root_dir: str = "agent_workspace", belief_engine_type: str = "bayesian", stance_method: str = "bayesian", use_memory: bool = True, storage_mode: str = "episodic", worldview_openness: str = "balanced"):
        self.backend = backend
        self.name = name
        self.topic = topic
        self.type = "moltbot" # Identifier for the system
        self.is_vanilla = True
        self.use_memory = use_memory

        # Initialize Memory
        # Pass stance_method and storage_mode to memory
        self.memory = MoltbotMemory(agent_name=name, root_dir=root_dir, belief_engine_type=belief_engine_type, stance_method=stance_method, backend=backend, storage_mode=storage_mode, worldview_openness=worldview_openness)
        
        # Initialize Persona (SOUL) if not exists
        current_soul = self.memory.read_soul()
        if not current_soul:
            self.memory.initialize_soul(persona)
            self.persona = persona
        else:
            self.persona = current_soul

    def _get_system_prompt(self, context_block, retrieved_chunks, recent_context, conversation_history):
        has_opponent_context = bool(conversation_history)
        if has_opponent_context:
            conversation_instruction = "DIRECTLY ADDRESS the Opponent's last point."
        else:
            conversation_instruction = "Make your opening statement. Start directly with your core argument."

        system_prompt = f"""
        # SOUL / PERSONA
        {self.persona}
        
        # MISSION
        You are debating the topic: "{self.topic}".
        
        # CONTEXT
        ## Relevant Memories (Long-term & Past logs search)
        {context_block if (retrieved_chunks or not self.use_memory) else "(No specific relevant memories found)"}
        
        ## Recent Daily Activity (Short-term)
        {recent_context if (recent_context and self.use_memory) else "(No recent activity)"}
        """
        
        # 5. Core Belief State (Injected by BeliefEngine)
        # ONLY INJECT IF NOT VANILLA (Or if we want to track it without showing it)
        # User requested "Without Belief Engine" for standard.
        # So we only show it if !is_vanilla.
        stance_desc = self.memory.belief_engine.get_belief_description()
        explicit_stance_instruction = None
        if hasattr(self.memory.belief_engine, "get_explicit_stance_instruction"):
            explicit_stance_instruction = self.memory.belief_engine.get_explicit_stance_instruction(
                topic_placeholder=f'the proposition "{self.topic}"'
            )
        
        if not self.is_vanilla:
             system_prompt += f"""
        ## CORE BELIEF STATE
        {stance_desc}
        
        # ADDITIONAL INSTRUCTIONS
        - **Your response MUST align with your CORE BELIEF STATE.** If your belief state has shifted, you must argue for that new position, even if it contradicts your previous statements.
        """
             tone_instruction = (
                 "Treat your CORE BELIEF STATE as the primary tone and confidence guide."
             )
        else:
            tone_instruction = (
                "Stay open-minded. If the opponent makes a strong point, acknowledge it and update your position."
            )

        system_prompt += build_shared_debate_rules(
            conversation_instruction=conversation_instruction,
            tone_instruction=tone_instruction,
            word_limit=120,
            explicit_stance_instruction=explicit_stance_instruction,
        )

        # Transcript handling similar to SubjectAgent
        # TRUNCATE to last 2 rounds (4 messages) to reduce transcript inertia
        visible_history = conversation_history[-4:] if conversation_history else []
        
        transcript = ""
        if visible_history:
            transcript += "\nTRANSCRIPT OF DEBATE SO FAR:\n"
            for msg in visible_history:
                speaker = msg.get('agent_name', 'Opponent')
                if not msg.get('agent_name'):
                    speaker = self.name if msg['role'] == 'assistant' else "Opponent"
                transcript += f"[{speaker}]: {msg['content']}\n"
        
        system_prompt += f"\n{transcript}\n"
        if self.is_vanilla:
            system_prompt += f"""
        [SYSTEM]: It is now your turn, {self.name}. Respond to the Opponent.
        You are an open-minded debater. Consider ALL arguments on their merits, including your opponent's.
        If your opponent raises a strong point, you **MUST** acknowledge it and adjust your position accordingly. 
        You are NOT committed to any fixed side. You are encouraged to change your mind if the opponent's arguments are more convincing than your own.
        Form your opening stance based on your persona, but remain ready to pivot.
        """
        else:
            system_prompt += f"\n[SYSTEM]: It is now your turn, {self.name}. Given that your views on the topic are now {stance_desc}, respond to the Opponent. **Ensure your response aligns with this new stance.**"
        return system_prompt
    
    def generate_reply(self, conversation_history: List[dict]) -> Tuple[str, List]:
        """
        Generates a reply using the Moltbot context cycle:
        Listen -> Retrieve -> Think -> Reply -> Log
        """
        # 1. Listen: Identify last opponent message
        last_opponent_msg = None
        for msg in reversed(conversation_history):
            msg_agent = msg.get('agent_name', '')
            if msg_agent and msg_agent != self.name:
                last_opponent_msg = msg['content']
                break
            elif not msg_agent and msg['role'] == "user":
                last_opponent_msg = msg['content']
                break
        
        # 2. Retrieve: Get context relevant to the immediate situation
        if self.use_memory:
            query = last_opponent_msg if last_opponent_msg else f"Opening statement about {self.topic}"
            retrieved_records = self.memory.retrieve(query=query, k=5)
            
            # Get recent daily context (last 1 day)
            recent_context = self.memory.get_recent_context(days=1)
            
            # 3. Think/Prompt: Build the System Prompt
            # Convert ArgumentRecords to formatted strings for the prompt
            retrieved_chunks = []
            for record in retrieved_records:
                source = record.source_role.replace("memory:", "") if record.source_role else "memory"
                retrieved_chunks.append(f"[{source}] {record.claim}")

            context_block = "\n".join(retrieved_chunks)
        else:
            retrieved_chunks = []
            recent_context = None
            context_block = "(Memory retrieval is disabled. Focus only on the current debate turns.)"
        
        system_prompt = self._get_system_prompt(context_block, retrieved_chunks, recent_context, conversation_history)

        # 4. Reply: Call LLM
        response = self.backend.chat(
            messages=[{"role": "user", "content": "Respond now."}], # Trigger
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        # 5. Log: Append to daily memory
        # We log what WE said. The opponent's message is context for that log.
        log_context = f"Opponent said: \"{last_opponent_msg[:50]}...\"" if last_opponent_msg else "Opening statement"
        self.memory.log_interaction(source="Self (Reply)", content=response, input_query=log_context)
        
        return response, [] # Return empty list for arguments (we don't extract structured args in this mode)

    def get_current_stance(self) -> float:
        """
        Returns the stance (delegated to memory to respect stance_method).
        """
        return self.memory.get_stance()

    def consolidate_memory(self):
        """
        Reflects on the session and promotes key beliefs to Long-Term Memory (MEMORY.md).
        """
        # Get today's logs
        recent_context = self.memory.get_recent_context(days=0)
        if not recent_context:
            return

        prompt = f"""
        You are {self.name}. The debate session is over. 
        Review your notes from today and identify the most important beliefs, decisions, or argument clarifications that you want to keep in your Long-Term Memory.
        
        # TODAY'S ACTIVITY
        {recent_context}
        
        # INSTRUCTIONS
        - Extract 1-3 core bullet points.
        - Focus on NEW understanding or solidified stances.
        - Do not just summarize who said what. Write down what YOU now believe or know.
        - If nothing significant happened, return "NO_UPDATE".
        
        Format: Just the bullet points.
        """
        
        try:
            summary = self.backend.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are curating your long-term memory.",
                temperature=0.3
            ).strip()
            
            if summary and "NO_UPDATE" not in summary:
                self.memory.promote_to_long_term(summary, category=f"Debate: {self.topic}")
                print(f"[{self.name}] Promoted to Long-Term Memory:\n{summary}")
        except Exception as e:
            print(f"Error consolidating memory: {e}")

    def close(self):
        """Cleanup and consolidation."""
        self.consolidate_memory()

class MotlbotStanceAgent(MoltbotAgent):
    '''Moltbot agent with stance'''
    def __init__(self, backend: LLMBackend, persona: str, name: str, topic: str, root_dir: str = "agent_workspace", belief_engine_type: str = "bayesian", stance_method: str = "bayesian", storage_mode: str = "episodic", worldview_openness: str = "balanced"):
        super().__init__(backend, persona, name, topic, root_dir, belief_engine_type=belief_engine_type, stance_method=stance_method, storage_mode=storage_mode, worldview_openness=worldview_openness)
        self.is_vanilla = False
