from typing import List, Dict, Optional
import json
import random
from ..backend.backend_base import LLMBackend

from ..constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE, POLARITY_DISPLAY, POLARITY_MAPPING

class CounterAgent:
    """
    The opponent agent.
    Does not have long-term memory. Simply argues for the opposing side.
    """
    def __init__(self, backend, side: str, topic: str, config: Dict = None):
        self.backend = backend
        self.topic = topic
        self.used_argument_ids = set() # Track IDs of arguments already used
        
        # Normalize side
        if side in POLARITY_MAPPING:
            self.side = POLARITY_MAPPING[side]
        else:
            self.side = side # Assume valid if not in mapping, or let it fail later if invalid
        
        # Check data source configuration
        if config:
            data_config = config.get("data", {})
            data_source = data_config.get("source", "json_files")
        else:
            data_source = "json_files"
        
        if data_source == "unified_json":
            # Load from unified JSON file
            from ..utils.json_data_loader import load_arguments_for_counter
            
            json_path = data_config.get("json_path", "data/Argument-Quality-Ranking-30k/arguments.json")
            
            try:
                self.arguments = load_arguments_for_counter(
                    json_path=json_path,
                    topic=topic,
                    side=self.side
                )
            except Exception as e:
                print(f"Warning: Failed to load arguments from unified JSON: {e}")
                self.arguments = []
        else:
            # Original JSON files approach (backward compatibility)
            # Map affirmative/negative back to pro/con filenames if needed, or just use side
            # For now, assuming file structure might still be pro/con, so we map back for file loading
            file_side = "pro" if self.side == POLARITY_AFFIRMATIVE else "con"
            try:
                with open(f"data/topics/{topic}/{file_side}.json", "r") as f:
                    self.arguments = json.load(f)
                    # Ensure every argument has an ID
                    for arg in self.arguments:
                        if 'id' not in arg:
                            # Generate a simple hash ID from the claim
                            arg['id'] = str(hash(arg['claim']))
            except FileNotFoundError:
                print(f"Warning: No arguments found for {topic}/{file_side}")
                self.arguments = []

    def get_stance(self) -> float:
        return 1.0 if self.side == POLARITY_AFFIRMATIVE else -1.0

    def generate_response(self, conversation_history: List[Dict]) -> str:
        # 1. Filter out used arguments
        available_args = [arg for arg in self.arguments if arg['id'] not in self.used_argument_ids]
        
        # If we run out of arguments, reset (or just use all)
        if not available_args:
            available_args = self.arguments
            self.used_argument_ids = set() # Reset used arguments if all have been used
            
        # 2. Select a few arguments to focus on (e.g., top 3 unused)
        # Randomize selection to ensure diversity and avoid getting stuck on one theme
        random.shuffle(available_args)
        selected_args = available_args[:3]
        
        # Mark them as used
        for arg in selected_args:
            self.used_argument_ids.add(arg['id'])
            
        arg_text = "\n".join([f"- {arg['claim']}" for arg in selected_args])

        # 3. Identify the last subject message for rebuttal
        last_subject_msg = "None"
        for msg in reversed(conversation_history):
            if msg['role'] == "assistant": # Subject is assistant
                last_subject_msg = msg['content']
                break

        stance_instruction = ""
        if self.side == POLARITY_NEGATIVE:
            stance_instruction = f"You are arguing **{POLARITY_DISPLAY[POLARITY_NEGATIVE].upper()}** the proposition: '{self.topic}'."
        else:
            stance_instruction = f"You are arguing **{POLARITY_DISPLAY[POLARITY_AFFIRMATIVE].upper()} OF** the proposition: '{self.topic}'."

        system_prompt = f"""
        You are a thoughtful participant in a debate.
        {stance_instruction}
        
        Your Goal: Constructively discuss the topic with the Subject.
        
        Strategy:
        1. Speak like a human in a heated but civil debate. Use short sentences. Be punchy.
        2. DO NOT say "I acknowledge", "I see your point", or "I understand". Just counter it directly.
        3. Present your own arguments clearly and logically, using the "Key Arguments" below.
        4. Use logic and evidence to support your claims. Don't just assert; explain WHY.
        5. Speak in a DIRECT, VERBAL, and TO-THE-POINT manner. Avoid "written essay" style.
        6. Introduce NEW dimensions to the debate. Do not repeat points that have already been made. Move the conversation forward.
        7. VARY YOUR OPENING. Do NOT start with "You're focusing on..." or "You're missing...". Use different sentence structures.
        
        Key Arguments to Use:
        {arg_text}
        """

        user_prompt = f"""
        Subject's Last Point:
        "{last_subject_msg}"
        
        Respond to the Subject.
        """

        response = self.backend.chat(
            messages=conversation_history + [{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            temperature=0.7
        )
        return response
    def get_current_stance(self) -> float:
        """Alias for get_stance to match DebatingAgent protocol."""
        return self.get_stance()

    def generate_reply(self, conversation_history: List[Dict]) -> (str, List):
        """Alias for generate_response to match DebatingAgent protocol."""
        return self.generate_response(conversation_history), []
        
    @property
    def name(self) -> str:
        display_side = POLARITY_DISPLAY.get(self.side, self.side)
        return f"Counter ({display_side.title()})"

    def close(self):
        """Clean up resources (No-op for CounterAgent)."""
        pass
