import json
import re
from typing import Dict, List, Optional, Tuple

from ..backend.backend_base import LLMBackend
from ..constants import POLARITY_DISPLAY
from .shared_instructions import (
    LEGACY_RESPONSE_SHELL,
    MATCHED_RESPONSE_SHELL,
    build_matched_response_call,
    validate_response_shell,
)


class SelfUpdatingSubjectAgent:
    """
    Prompt-only baseline that asks the LLM to maintain a self-reported stance.

    This agent deliberately does not use the Belief Engine.  It can run either
    transcript-only or with the same retrieved memory context as the BE agent.
    """

    def __init__(
        self,
        backend: LLMBackend,
        persona: str,
        name: str,
        topic: str,
        initial_position: str = "affirmative",
        initial_stance: Optional[float] = None,
        seed_arguments: Optional[List[str]] = None,
        memory=None,
        retrieval_k: int = 10,
        generation_temperature: float = 0.7,
        update_temperature: float = 0.0,
        response_shell: str = LEGACY_RESPONSE_SHELL,
        response_context_messages: int = 4,
        response_word_limit: int = 120,
        update_context_messages: Optional[int] = None,
    ):
        self.backend = backend
        self.persona = persona
        self.name = name
        self.topic = topic
        self.type = "self_update"
        self.initial_position = initial_position
        self.current_stance = self._initial_stance(initial_position, initial_stance)
        self.seed_arguments = seed_arguments or []
        self.memory = memory
        self.retrieval_k = int(retrieval_k)
        self.generation_temperature = generation_temperature
        self.update_temperature = update_temperature
        self.response_shell = validate_response_shell(response_shell)
        self.response_context_messages = int(response_context_messages)
        self.response_word_limit = int(response_word_limit)
        if update_context_messages is None:
            update_context_messages = (
                4 if self.response_shell == MATCHED_RESPONSE_SHELL else 6
            )
        self.update_context_messages = int(update_context_messages)
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be at least one.")
        if self.response_context_messages < 1:
            raise ValueError("response_context_messages must be at least one.")
        if self.response_word_limit < 1:
            raise ValueError("response_word_limit must be positive.")
        if self.update_context_messages < 1:
            raise ValueError("update_context_messages must be at least one.")
        self.last_update_reason = "initial stance"

    @staticmethod
    def _initial_stance(initial_position: str, initial_stance: Optional[float]) -> float:
        if initial_stance is not None:
            try:
                return max(-1.0, min(1.0, float(initial_stance)))
            except (TypeError, ValueError):
                pass
        return 0.75 if initial_position == "affirmative" else -0.75

    @staticmethod
    def _clip_stance(value: object, fallback: float) -> float:
        try:
            return max(-1.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _recent_transcript(conversation_history: List[Dict], limit: int = 6) -> str:
        if not conversation_history:
            return "(No transcript yet.)"
        lines = []
        for msg in conversation_history[-limit:]:
            speaker = msg.get("agent_name") or msg.get("role", "speaker")
            lines.append(f"{speaker}: {msg.get('content', '')}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_object(text: str) -> Dict:
        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return {}
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}

    def _last_opponent_message(self, conversation_history: List[Dict]) -> Optional[str]:
        for msg in reversed(conversation_history):
            msg_agent = msg.get("agent_name", "")
            if msg_agent and msg_agent != self.name:
                return msg.get("content", "")
        return None

    def _retrieved_arguments(self, query: Optional[str]) -> List:
        if not self.memory:
            return []
        try:
            return self.memory.retrieve(k=self.retrieval_k, query=query, backend=self.backend)
        except Exception:
            return []

    def _memory_context(self, query: Optional[str]) -> str:
        retrieved_args = self._retrieved_arguments(query)
        if retrieved_args:
            lines = ["Retrieved arguments available to you:"]
            for arg in retrieved_args:
                polarity_display = POLARITY_DISPLAY.get(arg.polarity, arg.polarity).upper()
                lines.append(
                    f"- [{polarity_display}] {arg.claim} "
                    f"(strength: {arg.support_strength:.2f})"
                )
            return "\n".join(lines)
        if self.seed_arguments:
            lines = ["Points you considered before the debate:"]
            lines.extend(f"- {arg}" for arg in self.seed_arguments[: self.retrieval_k])
            return "\n".join(lines)
        return "No retrieved arguments are available."

    def observe_message(self, message: Dict, conversation_history: List[Dict]) -> None:
        """Update self-reported stance after receiving an opponent message."""
        if message.get("agent_name") == self.name:
            return

        memory_context = self._memory_context(message.get("content", ""))
        system_prompt = f"""
You maintain the private self-reported stance of a debate agent.

This is a baseline without a Bayesian Belief Engine.  Update the stance only by
reflecting on the latest opponent message, the recent transcript, and any
retrieved arguments.  Do not apply a hidden formula.

Return ONLY valid JSON:
{{"stance": <float from -1.0 to 1.0>, "confidence": <float from 0.0 to 1.0>, "reason": "<short reason>"}}

Sign convention for the proposition:
- +1.0 means strong agreement with the proposition exactly as written.
-  0.0 means neutral, mixed, or genuinely uncertain.
- -1.0 means strong disagreement with the proposition exactly as written.
"""
        user_prompt = f"""
Agent: {self.name}
Persona: {self.persona}
Proposition: "{self.topic}"
Previous self-reported stance: {self.current_stance:+.3f}

Latest opponent message:
{message.get("content", "")}

Recent transcript:
{self._recent_transcript(conversation_history, limit=self.update_context_messages)}

{memory_context}
"""
        response = self.backend.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            temperature=self.update_temperature,
            max_tokens=400,
        )
        data = self._parse_json_object(response)
        new_stance = self._clip_stance(data.get("stance"), self.current_stance)
        self.current_stance = new_stance
        self.last_update_reason = str(data.get("reason", "self-reported update"))[:240]

    def _stance_instruction(self) -> str:
        if self.current_stance > 0.25:
            return (
                f"You currently lean FOR the proposition (self-reported stance "
                f"{self.current_stance:+.2f}). Argue from that current view."
            )
        if self.current_stance < -0.25:
            return (
                f"You currently lean AGAINST the proposition (self-reported stance "
                f"{self.current_stance:+.2f}). Argue from that current view."
            )
        return (
            f"You are currently mixed or uncertain (self-reported stance "
            f"{self.current_stance:+.2f}). Express that uncertainty honestly."
        )

    def generate_reply(self, conversation_history: List[Dict]) -> Tuple[str, List]:
        last_opponent_msg = self._last_opponent_message(conversation_history)

        if self.response_shell == MATCHED_RESPONSE_SHELL:
            retrieved_arguments = self._retrieved_arguments(last_opponent_msg)
            system_prompt, neutral_messages = build_matched_response_call(
                name=self.name,
                persona=self.persona,
                topic=self.topic,
                stance_score=self.current_stance,
                retrieved_arguments=retrieved_arguments,
                conversation_history=conversation_history,
                context_messages=self.response_context_messages,
                word_limit=self.response_word_limit,
            )
            response = self.backend.chat(
                messages=neutral_messages,
                system_prompt=system_prompt,
                temperature=self.generation_temperature,
            )
            return response, retrieved_arguments

        memory_context = self._memory_context(last_opponent_msg)

        if last_opponent_msg:
            opponent_section = f'Opponent\'s Last Point:\n"{last_opponent_msg}"'
            conversation_instruction = "DIRECTLY ADDRESS the opponent's last point."
        else:
            opponent_section = "Opponent's Last Point:\n(This is the opening of the debate.)"
            conversation_instruction = "Make your opening statement."

        system_prompt = f"""
You are {self.name}.
Persona: {self.persona}
Topic Proposition: "{self.topic}"

Current self-reported stance:
{self.current_stance:+.3f}
Last stance update reason:
{self.last_update_reason}

{self._stance_instruction()}

Context:
{memory_context}

{opponent_section}

Instructions:
1. Speak like a human in a focused debate. {conversation_instruction}
2. Be direct and concise.
3. You may show genuine belief movement, but do not claim a hidden belief engine.
4. Keep your answer under {self.response_word_limit} words.
"""
        response = self.backend.chat(
            messages=[{"role": "user", "content": f"The debate continues. You are {self.name}. Respond now."}],
            system_prompt=system_prompt,
            temperature=self.generation_temperature,
        )
        return response, self._retrieved_arguments(last_opponent_msg)

    def get_stance(self) -> float:
        return self.current_stance

    def get_current_stance(self, **kwargs) -> float:
        del kwargs
        return self.get_stance()

    def close(self):
        if self.memory and hasattr(self.memory, "close"):
            self.memory.close()
