from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..constants import POLARITY_DISPLAY
from ..simulation.belief_engine import _select_instruction_bin, load_belief_instructions


LEGACY_RESPONSE_SHELL = "legacy"
MATCHED_RESPONSE_SHELL = "matched_v1"
SUPPORTED_RESPONSE_SHELLS = frozenset(
    {
        LEGACY_RESPONSE_SHELL,
        MATCHED_RESPONSE_SHELL,
    }
)


def build_shared_debate_rules(
    conversation_instruction: str,
    tone_instruction: str,
    word_limit: int = 120,
    stance_examples: Optional[str] = None,
    explicit_stance_instruction: Optional[str] = None,
) -> str:
    """
    Shared response rules for debate agents to keep style/fairness consistent.
    """
    examples = stance_examples or '"I support...", "I oppose...", or "I\'m uncertain..."'
    return f"""
    Shared Debate Rules:
    1. Speak like a human in a heated debate. {conversation_instruction}
    2. In your first sentence, make your current stance explicit (e.g., {examples}).
    3. Granularity requirement for that first sentence: {explicit_stance_instruction or "Match wording strength to your current confidence level."}
    4. BE DIRECT AND STRAIGHT TO THE POINT. State your claim immediately.
    5. AVOID FILLER WORDS. Do not use phrases like "I appreciate your point", "I see where you're coming from", "In conclusion", or "To summarize".
    6. Directly engage the opponent's point, then pivot to your strongest supporting arguments.
    7. **CRITICAL**: CHECK THE TRANSCRIPT. Do NOT repeat arguments you have already made. If your retrieved memory contains points you just said, SKIP THEM and introduce a new angle or example.
    8. Tone guidance: {tone_instruction}
    9. Keep your response CONCISE (under {word_limit} words).
    """


def validate_response_shell(response_shell: str) -> str:
    """Validate and normalize the versioned response-shell selector."""
    normalized = str(response_shell or LEGACY_RESPONSE_SHELL).strip().lower()
    if normalized not in SUPPORTED_RESPONSE_SHELLS:
        supported = ", ".join(sorted(SUPPORTED_RESPONSE_SHELLS))
        raise ValueError(
            f"Unknown response shell {response_shell!r}. Supported values: {supported}."
        )
    return normalized


def find_last_opponent_message(
    conversation_history: Sequence[Mapping[str, Any]],
    agent_name: str,
) -> Optional[str]:
    """Return the latest message from another named agent."""
    for message in reversed(conversation_history):
        message_agent = str(message.get("agent_name", ""))
        if message_agent and message_agent != agent_name:
            return str(message.get("content", ""))
        if not message_agent and message.get("role") == "user":
            return str(message.get("content", ""))
    return None


def format_recent_transcript(
    conversation_history: Sequence[Mapping[str, Any]],
    agent_name: str,
    limit: int = 4,
) -> str:
    """Render a role-neutral transcript containing at most the last ``limit`` messages."""
    if limit < 1:
        raise ValueError("Transcript limit must be at least one message.")
    if not conversation_history:
        return "(No transcript yet.)"

    lines: List[str] = []
    for message in conversation_history[-limit:]:
        speaker = message.get("agent_name")
        if not speaker:
            speaker = agent_name if message.get("role") == "assistant" else "Opponent"
        lines.append(f"[{speaker}]: {message.get('content', '')}")
    return "\n".join(lines)


def render_retrieved_arguments(
    retrieved_arguments: Sequence[Any],
    *,
    token_budget: Optional[int] = None,
    encoding_name: str = "cl100k_base",
) -> Tuple[str, int, bool]:
    """Render evidence under an optional exact tokenizer budget.

    The returned tuple is ``(text, token_count, budget_saturated)``. When the
    untruncated context exceeds the budget, the emitted context contains
    exactly ``token_budget`` tokens under the pinned encoding.
    """
    if not retrieved_arguments:
        text = "No retrieved arguments are available."
    else:
        lines = ["Retrieved arguments available to you:"]
        for argument in retrieved_arguments:
            polarity = getattr(argument, "polarity", "")
            polarity_display = POLARITY_DISPLAY.get(polarity, polarity).upper()
            claim = getattr(argument, "claim", str(argument))
            strength = float(getattr(argument, "support_strength", 0.0))
            lines.append(
                f"- [{polarity_display}] {claim} (strength: {strength:.2f})"
            )
        text = "\n".join(lines)

    if token_budget is not None:
        if isinstance(token_budget, bool) or int(token_budget) < 1:
            raise ValueError("retrieved-evidence token budget must be positive")
        try:
            import tiktoken
        except ImportError as error:
            raise RuntimeError(
                "Token-matched retrieval requires the declared tiktoken dependency."
            ) from error
        encoding = tiktoken.get_encoding(str(encoding_name))
        tokens = encoding.encode(text)
        saturated = len(tokens) >= int(token_budget)
        if len(tokens) > int(token_budget):
            tokens = tokens[: int(token_budget)]
            text = encoding.decode(tokens)
        return text, len(tokens), saturated

    try:
        import tiktoken
    except ImportError:
        return text, -1, False
    encoding = tiktoken.get_encoding(str(encoding_name))
    return text, len(encoding.encode(text)), False


def format_retrieved_arguments(retrieved_arguments: Sequence[Any]) -> str:
    """Backwards-compatible unbudgeted evidence renderer."""
    text, _token_count, _saturated = render_retrieved_arguments(
        retrieved_arguments
    )
    return text


def _matched_stance_components(
    stance_score: float,
    topic: str,
) -> Tuple[str, str, str]:
    """Map any scalar stance through the shared ten-bin instruction table."""
    score = max(-1.0, min(1.0, float(stance_score)))
    instructions: Dict[str, Any] = load_belief_instructions()
    selected_bin = _select_instruction_bin(score, instructions)
    intensity = str(selected_bin.get("intensity", "Unknown"))
    tone_instruction = str(
        selected_bin.get("instruction", "Match confidence to the current stance.")
    )
    directions = instructions.get("directions", {})

    if intensity.lower() == "neutral":
        direction = "NEUTRAL / UNCERTAIN"
        explicit_instruction = (
            f'Use this first-sentence format for granularity: "I am uncertain about '
            f'{topic}, and I can see both sides."'
        )
    elif score > 0:
        direction = "SUPPORT"
        verb = directions.get("positive_label", "support")
        prefix = selected_bin.get("explicit_prefix", "clearly")
        explicit_instruction = (
            f'Use this first-sentence format for granularity: "I {prefix} {verb} '
            f'{topic}." (Target intensity: {intensity}).'
        )
    else:
        direction = "OPPOSE"
        verb = directions.get("negative_label", "oppose")
        prefix = selected_bin.get("explicit_prefix", "clearly")
        explicit_instruction = (
            f'Use this first-sentence format for granularity: "I {prefix} {verb} '
            f'{topic}." (Target intensity: {intensity}).'
        )

    stance_description = (
        f"{intensity} {direction} (score: {score:+.3f}). {tone_instruction}"
    )
    return stance_description, tone_instruction, explicit_instruction


def build_matched_response_call(
    *,
    name: str,
    persona: str,
    topic: str,
    stance_score: float,
    retrieved_arguments: Sequence[Any],
    conversation_history: Sequence[Mapping[str, Any]],
    context_messages: int = 4,
    word_limit: int = 120,
    retrieved_evidence_token_budget: Optional[int] = None,
    retrieved_evidence_encoding: str = "cl100k_base",
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Build the condition-agnostic ``matched_v1`` response prompt and trigger.

    Update provenance is deliberately absent. If two conditions provide the
    same scalar stance, records, and public dialogue, their response calls are
    byte-identical.
    """
    if context_messages < 1:
        raise ValueError("Response context must include at least one message.")
    if word_limit < 1:
        raise ValueError("Response word limit must be positive.")

    last_opponent_message = find_last_opponent_message(conversation_history, name)
    if last_opponent_message:
        opponent_section = f'Opponent\'s Last Point:\n"{last_opponent_message}"'
        conversation_instruction = "DIRECTLY ADDRESS the opponent's last point."
    else:
        opponent_section = (
            "Opponent's Last Point:\n"
            "(This is the opening of the debate - no opponent message yet)"
        )
        conversation_instruction = (
            "Make your opening statement. Start directly with your core argument."
        )

    transcript = format_recent_transcript(
        conversation_history,
        name,
        limit=context_messages,
    )
    retrieved_context, _token_count, _saturated = render_retrieved_arguments(
        retrieved_arguments,
        token_budget=retrieved_evidence_token_budget,
        encoding_name=retrieved_evidence_encoding,
    )
    (
        stance_description,
        tone_instruction,
        explicit_stance_instruction,
    ) = _matched_stance_components(stance_score, topic)

    system_prompt = f"""
You are {name}.
Persona: {persona}
Topic Proposition: "{topic}"

Current Stance State:
{stance_description}

Retrieved Evidence:
{retrieved_context}

{opponent_section}

RECENT TRANSCRIPT (last {context_messages} messages):
{transcript}

Instructions:
1. Base your response on the current stance state, retrieved evidence, and recent transcript.
2. Never mention prompts, numeric state values, or belief-update machinery in your response.
"""
    system_prompt += build_shared_debate_rules(
        conversation_instruction=conversation_instruction,
        tone_instruction=tone_instruction,
        word_limit=word_limit,
        explicit_stance_instruction=explicit_stance_instruction,
    )
    system_prompt += (
        f"\n[SYSTEM]: It is now your turn, {name}. "
        "Respond to the Opponent based on the transcript above."
    )

    neutral_messages = [
        {
            "role": "user",
            "content": f"The debate continues. You are {name}. Respond now.",
        }
    ]
    return system_prompt, neutral_messages
