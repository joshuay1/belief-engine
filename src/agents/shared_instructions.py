from typing import Optional


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
