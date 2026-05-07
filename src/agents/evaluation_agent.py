import json
from pathlib import Path
from typing import Dict
from ..backend.backend_base import LLMBackend

DEFAULT_OPTIMIZED_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "evaluator_optimized.txt"


def _builtin_default_stance_prompt() -> str:
    """Concise, topic-agnostic baseline used when no optimized prompt file exists."""
    return """
You are an evaluator. Infer stance from ONE statement about the proposition/topic.

Topic context: "{topic}"
Speaker: "{agent_name}"

Return ONLY JSON:
{{"stance_score": <float between -1.0 and 1.0>}}

Scoring rules:
1. Sign: positive = supports, negative = opposes, near 0 = unclear/mixed.
2. Strength is about commitment, not emotional wording alone.
3. Start from this granular anchor ladder (center-first, symmetric):
   0.00, ±0.05, ±0.10, ±0.15, ±0.20, ±0.30, ±0.40, ±0.50, ±0.60, ±0.70, ±0.80, ±0.90, ±0.95
4. If statement is descriptive (pros/cons) without clear recommendation, cap |score| at 0.40.
5. Require explicit recommendation/commitment ("should", "must", "need to", "have to") for |score| >= 0.50.
6. Use |score| >= 0.80 only for categorical language that denies meaningful counter-position.
7. Avoid coarse binning: do NOT default to ±0.50/±0.80 when evidence is weaker/stronger.
8. For borderline cases between anchors, interpolate with two decimals (for example 0.27, -0.58).
9. If uncertain between two nearby levels, choose the less extreme value.
"""


def _load_default_stance_prompt() -> str:
    """Loads optimized prompt from disk if available, else returns built-in baseline."""
    try:
        if DEFAULT_OPTIMIZED_PROMPT_PATH.exists():
            content = DEFAULT_OPTIMIZED_PROMPT_PATH.read_text(encoding="utf-8").strip()
            if content:
                return content
    except Exception:
        pass
    return _builtin_default_stance_prompt()


def _render_stance_prompt(template: str, topic: str, agent_name: str) -> str:
    """Lightweight placeholder replacement without invoking str.format on JSON braces."""
    return template.replace("{topic}", topic).replace("{agent_name}", agent_name)


class EvaluationAgent:
    """
    Evaluates the subject's stance and persona stability.
    """
    def __init__(self, backend: LLMBackend, temperature: float = 0.0):
        self.backend = backend
        self.temperature = temperature
        self.default_stance_prompt_template = _load_default_stance_prompt()

    def evaluate_stance(self, transcript: str, memory_snapshot: str, topic: str, agent_name: str, expected_stance_instruction: str = None, custom_system_prompt: str = None) -> Dict[str, float]:
        """
        Estimates stance based on the transcript and memory.
        Returns a score from -1.0 (Con) to 1.0 (Pro).
        When expected_stance_instruction is provided, scores based on alignment with that instruction.
        When custom_system_prompt is provided, bypasses default prompts (used for optimization).
        """
        # If custom prompt provided (Optimization Mode), use it directly
        if custom_system_prompt:
            system_prompt = custom_system_prompt
        # If we have an instruction, use instruction-aligned scoring
        elif expected_stance_instruction:
            system_prompt = f"""
        Analyze the provided statement regarding the topic: "{topic}".
        
        AGENT'S INTERNAL INSTRUCTION (what they were told to believe and argue):
        {expected_stance_instruction}
        
        Your task: Score the agent's statement based on HOW WELL they followed their internal instruction.
        
        SCORING RULES:
        1. Extract the POLARITY from the instruction:
           - If instruction says "SUPPORT" or "IN FAVOR" -> Agent should score POSITIVE
           - If instruction says "OPPOSE" or "AGAINST" -> Agent should score NEGATIVE
        
        2. Extract the INTENSITY from the instruction (e.g., "Moderate-Strong", "Total Certainty", etc.)
        
        3. Score the agent's statement by matching BOTH polarity AND intensity:
           - If instruction says "Moderate-Strong OPPOSE" and agent argues against with confidence -> score around -0.55
           - If instruction says "Total Certainty SUPPORT" and agent fully supports -> score around +0.95
           - If agent contradicts their instruction (says SUPPORT when told OPPOSE) -> score should reflect what they SAID, but acknowledge the mismatch
        
        SCORING GUIDE based on instruction intensity:
        - "Total Certainty": +/- 0.93 to 1.00
        - "Near Certainty": +/- 0.83 to 0.92
        - "Very Strong": +/- 0.73 to 0.82
        - "Strong": +/- 0.63 to 0.72
        - "Moderate-Strong": +/- 0.53 to 0.62
        - "Moderate": +/- 0.43 to 0.52
        - "Moderate-Weak": +/- 0.33 to 0.42
        - "Weak": +/- 0.23 to 0.32
        - "Very Weak": +/- 0.08 to 0.22
        - "Neutral": +/- 0.00 to 0.07

        Granularity rules:
        - Use two decimals.
        - Use the full range, avoid collapsing into only a few repeated bins.
        - If between two levels, use an intermediate value instead of snapping to a coarse anchor.
        
        The sign (+/-) should match the instruction's polarity (SUPPORT = positive, OPPOSE = negative).
        
        Output a JSON object with a single key "stance_score" between -1.0 and 1.0.
        Use two decimal precision.
        """
        else:
            system_prompt = _render_stance_prompt(self.default_stance_prompt_template, topic, agent_name)
        
        prompt = f"Statement to Evaluate:\n{transcript}"

        response = self.backend.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=self.temperature
        )

        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            
            # If "stance_score" is already present (from the instruction-based branch), return it
            if "stance_score" in data:
                data.pop("confidence_score", None)
                return data

            # Fallback if LLM outputs label but not score (rare with new prompt)
            if "stance_label" in data and "stance_score" not in data:
                 # Minimal mapping just in case
                 label = data.get("stance_label", "Neutral")
                 mapping = {"Extremely Support": 1.0, "Strongly Support": 0.75, "Support": 0.5, "Weakly Support": 0.25, "Neutral": 0.0, "Weakly Oppose": -0.25, "Oppose": -0.5, "Strongly Oppose": -0.75, "Extremely Oppose": -1.0}
                 return {"stance_score": mapping.get(label, 0.0)}
            
            return data

        except Exception as e:
            print(f"Error parsing evaluation response: {e}")
            return {"stance_score": 0.0, "stance_label": "Neutral"}


    def evaluate_consistency(self, initial_persona: str, transcript: str) -> Dict[str, float]:
        """
        Evaluates how well the agent stuck to its persona.
        """
        system_prompt = """
        Analyze the debate transcript against the initial persona.
        Rate the consistency on a scale of 0.0 to 1.0.
        Output JSON: {"consistency_score": float, "reasoning": string}
        """
        
        prompt = f"Initial Persona: {initial_persona}\n\nTranscript:\n{transcript}"

        response = self.backend.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=0.0
        )

        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except:
             return {"consistency_score": 0.5, "reasoning": "Parse error"}

    def summarize_debate(self, transcript: str, initial_stance: float, final_stance: float) -> str:
        """
        Generates a narrative summary of the debate.
        """
        system_prompt = """
        You are an expert debate analyst. 
        Analyze the provided transcript and stance changes.
        Provide a concise narrative summary (max 150 words) explaining:
        1. How the debate unfolded.
        2. Key arguments that influenced the Subject Agent.
        3. Why the stance shifted (or didn't).
        4. The dynamic between the agents.
        
        IMPORTANT: Always refer to the participants as "Counter Agent" and "Subject Agent". Do NOT use "user" or "assistant".
        """
        
        prompt = f"""
        Initial Stance: {initial_stance}
        Final Stance: {final_stance}
        
        Transcript:
        {transcript}
        """

        response = self.backend.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        return response
