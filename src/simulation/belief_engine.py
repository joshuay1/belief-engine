import os
import math
import json
import re
from typing import Dict, Optional, Tuple
from src.memory.argument_record import ArgumentRecord
from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE


def stance_to_log_odds(score: float) -> float:
    """Convert a stance score in [-1, 1] to log-odds, with finite clipping."""
    clipped = max(-0.999, min(0.999, float(score)))
    probability = (clipped + 1.0) / 2.0
    return math.log(probability / (1.0 - probability))


def _select_instruction_bin(score: float, instructions: Dict) -> Dict:
    """Selects the configured intensity bin for a stance score."""
    abs_score = abs(score)
    bins = instructions.get("positive_bins", [])
    selected_bin = {"intensity": "Unknown", "instruction": "State is undefined."}

    found = False
    for b in bins:
        min_val = b.get("min", 0.0)
        max_val = b.get("max", 1.0)
        if min_val <= abs_score < max_val:
            selected_bin = b
            found = True
            break

    if not found and bins:
        selected_bin = bins[-1]

    return selected_bin


def _format_belief_description(score: float, instructions: Dict) -> str:
    directions = instructions.get("directions", {})
    if score > 0:
        direction = "SUPPORT"
        action = directions.get("positive", "Argue IN FAVOR.")
    else:
        direction = "OPPOSE"
        action = directions.get("negative", "Argue AGAINST.")

    selected_bin = _select_instruction_bin(score, instructions)
    intensity = selected_bin.get("intensity", "Unknown")
    instruction = selected_bin.get("instruction", "")

    return f"**{intensity} {direction}** (Score: {score:.2f}). {action} {instruction}"


def _format_explicit_stance_instruction(
    score: float, instructions: Dict, topic_placeholder: str = "the proposition"
) -> str:
    selected_bin = _select_instruction_bin(score, instructions)
    intensity = selected_bin.get("intensity", "Unknown")
    directions = instructions.get("directions", {})

    # Neutral bin should explicitly signal uncertainty regardless of sign.
    if str(intensity).lower() == "neutral":
        return (
            f'Use this first-sentence format for granularity: "I am uncertain about {topic_placeholder}, '
            f'and I can see both sides."'
        )

    if score > 0:
        verb = directions.get("positive_label", "support")
    else:
        verb = directions.get("negative_label", "oppose")

    prefix = selected_bin.get("explicit_prefix", "clearly")
    return (
        f'Use this first-sentence format for granularity: "I {prefix} {verb} {topic_placeholder}." '
        f"(Target intensity: {intensity})."
    )


def load_belief_instructions() -> Dict:
    """Loads natural language instructions from JSON."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "belief_instructions.json")

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load belief_instructions.json: {e}. Using defaults.")
        return {
            "directions": {
                "positive": "Argue IN FAVOR.",
                "negative": "Argue AGAINST."
            },
            "positive_bins": [
                {"min": 0.0, "max": 0.1, "intensity": "Neutral", "instruction": "Undecided."},
                {"min": 0.1, "max": 1.1, "intensity": "Certain", "instruction": "Decided."}
            ]
        }

class BayesianBeliefEngine:
    """
    A standardized cognitive engine for managing an agent's belief state.
    
    Features:
    - Maintains a persistent 'log_odds' state (Bayesian Stance).
    - Updates belief deterministically based on accepted arguments.
    - Translates mathematical state into semantic instructions for the LLM.
    """
    
    def __init__(self, agent_name: str, memory_dir: str, config: Dict = None):
        self.agent_name = agent_name
        self.memory_dir = memory_dir
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir, exist_ok=True)
        self.state_file = os.path.join(memory_dir, "belief_state.json")
        
        # Configuration Defaults
        if config is None:
            config = {}
        self.sensitivity = config.get("update_sensitivity", 2.0)
        # Check for anchor_sensitivity, default to regular sensitivity if not set
        self.anchor_sensitivity = config.get("anchor_sensitivity", self.sensitivity)
        if self.anchor_sensitivity is None:
            self.anchor_sensitivity = self.sensitivity
        self.confirmation_bias = config.get("confirmation_bias", 0.0)
        
        # Load Instructions
        self.instructions = self.load_instructions()
        
        # State
        self.log_odds = 0.0
        self.history_count = 0
        
        # Load existing state if available
        self.load_state()

    def load_instructions(self) -> Dict:
        """Loads natural language instructions from JSON."""
        return load_belief_instructions()

    def update(self, argument: ArgumentRecord, backend=None, persist: bool = True, **kwargs) -> float:
        """
        Updates the internal belief state based on a new argument.
        Returns the delta (change in log-odds).
        """
        strength = argument.support_strength
        
        # Calculate Likelihood Ratio (Bayes Factor)
        # LR = 1 + (Strength * Sensitivity * Bias)
        
        # Determine Bias and Sensitivity to use
        bias_multiplier = 1.0
        current_sensitivity = self.sensitivity

        # Special handling for seeds (Anchors)
        if getattr(argument, 'source_role', '') == 'seed':
             bias_multiplier = 1.0 # No bias for seeds
             current_sensitivity = self.anchor_sensitivity # Use anchor sensitivity
        elif self.confirmation_bias > 0:
            current_polarity = 1 if self.log_odds > 0 else -1
            arg_polarity = 1 if argument.polarity == POLARITY_AFFIRMATIVE else -1
            
            if arg_polarity == current_polarity:
                bias_multiplier = 1.0 + self.confirmation_bias
            else:
                bias_multiplier = max(0.0, 1.0 - self.confirmation_bias)

        lr = 1.0 + (strength * current_sensitivity * bias_multiplier)
        
        delta = 0.0
        if argument.polarity == POLARITY_AFFIRMATIVE:
            delta = math.log(lr)
            self.log_odds += delta
        elif argument.polarity == POLARITY_NEGATIVE:
            delta = -math.log(lr)
            self.log_odds += delta
            
        self.history_count += 1
        if persist:
            self.save_state()
        
        return delta

    def get_belief_score(self) -> float:
        """
        Returns the standardized stance score between -1.0 and 1.0.
        """
        # Convert Log Odds to Probability (Sigmoid)
        probability = 1.0 / (1.0 + math.exp(-self.log_odds))
        
        # Map Probability (0 to 1) to Stance (-1 to 1)
        return (probability * 2.0) - 1.0

    def set_belief_score(self, score: float, persist: bool = True) -> None:
        """Directly set the latent Bayesian stance score."""
        self.log_odds = stance_to_log_odds(score)
        if persist:
            self.save_state()

    def get_belief_description(self) -> str:
        """
        Returns a detailed natural language instruction based on the belief score.
        Uses symmetric bins from external configuration file.
        """
        score = self.get_belief_score()
        return _format_belief_description(score, self.instructions)

    def get_explicit_stance_instruction(self, topic_placeholder: str = "the proposition") -> str:
        """
        Returns JSON-config-driven instruction for how explicit the first stance sentence should be.
        """
        score = self.get_belief_score()
        return _format_explicit_stance_instruction(score, self.instructions, topic_placeholder)

    def save_state(self):
        """Persists the belief state to JSON."""
        data = {
            "log_odds": self.log_odds,
            "history_count": self.history_count
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_state(self):
        """Loads belief state from JSON if it exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.log_odds = data.get("log_odds", 0.0)
                    self.history_count = data.get("history_count", 0)
            except Exception as e:
                print(f"Warning: Could not load belief state for {self.agent_name}: {e}")
                self.log_odds = 0.0
                self.history_count = 0


class AgenticBeliefEngine:
    """
    Agentic (LLM-driven) belief engine that updates stance based on the same inputs
    (polarity, strength, claim, and current state), but uses an LLM instead
    of Bayesian math to determine the next belief score.
    """

    def __init__(self, agent_name: str, memory_dir: str, config: Dict = None):
        self.agent_name = agent_name
        self.memory_dir = memory_dir
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir, exist_ok=True)
        self.state_file = os.path.join(memory_dir, "belief_state.json")

        if config is None:
            config = {}
        self.update_sensitivity = config.get("update_sensitivity", 1.0)
        self.anchor_sensitivity = config.get("anchor_sensitivity", None)
        self.confirmation_bias = config.get("confirmation_bias", 0.0)

        self.instructions = load_belief_instructions()

        self.score = 0.0
        self.history_count = 0
        self.last_rationale = ""

        self.load_state()

    def _parse_llm_response(self, text: str) -> Optional[Dict]:
        """Extract a JSON object from the LLM response."""
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    def update(self, argument: ArgumentRecord, backend=None, memory_context: Optional[str] = None, persist: bool = True, **kwargs) -> float:
        """
        Updates the belief score using an LLM.
        Returns the delta (change in score).
        """
        if backend is None:
            return 0.0

        current_score = self.score
        polarity = argument.polarity
        strength = argument.support_strength
        source_role = getattr(argument, "source_role", "")
        claim = argument.claim

        # Calculate constraints
        anchor_sensitivity = self.anchor_sensitivity if self.anchor_sensitivity is not None else self.update_sensitivity
        min_val = max(-1.0, current_score - self.update_sensitivity)
        max_val = min(1.0, current_score + self.update_sensitivity)

        memory_block = memory_context if memory_context else "(none)"

        prompt = f"""
You are a belief update engine. Update the stance score based on a single new argument.

Current stance score: {current_score:.3f}  (range -1 to +1)
New argument:
- polarity: {polarity}
- support_strength: {strength:.3f}
- source_role: {source_role}
- claim: "{claim}"

Current memory (active arguments):
{memory_block}

Rules:
1. Move the score toward +1 for affirmative and toward -1 for negative.
2. Use support_strength and update_sensitivity={self.update_sensitivity:.2f} to scale the step size.
3. If source_role is "seed", use anchor_sensitivity={anchor_sensitivity:.2f} to allow a stronger anchor.
4. confirmation_bias={self.confirmation_bias:.2f}: if argument polarity matches the current sign, amplify; if opposing, dampen.
5. Keep the output score in [-1.0, 1.0].

Constraint: The maximum change allowed is {self.update_sensitivity:.2f}.
The new score MUST be between {min_val:.3f} and {max_val:.3f}.

Return JSON only:
{{"stance_score": <number>, "rationale": "<short reason>"}}
"""

        response = backend.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You update belief scores deterministically from arguments.",
            temperature=0.1
        )

        data = self._parse_llm_response(response)
        if not data or "stance_score" not in data:
            return 0.0

        try:
            new_score = float(data.get("stance_score"))
        except Exception:
            return 0.0

        new_score = max(-1.0, min(1.0, new_score))
        self.last_rationale = str(data.get("rationale", "")).strip()

        delta = new_score - current_score
        self.score = new_score
        self.history_count += 1
        if persist:
            self.save_state()
        return delta

    def get_belief_score(self) -> float:
        return self.score

    def set_belief_score(self, score: float, persist: bool = True) -> None:
        """Directly set the latent stance score for calibrated initialization."""
        self.score = max(-1.0, min(1.0, float(score)))
        if persist:
            self.save_state()

    def get_belief_description(self) -> str:
        score = self.get_belief_score()
        return _format_belief_description(score, self.instructions)

    def get_explicit_stance_instruction(self, topic_placeholder: str = "the proposition") -> str:
        """
        Returns JSON-config-driven instruction for how explicit the first stance sentence should be.
        """
        score = self.get_belief_score()
        return _format_explicit_stance_instruction(score, self.instructions, topic_placeholder)

    def save_state(self):
        data = {
            "score": self.score,
            "history_count": self.history_count,
            "last_rationale": self.last_rationale
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    if "score" in data:
                        self.score = data.get("score", 0.0)
                    elif "log_odds" in data:
                        log_odds = data.get("log_odds", 0.0)
                        probability = 1.0 / (1.0 + math.exp(-log_odds))
                        self.score = (probability * 2.0) - 1.0
                    self.history_count = data.get("history_count", 0)
                    self.last_rationale = data.get("last_rationale", "")
            except Exception as e:
                print(f"Warning: Could not load belief state for {self.agent_name}: {e}")
                self.score = 0.0
                self.history_count = 0
                self.last_rationale = ""


def create_belief_engine(agent_name: str, memory_dir: str, config: Dict = None):
    if config is None:
        config = {}
    engine_type = config.get("belief_engine_type", "bayesian")
    if engine_type in ("agentic", "llm"):
        return AgenticBeliefEngine(agent_name=agent_name, memory_dir=memory_dir, config=config)
    return BayesianBeliefEngine(agent_name=agent_name, memory_dir=memory_dir, config=config)
