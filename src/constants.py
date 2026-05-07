"""
Project-wide constants.
"""

# Polarity Constants
POLARITY_AFFIRMATIVE = "affirmative"
POLARITY_NEGATIVE = "negative"

# Mapping from common LLM outputs and legacy terms to internal constants
POLARITY_MAPPING = {
    # Internal Constants (Identity)
    "affirmative": POLARITY_AFFIRMATIVE,
    "negative": POLARITY_NEGATIVE,
    
    # Common Side Names
    "pro": POLARITY_AFFIRMATIVE,
    "con": POLARITY_NEGATIVE,
    "for": POLARITY_AFFIRMATIVE,
    "against": POLARITY_NEGATIVE,
    
    # Argument Polarity
    "support": POLARITY_AFFIRMATIVE,
    "oppose": POLARITY_NEGATIVE,
    "refute": POLARITY_NEGATIVE,
    "refutation": POLARITY_NEGATIVE,
    "counter": POLARITY_NEGATIVE,
    "argument_against": POLARITY_NEGATIVE,
    "opposing": POLARITY_NEGATIVE,
    "disagree": POLARITY_NEGATIVE,
    "rejection": POLARITY_NEGATIVE
}

# Display Mapping for UI/Logs
POLARITY_DISPLAY = {
    POLARITY_AFFIRMATIVE: "Affirmative",
    POLARITY_NEGATIVE: "Negative"
}
