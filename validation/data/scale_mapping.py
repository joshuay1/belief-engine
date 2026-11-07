"""
Bidirectional mapping between DEBATE 6-point Likert scale and Belief Engine [-1, 1] stance.

DEBATE scale:
  1 = Certainly disagree
  2 = Probably disagree
  3 = Lean disagree
  4 = Lean agree
  5 = Probably agree
  6 = Certainly agree

Psychometric anchor mapping (replaces the previous linear mapping):

  The previous approach linearly mapped Likert 1->-1.0 and Likert 6->+1.0, then
  converted to log-odds. This caused Likert 6 to produce log-odds of ~6.9 (near-infinity),
  which violates Cromwell's Rule (Bayesian Epistemology): no subjective belief should
  ever carry probability 1.0, as this makes it mathematically impossible to update.

  The new mapping uses pre-defined ordinal anchors grounded in two theoretical pillars:

  1. Item Response Theory / Graded Response Model (Psychometrics):
     Likert scales are ORDINAL, not INTERVAL. The psychological distance between
     "Lean Agree" (4) and "Probably Agree" (5) is smaller than the distance between
     "Probably Agree" (5) and "Certainly Agree" (6). The anchor gaps reflect this
     by growing non-linearly toward the extremes.

  2. Cromwell's Rule (Bayesian Epistemology):
     Absolute certainty (log-odds = +-inf) must never be assigned to a subjective
     belief. The anchors are capped at +-3.0 log-odds (~95% probability), preserving
     the Belief Engine's mathematical ability to update any agent's stance.

  Anchor table:
    Likert 1 (Certainly disagree) -> -3.0  (~4.7% probability)
    Likert 2 (Probably disagree)  -> -1.5  (~18.2% probability)
    Likert 3 (Lean disagree)      -> -0.5  (~37.8% probability)
    Likert 4 (Lean agree)         ->  0.5  (~62.2% probability)
    Likert 5 (Probably agree)     ->  1.5  (~81.8% probability)
    Likert 6 (Certainly agree)    ->  3.0  (~95.3% probability)
"""

import math


# ---------------------------------------------------------------------------
# Psychometric anchor table (IRT-spaced, Cromwell-bounded)
# ---------------------------------------------------------------------------

_LIKERT_LOG_ODDS_ANCHORS: dict[int, float] = {
    1: -3.0,  # Certainly disagree  (~4.7% probability)
    2: -1.5,  # Probably disagree   (~18.2% probability)
    3: -0.5,  # Lean disagree       (~37.8% probability)
    4:  0.5,  # Lean agree          (~62.2% probability)
    5:  1.5,  # Probably agree      (~81.8% probability)
    6:  3.0,  # Certainly agree     (~95.3% probability)
}


def likert_to_log_odds(likert: float) -> float:
    """Map Likert 1-6 to log-odds using psychometric anchors.

    For integer values, returns the exact anchor. For fractional values
    (e.g. 4.5), linearly interpolates between the two neighbouring anchors
    so that the non-linear spacing is preserved between whole-number buckets.

    Grounded in:
      - Item Response Theory (Graded Response Model) for non-linear spacing.
      - Cromwell's Rule for the +-3.0 log-odds ceiling (~95% certainty max).
    """
    likert = max(1.0, min(6.0, float(likert)))  # Clamp to valid range

    lower = int(math.floor(likert))
    upper = int(math.ceil(likert))

    if lower == upper:
        return _LIKERT_LOG_ODDS_ANCHORS[lower]

    # Interpolate between neighbouring anchors
    frac = likert - lower
    return (
        (1.0 - frac) * _LIKERT_LOG_ODDS_ANCHORS[lower]
        + frac * _LIKERT_LOG_ODDS_ANCHORS[upper]
    )


def log_odds_to_stance(log_odds: float) -> float:
    """Convert log-odds to stance [-1, 1]. Mirrors BayesianBeliefEngine.get_belief_score()."""
    probability = 1.0 / (1.0 + math.exp(-log_odds))
    return (probability * 2.0) - 1.0


def likert_to_stance(likert: float) -> float:
    """Map Likert 1-6 to stance [-1, 1] via the psychometric anchor table."""
    return log_odds_to_stance(likert_to_log_odds(likert))


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward-compatibility with any callers that need
# the raw stance <-> Likert round-trip without going through log-odds)
# ---------------------------------------------------------------------------

def stance_to_likert(stance: float) -> float:
    """Approximate inverse: stance [-1, 1] back to Likert 1-6.

    Uses a simple linear approximation. Prefer using log_odds_to_stance
    for the authoritative conversion direction.
    """
    return (stance + 1.0) / 2.0 * 5.0 + 1.0


def stance_to_log_odds(stance: float) -> float:
    """Convert a raw stance [-1, 1] to log-odds.

    NOTE: This function is retained for callers that supply a continuous
    stance value rather than a discrete Likert score. It still clamps at
    +-3.0 to respect Cromwell's Rule.
    """
    p = (stance + 1.0) / 2.0
    # Clamp to [~4.7%, ~95.3%] — equivalent to the +-3.0 log-odds ceiling
    p = max(0.047, min(0.953, p))
    return math.log(p / (1.0 - p))
