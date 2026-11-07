"""
Evaluation metrics for comparing predicted vs actual opinion change.
"""

import math
from typing import Any, Dict, List

ReplayResult = Any


def compute_metrics(results: List[ReplayResult]) -> Dict[str, float]:
    """Compute aggregate metrics from a list of replay results.

    Returns dict with:
        opinion_change_mae: Mean Absolute Error of predicted vs actual opinion change
        opinion_change_rmse: Root Mean Squared Error of opinion change
        opinion_change_mse: Mean Squared Error of opinion change (loss for optimization)
        final_stance_mae: MAE of predicted vs actual final stance
        pearson_r: Pearson correlation between predicted and actual opinion change
        direction_accuracy: Fraction where predicted change direction matches actual
        n: Number of results
    """
    n = len(results)
    if n == 0:
        return {k: float("nan") for k in [
            "opinion_change_mae", "opinion_change_rmse", "opinion_change_mse",
            "final_stance_mae", "pearson_r", "direction_accuracy", "n"
        ]}

    actual_changes = [r.opinion_change_actual for r in results]
    predicted_changes = [r.opinion_change_predicted for r in results]
    actual_finals = [r.final_stance_actual for r in results]
    predicted_finals = [r.final_stance_predicted for r in results]

    # MAE and RMSE of opinion change
    abs_errors = [abs(p - a) for p, a in zip(predicted_changes, actual_changes)]
    sq_errors = [(p - a) ** 2 for p, a in zip(predicted_changes, actual_changes)]
    opinion_change_mae = sum(abs_errors) / n
    opinion_change_mse = sum(sq_errors) / n
    opinion_change_rmse = math.sqrt(opinion_change_mse)

    # MAE of final stance
    final_abs_errors = [abs(p - a) for p, a in zip(predicted_finals, actual_finals)]
    final_stance_mae = sum(final_abs_errors) / n

    # Pearson r
    pearson_r = _pearson_r(actual_changes, predicted_changes)

    # Direction accuracy
    correct_dir = 0
    for a, p in zip(actual_changes, predicted_changes):
        if a == 0.0 and abs(p) < 0.05:
            correct_dir += 1
        elif a * p > 0:  # Same sign
            correct_dir += 1
        elif a == 0.0 or p == 0.0:
            pass  # One is zero, other isn't — count as wrong
    direction_accuracy = correct_dir / n

    return {
        "opinion_change_mae": opinion_change_mae,
        "opinion_change_rmse": opinion_change_rmse,
        "opinion_change_mse": opinion_change_mse,
        "final_stance_mae": final_stance_mae,
        "pearson_r": pearson_r,
        "direction_accuracy": direction_accuracy,
        "n": n,
    }


def _pearson_r(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return float("nan")

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return float("nan")
    return cov / denom
