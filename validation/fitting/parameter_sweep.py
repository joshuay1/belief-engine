"""
Parameter fitting via dense grid search over (K, K_a).

With only two parameters, a dense grid gives full landscape coverage
without the complexity and cost of gradient-based optimization.
"""

import itertools
import time
from typing import Any, Dict, List, Tuple

from ..data.preprocess import ParticipantTimeline
from .metrics import compute_metrics

TranscriptReplayEngine = Any
ReplayResult = Any


# Default grid: 10x10 = 100 combinations
DEFAULT_K_GRID = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
DEFAULT_KA_GRID = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]


def replay_all(
    engine: TranscriptReplayEngine,
    timelines: List[ParticipantTimeline],
    K: float,
    K_a: float,
) -> List[ReplayResult]:
    """Replay all participants with given parameters."""
    results = []
    for t in timelines:
        try:
            result = engine.replay_participant(t, K=K, K_a=K_a)
            results.append(result)
        except Exception:
            continue
    return results


def run_grid_search(
    engine: TranscriptReplayEngine,
    timelines: List[ParticipantTimeline],
    K_grid: List[float] = None,
    K_a_grid: List[float] = None,
) -> List[Dict]:
    """Run grid search over (K, K_a) combinations.

    Returns list of dicts with params and metrics, sorted by opinion_change_mse.
    """
    K_grid = K_grid or DEFAULT_K_GRID
    K_a_grid = K_a_grid or DEFAULT_KA_GRID

    combos = list(itertools.product(K_grid, K_a_grid))
    print(f"Grid search: {len(combos)} combinations x {len(timelines)} participants")

    results = []
    start = time.time()

    for i, (K, K_a) in enumerate(combos):
        replay_results = replay_all(engine, timelines, K, K_a)
        metrics = compute_metrics(replay_results)

        entry = {"K": K, "K_a": K_a, **metrics}
        results.append(entry)

        elapsed = time.time() - start
        print(f"  [{i+1}/{len(combos)}] K={K:.2f} K_a={K_a:.2f} | "
              f"MAE={metrics['opinion_change_mae']:.4f} r={metrics['pearson_r']:.4f} "
              f"dir_acc={metrics['direction_accuracy']:.3f} ({elapsed:.1f}s)")

    results.sort(key=lambda x: x["opinion_change_mse"])
    return results


def run_full_sweep(
    engine: TranscriptReplayEngine,
    timelines: List[ParticipantTimeline],
    K_grid: List[float] = None,
    K_a_grid: List[float] = None,
) -> Tuple[List[Dict], Dict]:
    """Run dense grid search and return full results + best params.

    Returns:
        (grid_results, best_params) where grid_results is the full grid
        sorted by MSE, and best_params is the top result.
    """
    print("=" * 60)
    print("Grid search over (K, K_a)")
    print("=" * 60)
    grid_results = run_grid_search(engine, timelines, K_grid, K_a_grid)

    best = grid_results[0]

    print(f"\n{'='*60}")
    print("BEST RESULT")
    print("=" * 60)
    print(f"  Best K:   {best['K']:.4f}")
    print(f"  Best K_a: {best['K_a']:.4f}")
    print(f"  Opinion change MAE:  {best['opinion_change_mae']:.4f}")
    print(f"  Opinion change RMSE: {best['opinion_change_rmse']:.4f}")
    print(f"  Pearson r:           {best['pearson_r']:.4f}")
    print(f"  Direction accuracy:  {best['direction_accuracy']:.3f}")

    return grid_results, best
