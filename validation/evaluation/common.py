"""
Reusable helpers for DEBATE validation.

This module centralizes logic that had grown across exploratory scripts:
cleaning, argument-cache access, participant typing, fold construction,
baseline predictions, and group-level summaries.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE
from src.memory.argument_record import ArgumentRecord
from validation.data.preprocess import ParticipantTimeline


@dataclass(frozen=True)
class TimelineRecord:
    """A participant timeline plus precomputed validation annotations."""

    timeline: ParticipantTimeline
    n_args: int
    initial_text_net_polarity: float
    participant_type: str


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def reclass_hash(topic: str, claim: str) -> str:
    return hashlib.sha256(f"{topic}|||{claim}".encode("utf-8")).hexdigest()[:20]


def get_cached_args(
    text: str,
    topic: str,
    extraction_cache: dict,
    reclass_cache: dict | None = None,
) -> list[dict]:
    """Return cached argument dicts for text, optionally applying polarity reclass."""
    msg_hash = hash_text(text)
    args = extraction_cache.get(msg_hash, [])
    if not reclass_cache:
        return args

    out = []
    for arg in args:
        copied = dict(arg)
        key = reclass_hash(topic, copied.get("claim", ""))
        if key in reclass_cache:
            copied["polarity"] = reclass_cache[key].get(
                "reclassified_polarity", copied.get("polarity", "")
            )
            copied["polarity_confidence"] = reclass_cache[key].get("confidence", 1.0)
        else:
            copied["polarity_confidence"] = 1.0
        out.append(copied)
    return out


class CacheOnlyArgumentExtractor:
    """Extractor-compatible cache reader that never performs API calls."""

    def __init__(self, extraction_cache: dict):
        self._cache = extraction_cache
        self._record_cache: dict[tuple[str, str], list[ArgumentRecord]] = {}

    def extract(self, text: str, source_role: str) -> list[ArgumentRecord]:
        key = hash_text(text)
        cache_key = (key, source_role)
        if cache_key in self._record_cache:
            return self._record_cache[cache_key]
        records = [
            ArgumentRecord(
                topic_id=d.get("topic_id", "generic"),
                claim=d.get("claim", ""),
                polarity=d.get("polarity", ""),
                source_role=source_role,
                support_strength=d.get("support_strength", 0.0),
                support_strength_llm=d.get(
                    "support_strength_llm", d.get("support_strength", 0.0)
                ),
                support_strength_classifier=d.get(
                    "support_strength_classifier", d.get("support_strength", 0.0)
                ),
                evidence_snippets=d.get("evidence_snippets", []),
                category_tags=d.get("category_tags", []),
            )
            for d in self._cache.get(key, [])
        ]
        self._record_cache[cache_key] = records
        return records

    @property
    def cache_size(self) -> int:
        return len(self._cache)


def timeline_texts(timeline: ParticipantTimeline, include_initial: bool = True) -> list[str]:
    texts: list[str] = []
    if include_initial and timeline.initial_text:
        texts.append(timeline.initial_text)
    for rd in timeline.rounds:
        texts.extend(rd.tweets_received)
        texts.extend(rd.messages_received)
        if rd.tweet_sent:
            texts.append(rd.tweet_sent)
        texts.extend(rd.messages_sent)
    return texts


def is_ghost_session(timeline: ParticipantTimeline) -> bool:
    """Frontend artifact: no initial text and all sent messages are undefined/blank."""
    if timeline.initial_text and timeline.initial_text.strip():
        return False
    sent: list[str] = []
    for rd in timeline.rounds:
        sent.extend(rd.messages_sent)
        if rd.tweet_sent:
            sent.append(rd.tweet_sent)
    if not sent:
        return True
    return all(str(m).strip().lower() in ("", "undefined") for m in sent)


def count_cached_args(
    timeline: ParticipantTimeline,
    extraction_cache: dict,
    reclass_cache: dict | None = None,
) -> int:
    total = 0
    for text in timeline_texts(timeline):
        total += len(get_cached_args(text, timeline.topic, extraction_cache, reclass_cache))
    return total


def net_polarity_for_text(
    text: str,
    topic: str,
    extraction_cache: dict,
    reclass_cache: dict | None = None,
    weighted: bool = False,
) -> float:
    net = 0.0
    for arg in get_cached_args(text, topic, extraction_cache, reclass_cache):
        strength = float(arg.get("support_strength", 1.0)) if weighted else 1.0
        if arg.get("polarity") == POLARITY_AFFIRMATIVE:
            net += strength
        elif arg.get("polarity") == POLARITY_NEGATIVE:
            net -= strength
    return net


def argument_pressure(
    timeline: ParticipantTimeline,
    extraction_cache: dict,
    reclass_cache: dict | None = None,
    include_initial: bool = True,
) -> tuple[float, int]:
    """Compute weighted affirmative-minus-negative pressure for a timeline."""
    pressure = 0.0
    n_args = 0
    for text in timeline_texts(timeline, include_initial=include_initial):
        for arg in get_cached_args(text, timeline.topic, extraction_cache, reclass_cache):
            strength = float(arg.get("support_strength", 0.0))
            if arg.get("polarity") == POLARITY_AFFIRMATIVE:
                pressure += strength
            elif arg.get("polarity") == POLARITY_NEGATIVE:
                pressure -= strength
            n_args += 1
    return pressure, n_args


def assign_participant_type(initial_stance: float, initial_text_net_polarity: float) -> str:
    """Assign the pre-debate participant taxonomy from the cleaning protocol."""
    prior_extremity = abs(initial_stance)
    mismatch = (
        abs(initial_text_net_polarity) > 0
        and ((initial_text_net_polarity > 0) != (initial_stance > 0))
    )
    if prior_extremity < 0.25 or mismatch:
        return "open"
    if prior_extremity < 0.75:
        return "leaning"
    return "committed"


def build_records(
    timelines: Sequence[ParticipantTimeline],
    extraction_cache: dict,
    reclass_cache: dict | None = None,
) -> list[TimelineRecord]:
    records = []
    for timeline in timelines:
        n_args = count_cached_args(timeline, extraction_cache, reclass_cache)
        initial_net = (
            net_polarity_for_text(
                timeline.initial_text, timeline.topic, extraction_cache, reclass_cache
            )
            if timeline.initial_text
            else 0.0
        )
        records.append(
            TimelineRecord(
                timeline=timeline,
                n_args=n_args,
                initial_text_net_polarity=initial_net,
                participant_type=assign_participant_type(
                    timeline.initial_stance, initial_net
                ),
            )
        )
    return records


def filter_records(records: Sequence[TimelineRecord], mode: str) -> list[TimelineRecord]:
    """Filter records for canonical validation.

    Modes:
    - none: keep everything complete
    - quality: drop ghost sessions and zero-argument participants
    - movers: quality filter plus |opinion_change| > 0.01
    """
    if mode not in {"none", "quality", "movers"}:
        raise ValueError(f"Unknown clean mode: {mode}")

    filtered = []
    for record in records:
        timeline = record.timeline
        if mode in {"quality", "movers"}:
            if is_ghost_session(timeline) or record.n_args == 0:
                continue
        if mode == "movers":
            if timeline.opinion_change is None or abs(timeline.opinion_change) <= 0.01:
                continue
        filtered.append(record)
    return filtered


def make_folds(
    records: Sequence[TimelineRecord],
    n_folds: int,
    seed: int,
    split_by: str = "group",
) -> list[list[TimelineRecord]]:
    """Make participant, group, or topic folds."""
    if split_by not in {"participant", "group", "topic"}:
        raise ValueError(f"Unknown split_by: {split_by}")

    rng = random.Random(seed)
    folds: list[list[TimelineRecord]] = [[] for _ in range(n_folds)]

    if split_by == "participant":
        shuffled = list(records)
        rng.shuffle(shuffled)
        for idx, record in enumerate(shuffled):
            folds[idx % n_folds].append(record)
        return folds

    buckets: dict[str, list[TimelineRecord]] = defaultdict(list)
    for record in records:
        key = (
            record.timeline.group_file
            if split_by == "group"
            else record.timeline.topic
        )
        buckets[key].append(record)

    keys = list(buckets)
    rng.shuffle(keys)
    for idx, key in enumerate(keys):
        folds[idx % n_folds].extend(buckets[key])
    return folds


def metric_dict(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    n = len(actual)
    if n == 0:
        return {
            "n": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "mse": math.nan,
            "pearson_r": math.nan,
            "spearman_r": math.nan,
            "direction_accuracy": math.nan,
        }

    errors = [p - a for a, p in zip(actual, predicted)]
    abs_errors = [abs(e) for e in errors]
    sq_errors = [e * e for e in errors]
    mse = sum(sq_errors) / n
    return {
        "n": n,
        "mae": sum(abs_errors) / n,
        "rmse": math.sqrt(mse),
        "mse": mse,
        "pearson_r": pearson(actual, predicted),
        "spearman_r": spearman(actual, predicted),
        "direction_accuracy": direction_accuracy(actual, predicted),
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return math.nan
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom else math.nan


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return math.nan
    return pearson(rank_values(x), rank_values(y))


def rank_values(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def direction_accuracy(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if not actual:
        return math.nan
    correct = 0
    for a, p in zip(actual, predicted):
        if abs(a) <= 0.01 and abs(p) < 0.05:
            correct += 1
        elif a * p > 0:
            correct += 1
    return correct / len(actual)


def fit_prior_linear(records: Sequence[TimelineRecord]) -> tuple[float, float]:
    x = [r.timeline.initial_stance for r in records]
    y = [r.timeline.opinion_change or 0.0 for r in records]
    if len(x) < 2:
        return 0.0, sum(y) / len(y) if y else 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    denom = sum((v - mx) ** 2 for v in x)
    if denom == 0:
        return 0.0, my
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / denom
    return slope, my - slope * mx


def baseline_predictors(
    train_records: Sequence[TimelineRecord],
    extraction_cache: dict,
    reclass_cache: dict | None = None,
) -> dict[str, Callable[[TimelineRecord], float]]:
    """Return baseline functions that predict opinion change."""
    changes = [r.timeline.opinion_change or 0.0 for r in train_records]
    global_mean = sum(changes) / len(changes) if changes else 0.0

    topic_values: dict[str, list[float]] = defaultdict(list)
    for record in train_records:
        topic_values[record.timeline.topic].append(record.timeline.opinion_change or 0.0)
    topic_means = {
        topic: sum(vals) / len(vals)
        for topic, vals in topic_values.items()
    }

    slope, intercept = fit_prior_linear(train_records)

    train_pressures = [
        argument_pressure(r.timeline, extraction_cache, reclass_cache)[0]
        for r in train_records
    ]
    mean_abs_pressure = (
        sum(abs(p) for p in train_pressures) / len(train_pressures)
        if train_pressures
        else 1.0
    )
    mean_abs_change = (
        sum(abs(c) for c in changes) / len(changes)
        if changes
        else 0.0
    )
    pressure_scale = mean_abs_change / mean_abs_pressure if mean_abs_pressure else 0.0

    return {
        "baseline_no_change": lambda record: 0.0,
        "baseline_global_mean": lambda record: global_mean,
        "baseline_topic_mean": lambda record: topic_means.get(
            record.timeline.topic, global_mean
        ),
        "baseline_prior_linear": lambda record: (
            slope * record.timeline.initial_stance + intercept
        ),
        "baseline_raw_pressure": lambda record: (
            argument_pressure(record.timeline, extraction_cache, reclass_cache)[0]
            * pressure_scale
        ),
    }


def fit_linear_change_predictor(
    train_records: Sequence[TimelineRecord],
    feature_fn: Callable[[TimelineRecord], Sequence[float]],
    alpha: float = 1.0,
) -> tuple[Callable[[TimelineRecord], float], dict]:
    """Fit a small ridge linear model for opinion-change prediction.

    The intercept is not penalized. Features are intentionally supplied by the
    caller so fold-specific signals, such as held-out belief replay deltas, can
    be added without coupling this helper to the runner.
    """
    if not train_records:
        return lambda record: 0.0, {"intercept": 0.0, "coefficients": []}

    x = np.asarray([list(feature_fn(record)) for record in train_records], dtype=float)
    y = np.asarray(
        [record.timeline.opinion_change or 0.0 for record in train_records],
        dtype=float,
    )
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] == 0:
        mean_y = float(y.mean()) if len(y) else 0.0
        return lambda record: mean_y, {"intercept": mean_y, "coefficients": []}

    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)
    x_scaled = (x - feature_mean) / feature_std

    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    try:
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(design, y, rcond=None)[0]

    def predict(record: TimelineRecord) -> float:
        features = np.asarray(list(feature_fn(record)), dtype=float)
        features_scaled = (features - feature_mean) / feature_std
        return float(weights[0] + features_scaled @ weights[1:])

    diagnostics = {
        "intercept": float(weights[0]),
        "coefficients": [float(v) for v in weights[1:]],
        "feature_mean": [float(v) for v in feature_mean],
        "feature_std": [float(v) for v in feature_std],
        "alpha": alpha,
    }
    return predict, diagnostics


def prior_aware_combo_predictors(
    train_records: Sequence[TimelineRecord],
    extraction_cache: dict,
    belief_delta_by_worker: dict[str, float],
    reclass_cache: dict | None = None,
    alpha: float = 1.0,
) -> dict[str, tuple[Callable[[TimelineRecord], float], dict]]:
    """Fit fold-local linear baselines that test incremental belief value."""

    raw_pressure_cache: dict[str, float] = {}

    def raw_pressure(record: TimelineRecord) -> float:
        worker_id = str(record.timeline.worker_id)
        if worker_id not in raw_pressure_cache:
            raw_pressure_cache[worker_id] = argument_pressure(
                record.timeline, extraction_cache, reclass_cache
            )[0]
        return raw_pressure_cache[worker_id]

    def belief_delta(record: TimelineRecord) -> float:
        return float(belief_delta_by_worker.get(str(record.timeline.worker_id), 0.0))

    def participant_features(record: TimelineRecord) -> list[float]:
        return [
            1.0 if record.participant_type == "open" else 0.0,
            1.0 if record.participant_type == "leaning" else 0.0,
        ]

    feature_sets: dict[str, Callable[[TimelineRecord], Sequence[float]]] = {
        "baseline_prior_raw_pressure_linear": lambda record: [
            record.timeline.initial_stance,
            raw_pressure(record),
        ],
        "baseline_prior_belief_delta_linear": lambda record: [
            record.timeline.initial_stance,
            belief_delta(record),
        ],
        "baseline_prior_raw_belief_linear": lambda record: [
            record.timeline.initial_stance,
            raw_pressure(record),
            belief_delta(record),
        ],
        "baseline_prior_belief_type_linear": lambda record: [
            record.timeline.initial_stance,
            belief_delta(record),
            *participant_features(record),
            belief_delta(record) * (1.0 if record.participant_type == "open" else 0.0),
            belief_delta(record) * (1.0 if record.participant_type == "leaning" else 0.0),
            belief_delta(record) * (1.0 if record.participant_type == "committed" else 0.0),
        ],
    }

    fitted = {}
    for model_name, feature_fn in feature_sets.items():
        predictor, diagnostics = fit_linear_change_predictor(
            train_records, feature_fn, alpha=alpha
        )
        fitted[model_name] = (predictor, diagnostics)
    return fitted


def prediction_rows_for_model(
    records: Sequence[TimelineRecord],
    fold: int,
    model: str,
    predicted_change_fn: Callable[[TimelineRecord], float],
    params: dict | None = None,
) -> list[dict]:
    rows = []
    params = params or {}
    for record in records:
        timeline = record.timeline
        pred_change = predicted_change_fn(record)
        rows.append(
            {
                "fold": fold,
                "model": model,
                "worker_id": timeline.worker_id,
                "group_file": timeline.group_file,
                "topic": timeline.topic,
                "topic_category": timeline.topic_category,
                "participant_type": record.participant_type,
                "initial_stance": timeline.initial_stance,
                "final_stance_actual": timeline.final_stance,
                "opinion_change_actual": timeline.opinion_change,
                "opinion_change_predicted": pred_change,
                "final_stance_predicted": timeline.initial_stance + pred_change,
                "n_arguments": record.n_args,
                **params,
            }
        )
    return rows


def summarize_prediction_rows(rows: Sequence[dict], group_cols: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    summary = []
    group_key = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    for key, group in df.groupby(group_key, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        actual_change = group["opinion_change_actual"].tolist()
        pred_change = group["opinion_change_predicted"].tolist()
        actual_final = group["final_stance_actual"].tolist()
        pred_final = group["final_stance_predicted"].tolist()
        change_metrics = metric_dict(actual_change, pred_change)
        final_metrics = metric_dict(actual_final, pred_final)
        row = dict(zip(group_cols, key))
        row.update({f"change_{k}": v for k, v in change_metrics.items()})
        row.update({f"final_{k}": v for k, v in final_metrics.items()})
        summary.append(row)
    return pd.DataFrame(summary)


def group_dynamics(rows: Sequence[dict]) -> pd.DataFrame:
    """Compute actual/predicted group-level drift and convergence by group_file."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    out = []
    for (model, group_file), group in df.groupby(["model", "group_file"], dropna=False):
        if len(group) < 2:
            continue
        initial = group["initial_stance"]
        actual_final = group["final_stance_actual"]
        pred_final = group["final_stance_predicted"]
        out.append(
            {
                "model": model,
                "group_file": group_file,
                "topic": group["topic"].iloc[0],
                "topic_category": group["topic_category"].iloc[0],
                "n": len(group),
                "actual_group_drift": actual_final.mean() - initial.mean(),
                "predicted_group_drift": pred_final.mean() - initial.mean(),
                "drift_error": (pred_final.mean() - initial.mean())
                - (actual_final.mean() - initial.mean()),
                "initial_variance": initial.var(ddof=0),
                "actual_final_variance": actual_final.var(ddof=0),
                "predicted_final_variance": pred_final.var(ddof=0),
                "actual_convergence": initial.var(ddof=0) - actual_final.var(ddof=0),
                "predicted_convergence": initial.var(ddof=0) - pred_final.var(ddof=0),
            }
        )
    return pd.DataFrame(out)


def extraction_diagnostics(records: Sequence[TimelineRecord]) -> dict:
    n = len(records)
    if n == 0:
        return {"n_participants": 0}
    n_args = [r.n_args for r in records]
    types = defaultdict(int)
    topics = defaultdict(int)
    for r in records:
        types[r.participant_type] += 1
        topics[r.timeline.topic] += 1
    return {
        "n_participants": n,
        "n_zero_arg_participants": sum(1 for v in n_args if v == 0),
        "mean_args_per_participant": sum(n_args) / n,
        "min_args_per_participant": min(n_args),
        "max_args_per_participant": max(n_args),
        "participant_type_counts": dict(types),
        "n_topics": len(topics),
    }


def write_json(path: str, data: dict | list) -> None:
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_sanitize(data), f, indent=2, default=_json_default)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _json_sanitize(value):
    if isinstance(value, dict):
        return {k: _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    if hasattr(value, "item"):
        return _json_sanitize(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
