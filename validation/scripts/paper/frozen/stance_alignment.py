#!/usr/bin/env python3
"""Run and analyse a multi-topic internal-to-external stance alignment study.

The generator receives a programmatically assigned Bayesian Belief Engine
state through the production SubjectAgent prompt. The external judge receives
only the topic and generated response. Primary statistics are computed on
topic-by-stance cell means so repeated generations do not inflate n.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SCRIPT_DIR.parents[1]
RESULT_COLUMNS = [
    "task_id",
    "task_index",
    "topic",
    "topic_slug",
    "repeat",
    "requested_internal_stance",
    "internal_stance",
    "external_stance",
    "belief_description",
    "response",
    "generator_backend",
    "generator_model",
    "judge_backend",
    "judge_model",
    "generator_temperature",
    "judge_temperature",
    "generator_prompt_sha256",
    "judge_prompt_sha256",
    "started_at_utc",
    "finished_at_utc",
    "status",
    "error_type",
    "error_message",
]
PILOT_STANCES = [-1.0, -0.5, 0.0, 0.5, 1.0]
PROVENANCE_FILES = [
    "pyproject.toml",
    "src/agents/subject_agent.py",
    "src/agents/shared_instructions.py",
    "src/agents/evaluation_agent.py",
    "src/backend/backend_openai.py",
    "src/backend/factory.py",
    "src/simulation/belief_engine.py",
    "src/simulation/belief_instructions.json",
    "src/prompts/evaluator_optimized.txt",
]
TOPIC_SHORT_LABELS = {
    "Social media brings more harm than good": "Social media",
    "Homeopathy brings more harm than good": "Homeopathy",
    "Entrapment should be legalized": "Entrapment",
    "We should adopt a zero-tolerance policy in schools": "Zero-tolerance schools",
    "We should introduce compulsory voting": "Compulsory voting",
    "We should adopt an austerity regime": "Austerity",
    "We should legalize sex selection": "Sex selection",
    "We should adopt atheism": "Atheism",
    "We should subsidize journalism": "Journalism",
    "We should adopt libertarianism": "Libertarianism",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    task_index: int
    topic: str
    topic_slug: str
    repeat: int
    requested_internal_stance: float


class ControlledMemory:
    """Minimal memory facade that isolates the state-to-text link.

    SubjectAgent receives the real BayesianBeliefEngine and its production
    response prompt, but no retrieved evidence or conversation history.
    """

    stance_method = "bayesian"

    def __init__(self, topic: str, belief_engine: Any):
        self.belief_engine = belief_engine
        self.arguments = {"topic_marker": SimpleNamespace(topic_id=topic)}

    def get_stance(self, **_: Any) -> float:
        return float(self.belief_engine.get_belief_score())

    def retrieve(self, **_: Any) -> list[Any]:
        return []

    def close(self) -> None:
        return None


class CapturingBackend:
    """Per-task proxy that records the exact generation system prompt."""

    def __init__(self, backend: Any):
        self.backend = backend
        self.last_system_prompt = ""

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        self.last_system_prompt = system_prompt or ""
        return self.backend.chat(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        return self.backend.get_completion(
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def slugify(value: str) -> str:
    chars = [c.lower() if c.isalnum() else "_" for c in value.strip()]
    return "_".join(part for part in "".join(chars).split("_") if part)


def find_agent_mem_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("AGENT_MEM_ROOT"):
        candidates.append(Path(os.environ["AGENT_MEM_ROOT"]).expanduser())
    for parent in [Path.cwd(), *SCRIPT_DIR.parents]:
        candidates.extend([parent, parent / "agent_mem", parent.parent / "agent_mem"])

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "src/simulation/belief_engine.py").exists():
            return resolved
    raise FileNotFoundError(
        "Could not locate the agent_mem repository. Pass --agent-mem-root or "
        "set AGENT_MEM_ROOT."
    )


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def backend_label(config: dict[str, Any]) -> tuple[str, str]:
    return str(config.get("type", "openai")), str(config.get("model", ""))


def git_provenance(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "repo": str(repo),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": sha256_text(status),
    }


def make_tasks(
    topics: Sequence[str],
    stances: Sequence[float],
    repeats: int,
    seed: int,
    randomize: bool,
) -> list[Task]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if len(set(topics)) != len(topics):
        raise ValueError("topics must be unique")
    if len(set(float(x) for x in stances)) != len(stances):
        raise ValueError("internal stances must be unique")
    if any(not -1.0 <= float(x) <= 1.0 for x in stances):
        raise ValueError("internal stances must lie in [-1, 1]")

    raw: list[tuple[str, str, int, float]] = []
    for topic in topics:
        for stance in stances:
            for repeat in range(1, repeats + 1):
                raw.append((topic, slugify(topic), repeat, float(stance)))
    if randomize:
        random.Random(seed).shuffle(raw)

    tasks = []
    for index, (topic, topic_slug, repeat, stance) in enumerate(raw, start=1):
        key = f"{topic}|{stance:+.6f}|{repeat}"
        tasks.append(
            Task(
                task_id=sha256_text(key)[:16],
                task_index=index,
                topic=topic,
                topic_slug=topic_slug,
                repeat=repeat,
                requested_internal_stance=stance,
            )
        )
    return tasks


def ensure_result_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writeheader()


def completed_task_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    if "status" not in frame or "task_id" not in frame:
        return set()
    return set(frame.loc[frame["status"] == "ok", "task_id"].astype(str))


def append_result(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    normalized = {column: row.get(column, "") for column in RESULT_COLUMNS}
    with lock:
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writerow(normalized)
            handle.flush()


def select_topics(config_topics: list[str], raw_selection: str | None) -> list[str]:
    if not raw_selection:
        return config_topics
    requested = [item.strip() for item in raw_selection.split(",") if item.strip()]
    by_slug = {slugify(topic): topic for topic in config_topics}
    selected: list[str] = []
    for item in requested:
        if item in config_topics:
            selected.append(item)
        elif item in by_slug:
            selected.append(by_slug[item])
        else:
            raise ValueError(f"Unknown topic or topic slug: {item}")
    return selected


def parse_stances(raw: str | None, configured: list[float]) -> list[float]:
    if not raw:
        return [float(value) for value in configured]
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def run_task(
    task: Task,
    generation_backend: Any,
    evaluator: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    started_at = utc_now()
    gen_conf = config["generation"]
    eval_conf = config["evaluation"]
    gen_backend_name, gen_model = backend_label(gen_conf["backend"])
    judge_backend_name, judge_model = backend_label(eval_conf["backend"])
    gen_temperature = float(gen_conf.get("temperature", 0.7))
    judge_temperature = float(eval_conf.get("temperature", 0.0))

    base = {
        **asdict(task),
        "generator_backend": gen_backend_name,
        "generator_model": gen_model,
        "judge_backend": judge_backend_name,
        "judge_model": judge_model,
        "generator_temperature": gen_temperature,
        "judge_temperature": judge_temperature,
        "started_at_utc": started_at,
    }

    try:
        from src.agents.subject_agent import SubjectAgent
        from src.simulation.belief_engine import BayesianBeliefEngine

        with tempfile.TemporaryDirectory(prefix="be_stance_alignment_") as temp_dir:
            engine = BayesianBeliefEngine(
                agent_name="ControlledSubject",
                memory_dir=temp_dir,
                config={
                    "update_sensitivity": 0.4,
                    "anchor_sensitivity": 0.4,
                    "confirmation_bias": 0.0,
                },
            )
            engine.set_belief_score(task.requested_internal_stance, persist=False)
            memory = ControlledMemory(task.topic, engine)
            capturing_backend = CapturingBackend(generation_backend)
            agent = SubjectAgent(
                backend=capturing_backend,
                memory=memory,
                persona=str(config["study"]["persona"]),
                name="Subject Agent",
                temperature=gen_temperature,
            )
            response, _ = agent.generate_reply([])
            internal_stance = float(engine.get_belief_score())
            belief_description = engine.get_belief_description()
            generator_prompt_hash = sha256_text(capturing_backend.last_system_prompt)

        evaluation = evaluator.evaluate_stance(
            transcript=response,
            memory_snapshot="",
            topic=task.topic,
            agent_name="Subject Agent",
        )
        external_stance = float(evaluation["stance_score"])
        if not math.isfinite(external_stance) or not -1.0 <= external_stance <= 1.0:
            raise ValueError(f"Judge returned out-of-range stance: {external_stance}")

        return {
            **base,
            "internal_stance": internal_stance,
            "external_stance": external_stance,
            "belief_description": belief_description,
            "response": response,
            "generator_prompt_sha256": generator_prompt_hash,
            "judge_prompt_sha256": sha256_text(
                evaluator.default_stance_prompt_template
            ),
            "finished_at_utc": utc_now(),
            "status": "ok",
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:
        return {
            **base,
            "internal_stance": "",
            "external_stance": "",
            "belief_description": "",
            "response": "",
            "generator_prompt_sha256": "",
            "judge_prompt_sha256": sha256_text(
                getattr(evaluator, "default_stance_prompt_template", "")
            ),
            "finished_at_utc": utc_now(),
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
        }


def safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
        return float("nan"), float("nan")
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
        return float("nan"), float("nan")
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite_indices = np.where(np.isfinite(values))[0]
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, values[index] * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def metric_row(
    topic: str,
    frame: pd.DataFrame,
    sign_threshold: float,
) -> dict[str, Any]:
    x = frame["internal_stance"].to_numpy(dtype=float)
    y = frame["external_stance_mean"].to_numpy(dtype=float)
    pearson_r, pearson_p = safe_pearson(x, y)
    spearman_r, spearman_p = safe_spearman(x, y)
    regression = stats.linregress(x, y) if len(x) >= 2 else None
    residual = y - x
    directional = np.abs(x) >= sign_threshold
    sign_accuracy = (
        float(np.mean(np.sign(x[directional]) == np.sign(y[directional])))
        if directional.any()
        else float("nan")
    )
    neutral = np.abs(x) < sign_threshold
    return {
        "topic": topic,
        "n_positions": int(len(frame)),
        "n_raw_rows": int(frame["n_repeats"].sum()),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
        "slope": float(regression.slope) if regression else float("nan"),
        "intercept": float(regression.intercept) if regression else float("nan"),
        "r_squared": float(regression.rvalue**2) if regression else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias_external_minus_internal": float(np.mean(residual)),
        "sign_accuracy": sign_accuracy,
        "neutral_external_abs_mean": (
            float(np.mean(np.abs(y[neutral]))) if neutral.any() else float("nan")
        ),
        "mean_within_cell_sd": float(
            frame["external_stance_sd"].fillna(0.0).mean()
        ),
    }


def cluster_bootstrap(
    cells: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    topics = list(cells["topic"].drop_duplicates())
    if len(topics) < 2 or samples < 1:
        return {"pearson_r_ci95": [float("nan"), float("nan")], "slope_ci95": [float("nan"), float("nan")]}
    grouped = {topic: cells.loc[cells["topic"] == topic] for topic in topics}
    rng = np.random.default_rng(seed)
    correlations: list[float] = []
    slopes: list[float] = []
    for _ in range(samples):
        chosen = rng.choice(topics, size=len(topics), replace=True)
        sample = pd.concat([grouped[str(topic)] for topic in chosen], ignore_index=True)
        x = sample["internal_stance"].to_numpy(dtype=float)
        y = sample["external_stance_mean"].to_numpy(dtype=float)
        correlation, _ = safe_pearson(x, y)
        if math.isfinite(correlation):
            correlations.append(correlation)
            slopes.append(float(stats.linregress(x, y).slope))
    return {
        "pearson_r_ci95": np.quantile(correlations, [0.025, 0.975]).tolist(),
        "slope_ci95": np.quantile(slopes, [0.025, 0.975]).tolist(),
    }


def analyse_results(
    results_csv: Path,
    output_dir: Path,
    analysis_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    frame = pd.read_csv(results_csv)
    required = {"topic", "internal_stance", "external_stance", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Results CSV is missing columns: {sorted(missing)}")

    valid = frame.loc[frame["status"] == "ok"].copy()
    if valid.empty:
        raise ValueError("No successful rows are available for analysis")
    valid["internal_stance"] = pd.to_numeric(valid["internal_stance"], errors="raise")
    valid["external_stance"] = pd.to_numeric(valid["external_stance"], errors="raise")

    cells = (
        valid.groupby(["topic", "topic_slug", "internal_stance"], as_index=False)
        .agg(
            external_stance_mean=("external_stance", "mean"),
            external_stance_sd=("external_stance", "std"),
            n_repeats=("external_stance", "size"),
        )
        .sort_values(["topic", "internal_stance"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output_dir / "cell_means.csv", index=False)

    sign_threshold = float(analysis_config.get("sign_threshold", 0.1))
    topic_rows = [
        metric_row(topic, topic_frame, sign_threshold)
        for topic, topic_frame in cells.groupby("topic", sort=False)
    ]
    topic_metrics = pd.DataFrame(topic_rows)
    topic_metrics["pearson_p_holm"] = holm_adjust(topic_metrics["pearson_p"])
    topic_metrics.to_csv(output_dir / "topic_metrics.csv", index=False)

    x = cells["internal_stance"].to_numpy(dtype=float)
    y = cells["external_stance_mean"].to_numpy(dtype=float)
    pooled_r, pooled_p = safe_pearson(x, y)
    pooled_slope = float(stats.linregress(x, y).slope)
    pooled_intercept = float(stats.linregress(x, y).intercept)

    centered = cells.copy()
    centered["x_centered"] = centered["internal_stance"] - centered.groupby("topic")[
        "internal_stance"
    ].transform("mean")
    centered["y_centered"] = centered["external_stance_mean"] - centered.groupby(
        "topic"
    )["external_stance_mean"].transform("mean")
    within_r, within_p = safe_pearson(
        centered["x_centered"].to_numpy(dtype=float),
        centered["y_centered"].to_numpy(dtype=float),
    )
    within_slope = float(
        np.dot(centered["x_centered"], centered["y_centered"])
        / np.dot(centered["x_centered"], centered["x_centered"])
    )

    finite_r = topic_metrics["pearson_r"].dropna().clip(-0.999999, 0.999999)
    macro_fisher_r = float(np.tanh(np.mean(np.arctanh(finite_r))))
    threshold = float(analysis_config.get("primary_correlation_threshold", 0.8))
    required_fraction = float(analysis_config.get("required_topic_fraction", 0.8))
    passing_topics = int((topic_metrics["pearson_r"] >= threshold).sum())
    topic_fraction = passing_topics / len(topic_metrics)

    leave_one_out: list[dict[str, Any]] = []
    for held_out in cells["topic"].drop_duplicates():
        subset = cells.loc[cells["topic"] != held_out]
        value, _ = safe_pearson(
            subset["internal_stance"].to_numpy(dtype=float),
            subset["external_stance_mean"].to_numpy(dtype=float),
        )
        leave_one_out.append({"held_out_topic": held_out, "pooled_pearson_r": value})

    bootstrap = cluster_bootstrap(
        cells,
        samples=int(analysis_config.get("cluster_bootstrap_samples", 5000)),
        seed=seed,
    )
    residual = y - x
    metrics = {
        "analysis_unit": "topic-by-internal-stance cell mean",
        "n_topics": int(cells["topic"].nunique()),
        "n_cells": int(len(cells)),
        "n_successful_raw_rows": int(len(valid)),
        "n_error_rows": int((frame["status"] != "ok").sum()),
        "pooled": {
            "pearson_r": pooled_r,
            "pearson_p": pooled_p,
            "slope": pooled_slope,
            "intercept": pooled_intercept,
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "bias_external_minus_internal": float(np.mean(residual)),
            **bootstrap,
        },
        "within_topic_centered": {
            "pearson_r": within_r,
            "pearson_p": within_p,
            "common_slope": within_slope,
        },
        "macro": {
            "fisher_z_mean_pearson_r": macro_fisher_r,
            "median_topic_pearson_r": float(topic_metrics["pearson_r"].median()),
            "min_topic_pearson_r": float(topic_metrics["pearson_r"].min()),
            "median_topic_slope": float(topic_metrics["slope"].median()),
            "min_topic_sign_accuracy": float(topic_metrics["sign_accuracy"].min()),
        },
        "generality_gate": {
            "topic_correlation_threshold": threshold,
            "required_topic_fraction": required_fraction,
            "passing_topics": passing_topics,
            "observed_topic_fraction": topic_fraction,
            "passed": bool(topic_fraction >= required_fraction),
        },
        "leave_one_topic_out": leave_one_out,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir / "summary.md", metrics, topic_metrics)
    plot_results(cells, topic_metrics, output_dir)
    return metrics


def format_metric(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "NA"


def write_summary(
    path: Path,
    metrics: dict[str, Any],
    topic_metrics: pd.DataFrame,
) -> None:
    pooled = metrics["pooled"]
    within = metrics["within_topic_centered"]
    macro = metrics["macro"]
    gate = metrics["generality_gate"]
    lines = [
        "# Multi-topic internal-external stance alignment",
        "",
        "Primary statistics use topic-by-internal-stance cell means; repeats estimate",
        "generation variability and do not inflate the correlation sample size.",
        "",
        "## Headline results",
        "",
        f"- Topics: {metrics['n_topics']}",
        f"- Successful generated/judged responses: {metrics['n_successful_raw_rows']}",
        f"- Pooled Pearson r: {format_metric(pooled['pearson_r'])} "
        f"(topic-clustered 95% CI {format_metric(pooled['pearson_r_ci95'][0])} to "
        f"{format_metric(pooled['pearson_r_ci95'][1])})",
        f"- Pooled OLS slope: {format_metric(pooled['slope'])} "
        f"(topic-clustered 95% CI {format_metric(pooled['slope_ci95'][0])} to "
        f"{format_metric(pooled['slope_ci95'][1])})",
        f"- Within-topic-centered Pearson r: {format_metric(within['pearson_r'])}",
        f"- Fisher-z macro-average topic r: {format_metric(macro['fisher_z_mean_pearson_r'])}",
        f"- Pooled MAE: {format_metric(pooled['mae'])}; RMSE: {format_metric(pooled['rmse'])}",
        f"- Generality gate: {'PASS' if gate['passed'] else 'FAIL'} "
        f"({gate['passing_topics']}/{metrics['n_topics']} topics at r >= "
        f"{gate['topic_correlation_threshold']:.2f})",
        "",
        "## Per-topic results",
        "",
        "| Topic | n positions | Pearson r | Slope | MAE | Bias (external-internal) | Sign accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in topic_metrics.itertuples(index=False):
        lines.append(
            f"| {row.topic} | {row.n_positions} | {format_metric(row.pearson_r)} | "
            f"{format_metric(row.slope)} | {format_metric(row.mae)} | "
            f"{format_metric(row.bias_external_minus_internal)} | "
            f"{format_metric(row.sign_accuracy)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This test measures whether a controlled scalar Belief Engine state is",
            "expressed in generated text and recovered by the frozen automated judge",
            "across the listed propositions. It is not a human-perception study and does",
            "not validate whether the scalar state is a faithful model of human belief.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(
    cells: pd.DataFrame,
    topic_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    matplotlib_cache = Path(tempfile.gettempdir()) / "belief_engine_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text_color = "#293241"
    teal = "#19747e"
    teal_light = "#9fc9ce"
    red = "#b34a4a"
    gold = "#b49b3f"
    reference_gray = "#9ca3af"
    grid_color = "#e5e7eb"
    spine_color = "#d1d5db"
    neutral_band = "#f3f4f6"
    font_stack = [
        "Inter",
        "Avenir Next",
        "Helvetica Neue",
        "Arial",
        "DejaVu Sans",
    ]
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": font_stack,
            "text.color": text_color,
            "axes.labelcolor": text_color,
            "axes.titlecolor": text_color,
            "xtick.color": text_color,
            "ytick.color": text_color,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11.6,
            "axes.titlesize": 10.4,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "font.size": 10.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )

    available_topics = list(cells["topic"].drop_duplicates())
    topics = [topic for topic in TOPIC_SHORT_LABELS if topic in available_topics]
    topics.extend(topic for topic in available_topics if topic not in topics)
    columns = 2
    rows = math.ceil(len(topics) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.45, 1.86 * rows),
        sharex=True,
        sharey=True,
    )
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, topic in zip(flat_axes, topics):
        subset = cells.loc[cells["topic"] == topic]
        metric = topic_metrics.loc[topic_metrics["topic"] == topic].iloc[0]
        axis.axhspan(
            -0.1,
            0.1,
            color=neutral_band,
            alpha=1.0,
            linewidth=0,
            zorder=0,
        )
        axis.errorbar(
            subset["internal_stance"],
            subset["external_stance_mean"],
            yerr=subset["external_stance_sd"].fillna(0.0),
            fmt="o",
            markersize=4.2,
            markerfacecolor=teal,
            markeredgecolor="white",
            markeredgewidth=0.45,
            alpha=0.96,
            color=teal,
            ecolor=teal_light,
            elinewidth=0.85,
            capsize=1.8,
            zorder=4,
        )
        grid = np.linspace(-1.0, 1.0, 101)
        axis.plot(
            grid,
            grid,
            linestyle="--",
            color=reference_gray,
            linewidth=0.85,
            alpha=0.85,
            zorder=1,
        )
        axis.plot(
            grid,
            metric["intercept"] + metric["slope"] * grid,
            color=red,
            linewidth=1.9,
            zorder=3,
        )
        axis.axhline(
            0,
            color=reference_gray,
            linestyle="--",
            linewidth=0.8,
            alpha=0.75,
            zorder=1,
        )
        axis.axvline(0, color=grid_color, linewidth=0.7, zorder=1)
        axis.grid(axis="y", color=grid_color, linewidth=0.65, alpha=0.78)
        axis.grid(axis="x", color=grid_color, linewidth=0.5, alpha=0.50)
        axis.set_axisbelow(True)
        for spine in ["left", "bottom"]:
            axis.spines[spine].set_color(spine_color)
            axis.spines[spine].set_linewidth(0.8)
        axis.tick_params(axis="both", width=0.8, length=3.2)
        axis.set_title(
            f"{TOPIC_SHORT_LABELS.get(topic, topic)}\n"
            f"r={metric['pearson_r']:.3f}  |  slope={metric['slope']:.3f}",
            fontsize=10.4,
            fontweight=750,
            color=text_color,
            pad=4,
            linespacing=0.94,
        )
        axis.set_xlim(-1.04, 1.04)
        axis.set_ylim(-1.04, 1.04)
        axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    for axis in flat_axes[len(topics) :]:
        axis.set_visible(False)
    figure.supxlabel(
        "Belief Engine internal stance",
        fontsize=12.0,
        fontweight=700,
        color=text_color,
        y=0.015,
    )
    figure.supylabel(
        "External judged-text stance",
        fontsize=12.0,
        fontweight=700,
        color=text_color,
        x=0.012,
    )
    figure.suptitle(
        "Internal-external stance alignment by topic",
        fontsize=15.5,
        fontweight=800,
        color=text_color,
        y=0.992,
    )
    figure.subplots_adjust(
        left=0.115,
        right=0.985,
        bottom=0.065,
        top=0.925,
        hspace=0.53,
        wspace=0.16,
    )
    figure.savefig(output_dir / "topic_alignment.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "topic_alignment.pdf", bbox_inches="tight")
    plt.close(figure)

    ordered = topic_metrics.sort_values("pearson_r")
    display_labels = [
        TOPIC_SHORT_LABELS.get(topic, topic) for topic in ordered["topic"]
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.25, 5.15))
    y_positions = np.arange(len(ordered))
    axes[0].barh(
        y_positions,
        ordered["pearson_r"],
        color=teal,
        height=0.66,
        zorder=3,
    )
    axes[0].axvline(0.8, color=red, linestyle="--", linewidth=1.2, alpha=0.9)
    axes[0].set_yticks(y_positions, display_labels)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Pearson r")
    axes[0].set_title("Correlation", fontweight=750, fontsize=13.2, pad=7)
    axes[1].barh(
        y_positions,
        ordered["slope"],
        color=gold,
        height=0.66,
        zorder=3,
    )
    axes[1].axvline(
        1.0,
        color=reference_gray,
        linestyle="--",
        linewidth=1.2,
        alpha=0.95,
    )
    axes[1].set_yticks(y_positions, [""] * len(ordered))
    axes[1].set_xlim(0, 1.03)
    axes[1].set_xlabel("OLS slope")
    axes[1].set_title("Calibration slope", fontweight=750, fontsize=13.2, pad=7)
    for axis in axes:
        axis.grid(axis="x", color=grid_color, linewidth=0.65, alpha=0.78)
        axis.set_axisbelow(True)
        for spine in ["left", "bottom"]:
            axis.spines[spine].set_color(spine_color)
            axis.spines[spine].set_linewidth(0.8)
        axis.tick_params(axis="both", width=0.8, length=3.2)
    axes[0].bar_label(
        axes[0].containers[0],
        labels=[f"{value:.3f}" for value in ordered["pearson_r"]],
        padding=-35,
        color="white",
        fontsize=8.8,
        fontweight=700,
    )
    axes[1].bar_label(
        axes[1].containers[0],
        labels=[f"{value:.3f}" for value in ordered["slope"]],
        padding=-35,
        color="white",
        fontsize=8.8,
        fontweight=700,
    )
    figure.suptitle(
        "Topic-level alignment and calibration",
        fontsize=15.5,
        fontweight=800,
        color=text_color,
        y=0.985,
    )
    figure.subplots_adjust(left=0.20, right=0.985, bottom=0.12, top=0.87, wspace=0.16)
    figure.savefig(output_dir / "topic_metric_summary.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "topic_metric_summary.pdf", bbox_inches="tight")
    plt.close(figure)

    reference_paths = [
        PAPER_ROOT / "figures/ex_judge.pdf",
        PAPER_ROOT / "figures/sweep_be_agent.pdf",
        PAPER_ROOT / "figures/parameter_alignment.pdf",
    ]
    style_metadata = {
        "style_name": "selected_neurips_2026_figures",
        "font_stack": font_stack,
        "pdf_fonttype": 42,
        "colors": {
            "text": text_color,
            "teal": teal,
            "red": red,
            "gold": gold,
            "grid": grid_color,
            "spine": spine_color,
            "reference": reference_gray,
        },
        "reference_figure_sha256": {
            str(path.relative_to(PAPER_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in reference_paths
            if path.exists()
        },
        "figure_runner_sha256": sha256_text(Path(__file__).read_text(encoding="utf-8")),
    }
    (output_dir / "figure_style_metadata.json").write_text(
        json.dumps(style_metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def write_run_metadata(
    output_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    agent_mem_root: Path,
    tasks: Sequence[Task],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = snapshot_agent_mem_sources(output_dir, agent_mem_root)

    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    metadata = {
        "created_at_utc": utc_now(),
        "source_config": str(config_path.resolve()),
        "source_config_sha256": sha256_text(config_path.read_text(encoding="utf-8")),
        "task_count": len(tasks),
        "generation_calls_planned": len(tasks),
        "judge_calls_planned": len(tasks),
        "agent_mem": git_provenance(agent_mem_root),
        "snapshotted_agent_mem_files": source_hashes,
        "runner_sha256": sha256_text(Path(__file__).read_text(encoding="utf-8")),
        "python": sys.version,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def snapshot_agent_mem_sources(
    output_dir: Path,
    agent_mem_root: Path,
) -> dict[str, str]:
    """Copy the exact source inputs needed to reproduce prompt construction."""
    provenance_dir = output_dir / "provenance"
    source_hashes: dict[str, str] = {}
    for relative_path in PROVENANCE_FILES:
        source = agent_mem_root / relative_path
        if not source.exists():
            continue
        destination = provenance_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative_path] = hashlib.sha256(source.read_bytes()).hexdigest()
    return source_hashes


def add_provenance_to_existing_run(output_dir: Path, agent_mem_root: Path) -> None:
    """Snapshot source files without changing the historical runner hash."""
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["snapshotted_agent_mem_files"] = snapshot_agent_mem_sources(
        output_dir, agent_mem_root
    )
    metadata["provenance_snapshotted_at_utc"] = utc_now()
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def create_backends(config: dict[str, Any], agent_mem_root: Path) -> tuple[Any, Any]:
    if str(agent_mem_root) not in sys.path:
        sys.path.insert(0, str(agent_mem_root))
    try:
        from dotenv import load_dotenv

        load_dotenv(agent_mem_root / ".env", override=False)
    except Exception:
        pass

    from src.agents.evaluation_agent import EvaluationAgent
    from src.backend.factory import create_backend

    generation_backend = create_backend({"backend": config["generation"]["backend"]})
    judge_backend = create_backend({"backend": config["evaluation"]["backend"]})
    evaluator = EvaluationAgent(
        judge_backend,
        temperature=float(config["evaluation"].get("temperature", 0.0)),
    )
    return generation_backend, evaluator


def synthetic_self_test(output_dir: Path) -> None:
    rng = np.random.default_rng(42)
    rows = []
    topics = ["Synthetic topic A", "Synthetic topic B", "Synthetic topic C"]
    for topic_index, topic in enumerate(topics):
        for stance in np.linspace(-1, 1, 9):
            for repeat in range(1, 4):
                external = 0.84 * stance + (topic_index - 1) * 0.04 + rng.normal(0, 0.025)
                rows.append(
                    {
                        "topic": topic,
                        "topic_slug": slugify(topic),
                        "internal_stance": stance,
                        "external_stance": external,
                        "status": "ok",
                        "repeat": repeat,
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "synthetic_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    metrics = analyse_results(
        results_path,
        output_dir,
        {
            "sign_threshold": 0.1,
            "primary_correlation_threshold": 0.8,
            "required_topic_fraction": 0.8,
            "cluster_bootstrap_samples": 200,
        },
        seed=42,
    )
    assert metrics["n_topics"] == 3
    assert metrics["pooled"]["pearson_r"] > 0.98
    assert 0.75 < metrics["pooled"]["slope"] < 0.92
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "topic_alignment.png").exists()
    print(f"Self-test passed: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--agent-mem-root")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--topics", help="Comma-separated full topics or slugs")
    parser.add_argument("--stances", help="Comma-separated stance values")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use the first three selected topics, five stances, and one repeat",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyse-only", type=Path, metavar="RESULTS_CSV")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--snapshot-provenance-only",
        type=Path,
        metavar="RUN_DIR",
        help="Add current agent_mem source snapshots to an existing run",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.snapshot_provenance_only:
        agent_mem_root = find_agent_mem_root(args.agent_mem_root)
        output_dir = args.snapshot_provenance_only.resolve()
        add_provenance_to_existing_run(output_dir, agent_mem_root)
        print(f"Provenance snapshot added: {output_dir}")
        return 0
    if args.self_test:
        target = args.output_dir or Path(tempfile.mkdtemp(prefix="stance_alignment_selftest_"))
        synthetic_self_test(target)
        return 0

    config_path = args.config.resolve()
    config = load_config(config_path)
    study = config["study"]
    topics = select_topics(list(study["topics"]), args.topics)
    stances = parse_stances(args.stances, list(study["internal_stances"]))
    repeats = int(args.repeats or study["repeats_per_cell"])
    if args.pilot:
        topics = topics[:3]
        stances = PILOT_STANCES
        repeats = 1

    if args.analyse_only:
        output_dir = args.output_dir or args.analyse_only.resolve().parent / "analysis"
        analyse_results(
            args.analyse_only.resolve(),
            output_dir.resolve(),
            config.get("analysis", {}),
            seed=int(study["random_seed"]),
        )
        print(f"Analysis written to {output_dir.resolve()}")
        return 0

    tasks = make_tasks(
        topics=topics,
        stances=stances,
        repeats=repeats,
        seed=int(study["random_seed"]),
        randomize=bool(study.get("randomize_order", True)),
    )
    workers = int(args.workers or config.get("execution", {}).get("workers", 1))
    if workers < 1:
        raise ValueError("workers must be at least 1")

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = PAPER_ROOT / config["execution"]["output_root"]
        output_dir = (output_root / stamp).resolve()

    print(
        f"Planned study: {len(topics)} topics x {len(stances)} stances x "
        f"{repeats} repeats = {len(tasks)} rows ({2 * len(tasks)} LLM calls)."
    )
    print(f"Output: {output_dir}")
    if args.dry_run:
        return 0

    agent_mem_root = find_agent_mem_root(args.agent_mem_root)
    write_run_metadata(output_dir, config, config_path, agent_mem_root, tasks)
    results_path = output_dir / "results.csv"
    ensure_result_file(results_path)
    completed = completed_task_ids(results_path)
    pending = [task for task in tasks if task.task_id not in completed]
    if completed:
        print(f"Resuming: {len(completed)} completed, {len(pending)} pending.")

    generation_backend, evaluator = create_backends(config, agent_mem_root)
    write_lock = threading.Lock()
    successful = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_task, task, generation_backend, evaluator, config): task
            for task in pending
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            append_result(results_path, row, write_lock)
            if row["status"] == "ok":
                successful += 1
            else:
                errors += 1
                print(
                    f"ERROR {row['topic_slug']} stance="
                    f"{row['requested_internal_stance']}: {row['error_type']} "
                    f"{row['error_message']}",
                    file=sys.stderr,
                )
            if completed_count == 1 or completed_count % 10 == 0 or completed_count == len(pending):
                print(
                    f"Progress {completed_count}/{len(pending)} "
                    f"(ok={successful}, errors={errors})"
                )

    analyse_results(
        results_path,
        output_dir / "analysis",
        config.get("analysis", {}),
        seed=int(study["random_seed"]),
    )
    print(f"Completed. Results and analysis: {output_dir}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
