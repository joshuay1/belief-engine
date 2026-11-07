#!/usr/bin/env python3
"""Run a multi-model two-agent comparison.

The experiment compares multiple two-agent debate conditions. Each condition can
choose its own model, backend, and agent mode, so vanilla LLM rows do not need
to be paired with the same model as any Belief Engine row. Every summary,
trajectory artifact, and topic-grid figure uses the selected measurement source.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


TOPICS = [
    "Social media brings more harm than good",
    "Homeopathy brings more harm than good",
    "Entrapment should be legalized",
    "We should adopt a zero-tolerance policy in schools",
    "We should introduce compulsory voting",
    "We should adopt an austerity regime",
    "We should legalize sex selection",
    "We should adopt atheism",
    "We should subsidize journalism",
    "We should adopt libertarianism",
]

SHORT_TOPIC_LABELS = {
    "Social media brings more harm than good": "social_media",
    "Homeopathy brings more harm than good": "homeopathy",
    "Entrapment should be legalized": "entrapment",
    "We should adopt a zero-tolerance policy in schools": "zero_tolerance",
    "We should introduce compulsory voting": "compulsory_voting",
    "We should adopt an austerity regime": "austerity",
    "We should legalize sex selection": "sex_selection",
    "We should adopt atheism": "atheism",
    "We should subsidize journalism": "journalism",
    "We should adopt libertarianism": "libertarianism",
}

INITIAL_STANCE = 0.75
DEFAULT_ROUNDS = 15
DEFAULT_TRIALS = 3
BELIEF_SEMANTICS_MODES = (
    "incremental_v1",
    "active_seed_v1",
    "active_explicit_prior_v1",
)
RESPONSE_SHELLS = ("legacy", "matched_v1")
SCHEDULE_ORDERS = ("condition-major", "paired")
SPEAKER_ORDERS = ("pro-first", "con-first")
EXTRACTION_OUTPUT_MODES = (
    "legacy_text_v1",
    "structured_json_v1",
    "whole_message_json_v1",
)
EXTRACTION_AUDIT_OUTPUT_MODES = frozenset(
    {
        "structured_json_v1",
        "whole_message_json_v1",
    }
)

CONTROLLED_DEBATER_PERSONA = (
    "You are a careful, concise debater. Defend your assigned side, engage "
    "directly with the opposing side's arguments, and stay focused on the topic."
)

OPEN_MINDED_MEMORY = {
    "update_sensitivity": 0.40,
    "anchor_sensitivity": 0.20,
    "allow_self_arguments": True,
    "confirmation_bias": 0.0,
    "worldview_openness": "balanced",
    "argument_similarity_threshold": 0.60,
    "self_similarity_threshold": 0.45,
    "cross_polarity_dedup": False,
    "deduplication_enabled": True,
    "opinion_change_style": "human_like",
    "response_retrieval_enabled": True,
    "dynamic_evidence_horizon": "persistent",
    "retrieved_evidence_token_budget": None,
    "retrieved_evidence_encoding": "cl100k_base",
    "strength_estimation_method": "classifier",
    "uniform_strength_value": None,
    "classifier_model_path": "models/argument_quality_classifier",
}

VANILLA_DEFINITION = (
    "Vanilla LLM condition: VanillaSubjectAgent for both sides, using the "
    "configured topic, side assignment, seeding limit, backend, and model, but "
    "with no memory store, no retrieval, and no belief update. Stance is "
    "measured by the shared external judge."
)

PROMPT_SELF_UPDATE_DEFINITION = (
    "Prompt self-update baseline: each agent keeps a self-reported stance by "
    "asking the LLM to update after opponent turns. It has no Belief Engine and "
    "no memory retrieval."
)

RAG_MEMORY_DEFINITION = (
    "RAG-only baseline: agents receive the same extracted argument memory and "
    "retrieval budget as the Belief Engine condition, but use blind memory with "
    "no Bayesian belief update."
)

RAG_SELF_UPDATE_DEFINITION = (
    "RAG + self-update baseline: agents receive the same retrieved argument "
    "memory as the Belief Engine condition and ask the LLM to self-report stance "
    "after opponent turns, but use no Bayesian belief update."
)

BLIND_RAG_MEMORY = {
    **OPEN_MINDED_MEMORY,
    "stance_method": "blind",
    "belief_engine_type": "bayesian",
    "update_sensitivity": 0.0,
    "anchor_sensitivity": 0.0,
}

# Backwards-compatible name for scripts that imported the old runner.
BASELINE_DEFINITION = VANILLA_DEFINITION


class NoOpExtractor:
    """Carries topic metadata for judged stance without spending extraction calls."""

    def __init__(self, topic_id: str):
        self.topic_id = topic_id

    def extract(self, text: str, source_role: str, expected_polarity: str = None) -> List[Any]:
        del text, source_role, expected_polarity
        return []


def build_evaluator(config: Dict[str, Any], generation_backend: Any = None) -> Any:
    """Build the standard stance evaluator used elsewhere in the project."""
    eval_conf = config.get("evaluation", {})
    if not eval_conf.get("enabled", False):
        return None

    from src.agents.evaluation_agent import EvaluationAgent
    from src.backend.factory import create_backend

    judge_backend_conf = eval_conf.get("backend")
    if judge_backend_conf:
        evaluator_backend = create_backend({"backend": judge_backend_conf})
    else:
        evaluator_backend = generation_backend or create_backend(config)
    return EvaluationAgent(
        evaluator_backend,
        temperature=float(eval_conf.get("temperature", 0.0)),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _non_placeholder_api_key(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.strip().lower()
    return not (
        normalized.startswith("sk-dummy")
        or normalized.startswith("sk-example")
        or normalized in {"dummy", "example", "changeme"}
    )


def require_configured_backend_credentials(config: Dict[str, Any]) -> None:
    """Fail before artifact creation if a configured remote key is absent."""
    backend_configs = [
        ("generation", config.get("backend", {}) or {}),
    ]
    evaluation = config.get("evaluation", {}) or {}
    if evaluation.get("enabled", False):
        backend_configs.append(
            ("evaluator", evaluation.get("backend", {}) or {})
        )

    for role, backend_config in backend_configs:
        backend_type = str(backend_config.get("type", "openai"))
        if backend_type == "openai":
            key = backend_config.get("api_key") or os.environ.get(
                "OPENAI_API_KEY"
            )
            if not _non_placeholder_api_key(key):
                raise ValueError(
                    f"{role} OpenAI backend requires a non-placeholder "
                    "OPENAI_API_KEY in the operator environment before launch."
                )
        elif backend_type == "openrouter":
            key = backend_config.get("api_key") or os.environ.get(
                "OPENROUTER_API_KEY"
            )
            if not _non_placeholder_api_key(key):
                raise ValueError(
                    f"{role} OpenRouter backend requires a non-placeholder "
                    "OPENROUTER_API_KEY in the operator environment before "
                    "launch."
                )


def _base_url_category(backend: Any) -> str:
    client = getattr(backend, "client", None)
    raw_url = getattr(client, "base_url", None)
    if raw_url is None:
        return "none"
    parsed = urlparse(str(raw_url))
    hostname = (parsed.hostname or "").lower()
    if hostname == "api.openai.com":
        return "official_openai"
    if hostname == "openrouter.ai":
        return "openrouter"
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return "local"
    return "custom_remote"


def attest_runtime_backend(
    backend: Any,
    backend_config: Dict[str, Any],
    *,
    role: str,
) -> Dict[str, Any]:
    """Assert configured/runtime provider identity before the first API call."""
    configured_type = str(backend_config.get("type", "openai"))
    expected_classes = {
        "openai": "src.backend.backend_openai.OpenAIBackend",
        "openrouter": "src.backend.backend_openrouter.OpenRouterBackend",
        "local": "src.backend.backend_local.LocalLLMBackend",
        "mock": "src.backend.backend_mock.MockLLMBackend",
    }
    expected_class = expected_classes.get(configured_type)
    actual_class = (
        f"{backend.__class__.__module__}.{backend.__class__.__name__}"
    )
    if expected_class is None:
        raise RuntimeError(
            f"{role} has unsupported configured backend {configured_type!r}."
        )
    if actual_class != expected_class:
        raise RuntimeError(
            f"{role} backend identity mismatch: configured "
            f"{configured_type!r}, instantiated {actual_class!r}."
        )

    configured_model = backend_config.get("model")
    runtime_model = getattr(backend, "model", None)
    if (
        configured_type != "mock"
        and configured_model is not None
        and runtime_model != configured_model
    ):
        raise RuntimeError(
            f"{role} backend model mismatch: configured "
            f"{configured_model!r}, instantiated {runtime_model!r}."
        )

    base_url_category = _base_url_category(backend)
    if (
        configured_type == "openai"
        and base_url_category != "official_openai"
    ):
        raise RuntimeError(
            f"{role} OpenAI backend did not resolve to the official endpoint."
        )

    attestation = {
        "configured_type": configured_type,
        "runtime_class": actual_class,
        "model": runtime_model,
        "base_url_category": base_url_category,
    }
    for field in ("reasoning_effort", "verbosity"):
        configured_value = backend_config.get(field)
        runtime_value = getattr(backend, field, None)
        if configured_value != runtime_value:
            raise RuntimeError(
                f"{role} backend {field} mismatch: configured "
                f"{configured_value!r}, instantiated {runtime_value!r}."
            )
        attestation[field] = runtime_value
    return attestation


def _partial_run_inventory(raw_path: Path) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(raw_path.rglob("*")):
        relative = path.relative_to(raw_path).as_posix()
        if path.is_symlink():
            entries.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target_sha256": hashlib.sha256(
                        os.readlink(path).encode("utf-8")
                    ).hexdigest(),
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    tree_payload = json.dumps(
        entries,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "captured_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "run_id": raw_path.name,
        "topic_slug": raw_path.parent.name,
        "entries": entries,
        "tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
    }


def quarantine_partial_run(raw_path: Path) -> Dict[str, Any]:
    """Inventory and move an incomplete run before an explicit resume retry."""
    if not raw_path.is_dir() or (raw_path / "results.json").is_file():
        raise ValueError(
            "quarantine_partial_run requires an incomplete run directory."
        )
    inventory = _partial_run_inventory(raw_path)
    write_json_atomic(raw_path / "PARTIAL_RUN_INVENTORY.json", inventory)

    campaign_root = raw_path.parents[2]
    quarantine_parent = (
        campaign_root
        / "quarantine"
        / "partial_runs"
        / raw_path.parent.name
    )
    quarantine_parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    target = quarantine_parent / (
        f"{raw_path.name}__{timestamp}__"
        f"{inventory['tree_sha256'][:12]}"
    )
    suffix = 1
    while target.exists():
        target = quarantine_parent / (
            f"{raw_path.name}__{timestamp}__"
            f"{inventory['tree_sha256'][:12]}__{suffix:02d}"
        )
        suffix += 1
    raw_path.replace(target)
    if raw_path.exists() or not target.is_dir():
        raise RuntimeError("Incomplete-run quarantine move did not complete.")
    return {
        "quarantined_partial": str(target.relative_to(campaign_root)),
        "tree_sha256": inventory["tree_sha256"],
        "entry_count": len(inventory["entries"]),
    }


def evaluator_provenance_from_config(
    config: Dict[str, Any],
    evaluator: Any = None,
) -> Dict[str, Any]:
    """Describe the exact evaluator implementation without making an API call."""
    from src.agents.evaluation_agent import (
        DEFAULT_OPTIMIZED_PROMPT_PATH,
        EvaluationAgent,
    )

    eval_conf = config.get("evaluation", {}) or {}
    backend_conf = eval_conf.get("backend", {}) or {}
    evaluator_code_path = Path(sys.modules[EvaluationAgent.__module__].__file__).resolve()
    prompt_path_value = getattr(evaluator, "default_stance_prompt_path", None)
    prompt_sha256 = getattr(evaluator, "default_stance_prompt_sha256", None)

    if not prompt_path_value:
        prompt_path = DEFAULT_OPTIMIZED_PROMPT_PATH.resolve()
        prompt_path_value = str(prompt_path)
        prompt_sha256 = file_sha256(prompt_path) if prompt_path.exists() else ""

    prompt_path_text = str(prompt_path_value)
    if not prompt_path_text.startswith("builtin:"):
        prompt_path_text = display_path(Path(prompt_path_text))

    return {
        "backend": backend_conf.get("type"),
        "model": backend_conf.get("model", eval_conf.get("model")),
        "reasoning_effort": backend_conf.get("reasoning_effort"),
        "verbosity": backend_conf.get("verbosity"),
        "temperature": float(eval_conf.get("temperature", 0.0)),
        "prompt_path": prompt_path_text,
        "prompt_sha256": prompt_sha256 or "",
        "evaluator_code_path": display_path(evaluator_code_path),
        "evaluator_code_sha256": file_sha256(evaluator_code_path),
        "runner_path": display_path(Path(__file__).resolve()),
        "runner_code_sha256": file_sha256(Path(__file__).resolve()),
        "runner_commit": git_metadata().get("commit", ""),
    }


def evaluator_provenance_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    backend_conf: Dict[str, Any] = {
        "type": getattr(args, "judge_backend", None),
        "model": getattr(args, "judge_model", None),
    }
    reasoning_effort = getattr(args, "judge_reasoning_effort", None)
    verbosity = getattr(args, "judge_verbosity", None)
    if reasoning_effort:
        backend_conf["reasoning_effort"] = reasoning_effort
    if verbosity:
        backend_conf["verbosity"] = verbosity
    return evaluator_provenance_from_config(
        {
            "evaluation": {
                "enabled": True,
                "backend": backend_conf,
                "temperature": float(getattr(args, "judge_temperature", 0.0)),
            }
        }
    )


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Replace a JSON artifact atomically after all validation has succeeded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            shutil.copystat(path, temporary_path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def latest_agent_message_for_round(
    transcript: List[Dict[str, Any]],
    agent_name: str,
    round_index: int,
    agents_per_round: int,
) -> str:
    """Recover the message that SimulationRunner judged at a saved round."""
    window = transcript[: max(0, round_index * agents_per_round)]
    if any(msg.get("round") is not None for msg in transcript):
        window = [
            msg for msg in transcript
            if msg.get("round") is None or int(msg.get("round", round_index)) <= round_index
        ]

    for msg in reversed(window):
        if msg.get("agent_name") == agent_name:
            return str(msg.get("content", ""))
    return ""


def rejudge_existing_run(config_path: str, raw_dir: str, backup: bool = True) -> Optional[str]:
    """Strictly recompute stance fields and replace results only after full success."""
    try:
        from src.utils.config import ConfigLoader

        raw_path = Path(raw_dir)
        result_path = raw_path / "results.json"
        if not result_path.exists():
            return "missing results.json"

        config = ConfigLoader.load_config(config_path)
        evaluator = build_evaluator(config)
        if evaluator is None:
            return "evaluation is disabled"

        result = read_json(result_path)
        transcript = result.get("transcript", [])
        history = result.get("stance_history", [])
        topic = config["debate"]["topic"]
        configured_agents = config.get("simulation", {}).get("agents", [])
        agent_names = [
            str(agent_conf.get("name", fallback))
            for agent_conf, fallback in zip(configured_agents, ["Pro", "Con"])
        ] or ["Pro", "Con"]
        agents_per_round = max(1, len(agent_names))

        judged = 0
        for entry in history:
            round_index = int(entry.get("round", 0) or 0)
            for agent_name in agent_names:
                if round_index == 0:
                    initial_score = entry.get(agent_name)
                    if (
                        isinstance(initial_score, bool)
                        or not isinstance(initial_score, (int, float))
                        or not math.isfinite(float(initial_score))
                        or not -1.0 <= float(initial_score) <= 1.0
                    ):
                        raise ValueError(
                            f"round 0 has invalid initial stance for {agent_name}"
                        )
                    entry[f"{agent_name}_evaluated"] = float(initial_score)
                    entry.pop(f"{agent_name}_confidence", None)
                    continue

                last_message = latest_agent_message_for_round(
                    transcript,
                    agent_name,
                    round_index,
                    agents_per_round,
                )
                if not last_message:
                    raise ValueError(
                        f"round {round_index} has no saved message for {agent_name}"
                    )

                try:
                    eval_result = evaluator.evaluate_stance(
                        transcript=last_message,
                        memory_snapshot="",
                        topic=topic,
                        agent_name=agent_name,
                        strict=True,
                    )
                except Exception as exc:
                    raise ValueError(
                        f"strict stance parse failed at round {round_index} "
                        f"for {agent_name}: {exc}"
                    ) from exc
                entry[f"{agent_name}_evaluated"] = float(eval_result["stance_score"])
                entry.pop(f"{agent_name}_confidence", None)
                judged += 1

        if judged == 0:
            return "no stance entries were rejudged"

        rejudged_at = datetime.now().astimezone().isoformat(timespec="seconds")
        rejudge_metadata = evaluator_provenance_from_config(config, evaluator=evaluator)
        rejudge_metadata.update(
            {
                "rejudged_count": judged,
                "timestamp": rejudged_at,
                "strict_parsing": True,
            }
        )
        result["stance_history"] = history
        result.setdefault("metadata", {})["external_stance_evaluator"] = "src.agents.evaluation_agent.EvaluationAgent"
        result["metadata"]["external_stance_rejudged_at"] = rejudged_at
        result["metadata"]["external_stance_rejudge"] = rejudge_metadata

        if backup:
            backup_path = raw_path / "results.strict_topic_judge_backup.json"
            if not backup_path.exists():
                shutil.copy2(result_path, backup_path)
        write_json_atomic(result_path, result)
        return None
    except Exception as exc:
        return str(exc)


@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    backend_type: str
    model: str
    use_belief_engine: bool
    note: str
    agent_type: str = "subject"
    use_memory: bool = True
    memory_overrides: Dict[str, Any] = field(default_factory=dict)
    extraction_overrides: Dict[str, Any] = field(default_factory=dict)
    backend_config: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    topic: str
    topic_slug: str
    condition_name: str
    condition_label: str
    backend_type: str
    model: str
    backend_config: Dict[str, Any]
    use_belief_engine: bool
    agent_type: str
    use_memory: bool
    memory_overrides: Dict[str, Any]
    trial: int
    seed: int
    extraction_overrides: Dict[str, Any] = field(default_factory=dict)


def slugify(text: str) -> str:
    chars = []
    for char in text.lower():
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "_", "/"}:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def short_topic_slug(topic: str) -> str:
    return SHORT_TOPIC_LABELS.get(topic, slugify(topic))


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def git_metadata() -> Dict[str, Any]:
    def run_git(args: List[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return ""

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status,
    }


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_comparison_spec(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Comparison spec not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Comparison spec must be a YAML mapping: {path}")
    return data


def bool_from_config(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def backend_config_from_spec_item(item: Dict[str, Any]) -> Dict[str, Any]:
    backend_config = dict(item.get("backend_config", {}) or {})
    for key in ("api_key", "url", "base_url"):
        if key in item:
            backend_config[key] = item[key]
    return backend_config


def condition_agent_settings(item: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve public condition labels into concrete factory settings."""
    requested = str(item.get("agent_type", "")).strip().lower().replace("-", "_")

    if requested in {"belief_engine", "be", "subject"}:
        defaults = {
            "agent_type": "subject",
            "use_belief_engine": True,
            "use_memory": True,
            "memory_overrides": dict(OPEN_MINDED_MEMORY),
            "note": "Open-minded Belief Engine agents.",
        }
    elif requested in {"vanilla", "no_be", "no_belief_engine"}:
        defaults = {
            "agent_type": "vanilla",
            "use_belief_engine": False,
            "use_memory": False,
            "memory_overrides": {},
            "note": VANILLA_DEFINITION,
        }
    elif requested in {"prompt_self_update", "self_update", "self_report"}:
        defaults = {
            "agent_type": "self_update",
            "use_belief_engine": False,
            "use_memory": False,
            "memory_overrides": {},
            "note": PROMPT_SELF_UPDATE_DEFINITION,
        }
    elif requested in {"rag", "rag_only", "rag_memory"}:
        defaults = {
            "agent_type": "subject",
            "use_belief_engine": False,
            "use_memory": True,
            "memory_overrides": dict(BLIND_RAG_MEMORY),
            "note": RAG_MEMORY_DEFINITION,
        }
    elif requested in {"rag_self_update", "rag_self_report", "rag_prompt_update"}:
        defaults = {
            "agent_type": "self_update",
            "use_belief_engine": False,
            "use_memory": True,
            "memory_overrides": dict(BLIND_RAG_MEMORY),
            "note": RAG_SELF_UPDATE_DEFINITION,
        }
    else:
        use_be_default = requested not in {"vanilla", "no_be", "no_belief_engine"}
        defaults = {
            "agent_type": "subject" if use_be_default else "vanilla",
            "use_belief_engine": use_be_default,
            "use_memory": use_be_default,
            "memory_overrides": dict(OPEN_MINDED_MEMORY) if use_be_default else {},
            "note": "",
        }

    use_belief_engine = bool_from_config(
        item.get("use_belief_engine"),
        defaults["use_belief_engine"],
    )
    use_memory = bool_from_config(item.get("use_memory"), defaults["use_memory"])
    memory_overrides = dict(defaults["memory_overrides"])
    memory_overrides.update(item.get("memory", {}) or {})

    return {
        "agent_type": str(item.get("factory_type", defaults["agent_type"])),
        "use_belief_engine": use_belief_engine,
        "use_memory": use_memory,
        "memory_overrides": memory_overrides,
        "note": str(item.get("note", defaults["note"])),
    }


def make_spec_conditions(args: argparse.Namespace) -> List[Condition]:
    matrix = load_comparison_spec(Path(args.comparison_spec))

    explicit_conditions = matrix.get("conditions")
    if explicit_conditions:
        conditions: List[Condition] = []
        for item in explicit_conditions:
            if not isinstance(item, dict):
                raise ValueError("Each explicit comparison-spec condition must be a mapping.")
            name = item.get("name")
            model = item.get("model")
            if not name or not model:
                raise ValueError("Each explicit condition needs `name` and `model`.")
            agent_settings = condition_agent_settings(item)
            conditions.append(
                Condition(
                    name=str(name),
                    label=str(item.get("label", name)),
                    backend_type=str(item.get("backend_type", item.get("backend", "openai"))),
                    model=str(model),
                    use_belief_engine=agent_settings["use_belief_engine"],
                    note=agent_settings["note"],
                    agent_type=agent_settings["agent_type"],
                    use_memory=agent_settings["use_memory"],
                    memory_overrides=agent_settings["memory_overrides"],
                    extraction_overrides=dict(
                        item.get("extraction", {}) or {}
                    ),
                    backend_config=backend_config_from_spec_item(item),
                )
            )
        return conditions

    models = matrix.get("models", [])
    if not models:
        raise ValueError("Comparison spec must define either `conditions` or `models`.")

    include = matrix.get("include", {}) or {}
    include_be_default = bool_from_config(include.get("belief_engine"), True)
    include_vanilla_default = bool_from_config(include.get("vanilla"), True)

    conditions = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Each model shorthand entry must be a mapping.")
        model = item.get("model")
        if not model:
            raise ValueError("Each model entry needs `model`.")
        key = str(item.get("key") or slugify(str(model).split("/")[-1]))
        label = str(item.get("label", model))
        backend_type = str(item.get("backend_type", item.get("backend", "openai")))
        backend_config = backend_config_from_spec_item(item)
        note = str(item.get("note", ""))

        include_be = bool_from_config(item.get("belief_engine"), include_be_default)
        include_vanilla = bool_from_config(item.get("vanilla"), include_vanilla_default)

        if include_be:
            conditions.append(
                Condition(
                    name=str(item.get("be_name", f"be_{key}")),
                    label=str(item.get("be_label", f"Belief Engine\n({label})")),
                    backend_type=backend_type,
                    model=str(model),
                    use_belief_engine=True,
                    note=note or "Open-minded Belief Engine agents.",
                    agent_type="subject",
                    use_memory=True,
                    memory_overrides=dict(OPEN_MINDED_MEMORY),
                    backend_config=dict(backend_config),
                )
            )
        if include_vanilla:
            conditions.append(
                Condition(
                    name=str(item.get("vanilla_name", f"vanilla_{key}")),
                    label=str(item.get("vanilla_label", f"Vanilla LLM\n({label})")),
                    backend_type=backend_type,
                    model=str(model),
                    use_belief_engine=False,
                    note=note or VANILLA_DEFINITION,
                    agent_type="vanilla",
                    use_memory=False,
                    memory_overrides={},
                    backend_config=dict(backend_config),
                )
            )

    if not conditions:
        raise ValueError("Comparison spec produced no conditions.")
    return conditions


def make_conditions(args: argparse.Namespace) -> List[Condition]:
    if args.comparison_spec:
        conditions = make_spec_conditions(args)
    else:
        conditions = [
            Condition(
                name="be_gpt54mini",
                label="GPT-5.4-mini BE",
                backend_type=args.gpt_backend,
                model=args.gpt_model,
                use_belief_engine=True,
                note="Open-minded Belief Engine agents.",
                agent_type="subject",
                use_memory=True,
                memory_overrides=dict(OPEN_MINDED_MEMORY),
            ),
            Condition(
                name="be_qwen35_9b",
                label="Qwen3.5 9B BE",
                backend_type=args.qwen_backend,
                model=args.qwen_model,
                use_belief_engine=True,
                note="Open-minded Belief Engine agents.",
                agent_type="subject",
                use_memory=True,
                memory_overrides=dict(OPEN_MINDED_MEMORY),
            ),
            Condition(
                name="be_gemma4_e4b",
                label="Gemma4 E4B BE",
                backend_type=args.gemma_backend,
                model=args.gemma_model,
                use_belief_engine=True,
                note="Open-minded Belief Engine agents.",
                agent_type="subject",
                use_memory=True,
                memory_overrides=dict(OPEN_MINDED_MEMORY),
            ),
            Condition(
                name="vanilla_gpt54mini",
                label="GPT-5.4-mini Vanilla",
                backend_type=args.baseline_backend,
                model=args.baseline_model,
                use_belief_engine=False,
                note=VANILLA_DEFINITION,
                agent_type="vanilla",
                use_memory=False,
                memory_overrides={},
            ),
        ]
    if args.conditions:
        wanted = {item.strip() for item in args.conditions.split(",") if item.strip()}
        known = {condition.name for condition in conditions}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"Unknown condition(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
        conditions = [condition for condition in conditions if condition.name in wanted]
    conditions = apply_be_parameter_overrides(conditions, args)
    return conditions


def apply_be_parameter_overrides(conditions: List[Condition], args: argparse.Namespace) -> List[Condition]:
    if args.be_update_sensitivity is None and args.be_anchor_sensitivity is None:
        return conditions

    updated_conditions = []
    for condition in conditions:
        if not condition.use_belief_engine:
            updated_conditions.append(condition)
            continue

        memory_overrides = dict(condition.memory_overrides or OPEN_MINDED_MEMORY)
        if args.be_update_sensitivity is not None:
            memory_overrides["update_sensitivity"] = args.be_update_sensitivity
        if args.be_anchor_sensitivity is not None:
            memory_overrides["anchor_sensitivity"] = args.be_anchor_sensitivity
        updated_conditions.append(replace(condition, memory_overrides=memory_overrides))
    return updated_conditions


def build_specs(
    conditions: Iterable[Condition],
    trials: int,
    topics: Iterable[str],
    *,
    schedule_order: str = "condition-major",
) -> List[RunSpec]:
    if schedule_order not in SCHEDULE_ORDERS:
        raise ValueError(
            f"Unknown schedule_order={schedule_order!r}; "
            f"expected one of {SCHEDULE_ORDERS}."
        )
    condition_list = list(conditions)
    specs: List[RunSpec] = []
    for topic in topics:
        topic_slug = short_topic_slug(topic)
        if schedule_order == "paired":
            cells = (
                (condition, trial)
                for trial in range(1, trials + 1)
                for condition in condition_list
            )
        else:
            cells = (
                (condition, trial)
                for condition in condition_list
                for trial in range(1, trials + 1)
            )
        for condition, trial in cells:
            seed = stable_seed("two_agent_model_comparison", topic, trial)
            run_id = f"{topic_slug}__{condition.name}__trial_{trial:02d}"
            specs.append(
                RunSpec(
                    run_id=run_id,
                    topic=topic,
                    topic_slug=topic_slug,
                    condition_name=condition.name,
                    condition_label=condition.label,
                    backend_type=condition.backend_type,
                    model=condition.model,
                    backend_config=dict(condition.backend_config),
                    use_belief_engine=condition.use_belief_engine,
                    agent_type=condition.agent_type,
                    use_memory=condition.use_memory,
                    memory_overrides=dict(condition.memory_overrides),
                    extraction_overrides=dict(
                        condition.extraction_overrides
                    ),
                    trial=trial,
                    seed=seed,
                )
            )
    return specs


def resolve_topics(topic_filter: Optional[str]) -> List[str]:
    if not topic_filter:
        return list(TOPICS)

    by_slug = {short_topic_slug(topic): topic for topic in TOPICS}
    by_name = {topic.lower(): topic for topic in TOPICS}
    selected: List[str] = []
    unknown: List[str] = []

    for raw_item in topic_filter.split(","):
        item = raw_item.strip()
        if not item:
            continue
        topic = by_slug.get(item) or by_name.get(item.lower())
        if topic:
            if topic not in selected:
                selected.append(topic)
        else:
            unknown.append(item)

    if unknown:
        known = ", ".join(f"{short_topic_slug(topic)}={topic}" for topic in TOPICS)
        raise ValueError(f"Unknown topic selector(s): {', '.join(unknown)}. Known: {known}")
    if not selected:
        raise ValueError("--topics did not select any topics.")
    return selected


def agent_config(
    spec: RunSpec,
    side: str,
    *,
    effective_memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    is_pro = side == "pro"
    seed_stance = "affirmative" if is_pro else "negative"
    target_initial_stance = INITIAL_STANCE if is_pro else -INITIAL_STANCE
    name = "Pro" if is_pro else "Con"
    side_text = "in favor of" if is_pro else "against"
    agent_type = spec.agent_type

    config: Dict[str, Any] = {
        "name": name,
        "type": agent_type,
        "seed_stance": seed_stance,
        "target_initial_stance": target_initial_stance,
        "persona": (
            f"You argue {side_text} the proposition: '{spec.topic}'. "
            f"{CONTROLLED_DEBATER_PERSONA}"
        ),
    }
    if spec.use_memory:
        config["use_memory"] = True
        config["memory"] = dict(
            effective_memory
            if effective_memory is not None
            else (spec.memory_overrides or OPEN_MINDED_MEMORY)
        )
    return config


def build_config(spec: RunSpec, args: argparse.Namespace) -> Dict[str, Any]:
    backend_config = dict(spec.backend_config)
    backend_config.update(
        {
            "type": spec.backend_type,
            "model": spec.model,
        }
    )
    judge_backend_config = {
        "type": args.judge_backend,
        "model": args.judge_model,
    }
    if args.judge_reasoning_effort:
        judge_backend_config["reasoning_effort"] = args.judge_reasoning_effort
    if args.judge_verbosity:
        judge_backend_config["verbosity"] = args.judge_verbosity
    effective_memory = dict(OPEN_MINDED_MEMORY)
    if spec.use_memory:
        effective_memory.update(spec.memory_overrides)
    if args.allow_self_arguments is not None:
        effective_memory["allow_self_arguments"] = args.allow_self_arguments
    effective_memory["retrieval_k"] = args.retrieval_k
    effective_memory["belief_semantics_mode"] = (
        args.belief_semantics_mode
        if spec.use_belief_engine
        else "incremental_v1"
    )
    effective_memory["response_retrieval_enabled"] = bool_from_config(
        effective_memory.get("response_retrieval_enabled"),
        True,
    )
    dynamic_evidence_horizon = str(
        effective_memory.get("dynamic_evidence_horizon", "persistent")
    )
    if dynamic_evidence_horizon not in {
        "persistent",
        "latest_opponent_turn",
    }:
        raise ValueError(
            "dynamic_evidence_horizon must be 'persistent' or "
            f"'latest_opponent_turn', got {dynamic_evidence_horizon!r}"
        )
    if (
        dynamic_evidence_horizon == "latest_opponent_turn"
        and effective_memory["belief_semantics_mode"]
        != "active_explicit_prior_v1"
    ):
        raise ValueError(
            "latest_opponent_turn requires "
            "--belief-semantics-mode active_explicit_prior_v1"
        )
    effective_memory["dynamic_evidence_horizon"] = dynamic_evidence_horizon
    self_update_context_messages = args.self_update_context_messages
    if self_update_context_messages is None:
        self_update_context_messages = (
            args.response_context_messages
            if args.response_shell == "matched_v1"
            else 6
        )
    extraction_output_mode = getattr(
        args,
        "extraction_output_mode",
        "legacy_text_v1",
    )
    extraction_config = {
        "output_mode": extraction_output_mode,
        "strict_parsing": bool(getattr(args, "strict_extraction", False)),
        "max_attempts": int(getattr(args, "extraction_max_attempts", 2)),
        "max_tokens": int(getattr(args, "extraction_max_tokens", 2000)),
    }
    extraction_config.update(spec.extraction_overrides)
    extraction_output_mode = str(extraction_config["output_mode"])
    if extraction_output_mode not in EXTRACTION_OUTPUT_MODES:
        raise ValueError(
            f"Unknown condition extraction output_mode "
            f"{extraction_output_mode!r}."
        )
    extraction_strict = bool(extraction_config["strict_parsing"])
    memory_strict = bool(
        getattr(args, "strict_memory_ingestion", False)
    )
    extraction_max_attempts = int(extraction_config["max_attempts"])
    extraction_max_tokens = int(extraction_config["max_tokens"])
    return {
        "backend": backend_config,
        "data": {
            "source": "unified_json",
            "json_path": "data/Argument-Quality-Ranking-30k/arguments.json",
        },
        "debate": {
            "topic": spec.topic,
            "rounds": args.rounds,
            "seeding_limit": args.seeding_limit,
            "seeding_strategy": "random",
            "seeding_random_seed": spec.seed,
            "seed_dedup": False,
        },
        "generation": {
            "temperature": args.temperature,
            "self_update_temperature": args.self_update_temperature,
            "response_shell": args.response_shell,
            "response_context_messages": args.response_context_messages,
            "response_word_limit": args.response_word_limit,
            "self_update_context_messages": self_update_context_messages,
        },
        "extraction": extraction_config,
        "memory": {
            "storage_mode": "episodic",
            "stance_method": effective_memory.get("stance_method", "bayesian"),
            "belief_engine_type": effective_memory.get(
                "belief_engine_type",
                "bayesian",
            ),
            "opinion_change_style": effective_memory.get(
                "opinion_change_style",
                "human_like",
            ),
            "worldview_openness": effective_memory.get(
                "worldview_openness",
                "balanced",
            ),
            "retrieval_strategy": args.retrieval_strategy,
            "argument_similarity_threshold": effective_memory["argument_similarity_threshold"],
            "self_similarity_threshold": effective_memory["self_similarity_threshold"],
            "cross_polarity_dedup": effective_memory["cross_polarity_dedup"],
            "deduplication_enabled": bool_from_config(
                effective_memory.get("deduplication_enabled"),
                True,
            ),
            "confirmation_bias": effective_memory["confirmation_bias"],
            "allow_self_arguments": effective_memory["allow_self_arguments"],
            "belief_semantics_mode": effective_memory["belief_semantics_mode"],
            "retrieval_k": effective_memory["retrieval_k"],
            "response_retrieval_enabled": effective_memory[
                "response_retrieval_enabled"
            ],
            "dynamic_evidence_horizon": effective_memory[
                "dynamic_evidence_horizon"
            ],
            "retrieved_evidence_token_budget": effective_memory.get(
                "retrieved_evidence_token_budget"
            ),
            "retrieved_evidence_encoding": effective_memory.get(
                "retrieved_evidence_encoding",
                "cl100k_base",
            ),
            "update_sensitivity": effective_memory["update_sensitivity"],
            "anchor_sensitivity": effective_memory["anchor_sensitivity"],
            "strength_estimation_method": effective_memory.get(
                "strength_estimation_method",
                "classifier",
            ),
            "classifier_model_path": effective_memory.get(
                "classifier_model_path",
                "models/argument_quality_classifier",
            ),
            "uniform_strength_value": effective_memory.get(
                "uniform_strength_value"
            ),
            "strict_ingestion": memory_strict,
        },
        "simulation": {
            "rounds": args.rounds,
            "speaker_order": getattr(args, "speaker_order", "pro-first"),
            "agents": [
                agent_config(spec, "pro", effective_memory=effective_memory),
                agent_config(spec, "con", effective_memory=effective_memory),
            ],
            "live_plot": False,
        },
        "evaluation": {
            "enabled": True,
            "backend": judge_backend_config,
            "temperature": args.judge_temperature,
            "strict_parsing": args.strict_judge,
            "stance_source": "external_judge_last_turn",
        },
        "logging": {
            "log_dir": "experiments/logs",
            "level": "INFO",
            "verbosity": 1,
            "artifact_mode": args.artifact_mode,
            "live_plot": False,
            "live_json": False,
        },
        "validation_metadata": {
            "experiment": "two_agent_model_comparison",
            "run_id": spec.run_id,
            "topic": spec.topic,
            "condition_name": spec.condition_name,
            "condition_label": spec.condition_label,
            "use_belief_engine": spec.use_belief_engine,
            "agent_type": spec.agent_type,
            "use_memory": spec.use_memory,
            "memory_overrides": dict(spec.memory_overrides),
            "extraction_overrides": dict(spec.extraction_overrides),
            "vanilla_definition": VANILLA_DEFINITION,
            "prompt_self_update_definition": PROMPT_SELF_UPDATE_DEFINITION,
            "rag_memory_definition": RAG_MEMORY_DEFINITION,
            "rag_self_update_definition": RAG_SELF_UPDATE_DEFINITION,
            "open_minded_memory": dict(OPEN_MINDED_MEMORY),
            "effective_memory": dict(effective_memory),
            "initial_pro_stance": INITIAL_STANCE,
            "initial_con_stance": -INITIAL_STANCE,
            "seed": spec.seed,
            "trial": spec.trial,
            "speaker_order": getattr(args, "speaker_order", "pro-first"),
            "generation_temperature": args.temperature,
            "self_update_temperature": args.self_update_temperature,
            "response_shell": args.response_shell,
            "response_context_messages": args.response_context_messages,
            "response_word_limit": args.response_word_limit,
            "self_update_context_messages": self_update_context_messages,
            "retrieval_k": args.retrieval_k,
            "response_retrieval_enabled": effective_memory[
                "response_retrieval_enabled"
            ],
            "dynamic_evidence_horizon": effective_memory[
                "dynamic_evidence_horizon"
            ],
            "deduplication_enabled": bool_from_config(
                effective_memory.get("deduplication_enabled"),
                True,
            ),
            "strength_estimation_method": effective_memory.get(
                "strength_estimation_method",
                "classifier",
            ),
            "uniform_strength_value": effective_memory.get(
                "uniform_strength_value"
            ),
            "retrieved_evidence_token_budget": effective_memory.get(
                "retrieved_evidence_token_budget"
            ),
            "retrieved_evidence_encoding": effective_memory.get(
                "retrieved_evidence_encoding",
                "cl100k_base",
            ),
            "allow_self_arguments": effective_memory["allow_self_arguments"],
            "belief_semantics_mode": effective_memory["belief_semantics_mode"],
            "judge_temperature": args.judge_temperature,
            "strict_judge": args.strict_judge,
            "extraction_output_mode": extraction_output_mode,
            "strict_extraction": extraction_strict,
            "strict_memory_ingestion": memory_strict,
            "extraction_max_attempts": extraction_max_attempts,
            "extraction_max_tokens": extraction_max_tokens,
        },
    }


def config_path_for(output_dir: Path, spec: RunSpec) -> Path:
    return output_dir / "configs" / spec.topic_slug / f"{spec.run_id}.yaml"


def partition_existing_tasks(
    tasks: Iterable[Tuple[RunSpec, str, str]],
) -> Tuple[
    List[Tuple[RunSpec, str, str]],
    List[Dict[str, Any]],
]:
    """Separate rejudgeable historical runs from explicitly missing cells."""
    existing = []
    skipped = []
    for spec, config_path, raw_dir in tasks:
        if (Path(raw_dir) / "results.json").exists():
            existing.append((spec, config_path, raw_dir))
        else:
            skipped.append(
                {
                    "run_id": spec.run_id,
                    "reason": "missing historical results.json",
                }
            )
    return existing, skipped


def raw_dir_for(output_dir: Path, spec: RunSpec) -> Path:
    return output_dir / "raw_runs" / spec.topic_slug / spec.run_id


def requires_complete_extraction_audit(output_mode: str) -> bool:
    """Return whether the runner must persist the strict completion contract."""
    return output_mode in EXTRACTION_AUDIT_OUTPUT_MODES


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            yaml.safe_dump(data, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_generated_config(
    config_path: str,
    raw_dir: str,
    resume: bool,
    execution_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Run one generated config with a separate optional judge backend."""
    raw_path = Path(raw_dir)
    results_path = raw_path / "results.json"
    if resume and results_path.exists():
        return None
    execution_started_at = datetime.now().astimezone().isoformat(
        timespec="microseconds"
    )
    recovery_metadata: Optional[Dict[str, Any]] = None
    if raw_path.exists():
        if not resume:
            return (
                "raw run directory already exists; refusing to overwrite "
                "unreviewed artifacts"
            )
        recovery_metadata = quarantine_partial_run(raw_path)
    raw_path.mkdir(parents=True, exist_ok=True)

    agents = []
    try:
        from src.agents.extractor_agent import ArgumentExtractionAgent
        from src.agents.factory import create_agent
        from src.backend.factory import create_backend
        from src.simulation.simulation_runner import SimulationRunner
        from src.utils.config import ConfigLoader
        from src.utils.result_writer import ResultWriter

        config = ConfigLoader.load_config(config_path)
        require_configured_backend_credentials(config)
        config.setdefault("logging", {})["log_dir"] = str(raw_path)

        writer = ResultWriter(str(raw_path))
        write_yaml(raw_path / "resolved_config.yaml", config)

        generation_backend = create_backend(config)
        generation_attestation = attest_runtime_backend(
            generation_backend,
            config.get("backend", {}) or {},
            role="generation",
        )
        topic = config["debate"]["topic"]
        agent_configs = config.get("simulation", {}).get("agents", [])
        needs_extractor = any(
            agent_conf.get("type", "subject") in {"subject", "moltbot"}
            or (
                agent_conf.get("type") == "self_update"
                and bool(agent_conf.get("use_memory", False))
            )
            for agent_conf in agent_configs
        )
        extractor = (
            ArgumentExtractionAgent(generation_backend, topic_id=topic, config=config)
            if needs_extractor
            else NoOpExtractor(topic)
        )
        runtime_backends: Dict[str, Any] = {
            "generation": generation_attestation,
        }
        if needs_extractor:
            runtime_backends["extractor"] = {
                **generation_attestation,
                "shared_with_generation": True,
            }

        for agent_conf in agent_configs:
            agents.append(create_agent(agent_conf, config, generation_backend, log_dir=str(raw_path)))

        evaluator = None
        eval_conf = config.get("evaluation", {})
        if eval_conf.get("enabled", False):
            evaluator = build_evaluator(config, generation_backend=generation_backend)
            runtime_backends["evaluator"] = attest_runtime_backend(
                evaluator.backend,
                eval_conf.get("backend", {}) or config.get("backend", {}),
                role="evaluator",
            )

        logging_config = config.get("logging", {})
        artifact_mode = logging_config.get("artifact_mode", "compact")
        runner = SimulationRunner(
            agents,
            rounds=int(config.get("simulation", {}).get("rounds", config["debate"]["rounds"])),
            extractor=extractor,
            evaluator=evaluator,
            output_dir=str(raw_path),
            live_plot=False,
            live_json=False,
            artifact_mode=artifact_mode,
            evaluator_strict=bool(eval_conf.get("strict_parsing", False)),
            extraction_strict=bool(
                config.get("extraction", {}).get("strict_parsing", False)
            ),
            memory_strict=bool(
                config.get("memory", {}).get("strict_ingestion", False)
            ),
            speaker_order=config.get("simulation", {}).get(
                "speaker_order",
                "pro-first",
            ),
        )
        results = runner.run()
        results.setdefault("metadata", {})[
            "runtime_backends"
        ] = runtime_backends
        if recovery_metadata is not None:
            results.setdefault("metadata", {})[
                "operational_recovery"
            ] = recovery_metadata
        if (
            needs_extractor
            and requires_complete_extraction_audit(
                config.get("extraction", {}).get("output_mode", "")
            )
            and hasattr(extractor, "audit_summary")
        ):
            extraction_audit = extractor.audit_summary()
            expected_extract_calls = len(results.get("transcript", []))
            completed_extract_calls = int(
                extraction_audit.get("extract_calls", 0)
            )
            extractor_code_path = Path(
                sys.modules[extractor.__class__.__module__].__file__
            ).resolve()
            extraction_audit.update(
                {
                    "output_mode": config["extraction"]["output_mode"],
                    "strict_parsing": bool(
                        config["extraction"]["strict_parsing"]
                    ),
                    "max_attempts": int(
                        config["extraction"]["max_attempts"]
                    ),
                    "max_tokens": int(config["extraction"]["max_tokens"]),
                    "backend": config.get("backend", {}).get("type"),
                    "model": config.get("backend", {}).get("model"),
                    "reasoning_effort": config.get("backend", {}).get(
                        "reasoning_effort"
                    ),
                    "verbosity": config.get("backend", {}).get("verbosity"),
                    "expected_extract_calls": expected_extract_calls,
                    "completed_extract_calls": completed_extract_calls,
                    "complete": completed_extract_calls
                    == expected_extract_calls,
                    "extractor_code_path": display_path(
                        extractor_code_path
                    ),
                    "extractor_code_sha256": file_sha256(
                        extractor_code_path
                    ),
                    "runner_commit": git_metadata().get("commit", ""),
                }
            )
            if (
                config.get("extraction", {}).get("strict_parsing", False)
                and completed_extract_calls != expected_extract_calls
            ):
                raise RuntimeError(
                    "Strict extraction audit mismatch: "
                    f"expected {expected_extract_calls} calls, observed "
                    f"{completed_extract_calls}."
                )
            results.setdefault("metadata", {})[
                "argument_extraction"
            ] = extraction_audit
        if evaluator is not None:
            evaluated_count = sum(
                1
                for entry in results.get("stance_history", [])
                if int(entry.get("round", 0) or 0) > 0
                for agent_conf in agent_configs
                if f"{agent_conf.get('name')}_evaluated" in entry
            )
            evaluated_at = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            evaluator_metadata = evaluator_provenance_from_config(
                config,
                evaluator=evaluator,
            )
            evaluator_metadata.update(
                {
                    "evaluated_count": evaluated_count,
                    "timestamp": evaluated_at,
                    "strict_parsing": bool(
                        eval_conf.get("strict_parsing", False)
                    ),
                }
            )
            results.setdefault("metadata", {})[
                "external_stance_generation"
            ] = evaluator_metadata
        if execution_context is not None:
            schedule_metadata = copy.deepcopy(execution_context)
            schedule_metadata.update(
                {
                    "worker_started_at": execution_started_at,
                    "worker_completed_at": (
                        datetime.now().astimezone().isoformat(
                            timespec="microseconds"
                        )
                    ),
                    "system_fingerprint_recorded": False,
                }
            )
            results.setdefault("metadata", {})[
                "execution_schedule"
            ] = schedule_metadata
            # Keep execution attestation independently auditable without
            # requiring the schedule verifier to parse scientific outcomes.
            write_json_atomic(
                raw_path / "execution_schedule.json",
                schedule_metadata,
            )
        memory_dump = {
            agent.name: agent.memory.get_all_arguments()
            for agent in agents
            if getattr(agent, "memory", None) is not None
        }
        if memory_dump:
            write_json_atomic(raw_path / "memory_dump.json", memory_dump)
        if artifact_mode != "compact":
            writer.save_settings_summary(Path(config_path).stem, config_path, config)
        # results.json is the completion marker. Write it atomically and last
        # so interrupted or disk-full runs cannot be resumed or aggregated as
        # successful cells.
        write_json_atomic(results_path, results)
        return None
    except Exception as exc:
        return str(exc)
    finally:
        for agent in agents:
            if hasattr(agent, "close"):
                try:
                    agent.close()
                except Exception:
                    pass


def coerce_stance(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stance_value_with_source(
    entry: Dict[str, Any],
    side: str,
    *,
    use_internal: bool = False,
) -> Tuple[Optional[float], str]:
    if entry.get("round") == 0:
        return (INITIAL_STANCE if side == "pro" else -INITIAL_STANCE), "initial_anchor"

    agent_name = "Pro" if side == "pro" else "Con"

    if use_internal:
        value = coerce_stance(entry.get(agent_name))
        if value is not None:
            return value, "internal_belief_engine"
        value = coerce_stance(entry.get(f"{agent_name}_evaluated"))
        if value is not None:
            return value, "external_judge_fallback"
    else:
        value = coerce_stance(entry.get(f"{agent_name}_evaluated"))
        if value is not None:
            return value, "external_judge"
        value = coerce_stance(entry.get(agent_name))
        if value is not None:
            return value, "internal_belief_engine_fallback"

    return None, "missing"


def stance_value(entry: Dict[str, Any], side: str, *, use_internal: bool = False) -> Optional[float]:
    value, _source = stance_value_with_source(entry, side, use_internal=use_internal)
    return value


def should_use_internal_be(spec: RunSpec, be_stance_source: str) -> bool:
    return spec.use_belief_engine and be_stance_source == "internal"


def collect_trajectories(spec: RunSpec, raw_dir: Path, be_stance_source: str = "external") -> List[Dict[str, Any]]:
    result_path = raw_dir / "results.json"
    if not result_path.exists():
        return []

    result = read_json(result_path)
    rows: List[Dict[str, Any]] = []
    use_internal = should_use_internal_be(spec, be_stance_source)
    for entry in result.get("stance_history", []):
        round_idx = entry.get("round")
        for side in ("pro", "con"):
            stance, stance_source = stance_value_with_source(entry, side, use_internal=use_internal)
            rows.append(
                {
                    "run_id": spec.run_id,
                    "topic": spec.topic,
                    "topic_slug": spec.topic_slug,
                    "condition": spec.condition_name,
                    "condition_label": spec.condition_label,
                    "model": spec.model,
                    "backend_type": spec.backend_type,
                    "use_belief_engine": spec.use_belief_engine,
                    "agent_type": spec.agent_type,
                    "use_memory": spec.use_memory,
                    "trial": spec.trial,
                    "seed": spec.seed,
                    "agent": "Pro" if side == "pro" else "Con",
                    "side": side,
                    "round": round_idx,
                    "stance": stance,
                    "stance_source": stance_source,
                }
            )
    return rows


def summarize_run(spec: RunSpec, raw_dir: Path, be_stance_source: str = "external") -> Optional[Dict[str, Any]]:
    result_path = raw_dir / "results.json"
    if not result_path.exists():
        return None

    result = read_json(result_path)
    history = result.get("stance_history", [])
    if not history:
        return None

    final = history[-1]
    pro_initial = INITIAL_STANCE
    con_initial = -INITIAL_STANCE
    use_internal = should_use_internal_be(spec, be_stance_source)
    pro_final, pro_source = stance_value_with_source(final, "pro", use_internal=use_internal)
    con_final, con_source = stance_value_with_source(final, "con", use_internal=use_internal)
    if pro_final is None or con_final is None:
        return None
    stance_source = pro_source if pro_source == con_source else f"{pro_source}/{con_source}"

    initial_distance = abs(pro_initial - con_initial)
    final_distance = abs(pro_final - con_final)
    updates = result.get("argument_updates", [])
    internalized = [update for update in updates if update.get("status") != "ignored"]

    return {
        "run_id": spec.run_id,
        "topic": spec.topic,
        "topic_slug": spec.topic_slug,
        "condition": spec.condition_name,
        "condition_label": spec.condition_label,
        "backend_type": spec.backend_type,
        "model": spec.model,
        "use_belief_engine": spec.use_belief_engine,
        "agent_type": spec.agent_type,
        "use_memory": spec.use_memory,
        "trial": spec.trial,
        "seed": spec.seed,
        "rounds_recorded": len(history),
        "stance_source": stance_source,
        "pro_initial": pro_initial,
        "con_initial": con_initial,
        "pro_final": pro_final,
        "con_final": con_final,
        "pro_shift": pro_final - pro_initial,
        "con_shift": con_final - con_initial,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "convergence": initial_distance - final_distance,
        "dominance": ((pro_final + con_final) / 2.0) - ((pro_initial + con_initial) / 2.0),
        "final_extremity_mean": (abs(pro_final) + abs(con_final)) / 2.0,
        "argument_updates": len(updates),
        "arguments_internalized": len(internalized),
        "uptake_rate": (len(internalized) / len(updates)) if updates else math.nan,
        "raw_dir": display_path(raw_dir),
    }


def aggregate_outputs(
    output_dir: Path,
    specs: Iterable[RunSpec],
    conditions: List[Condition],
    be_stance_source: str = "external",
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    topic_grid_trajectory_rows: List[Dict[str, Any]] = []

    for spec in specs:
        raw_dir = raw_dir_for(output_dir, spec)
        summary = summarize_run(spec, raw_dir, be_stance_source=be_stance_source)
        if summary:
            summary_rows.append(summary)
        trajectory_rows.extend(collect_trajectories(spec, raw_dir, be_stance_source=be_stance_source))
        topic_grid_trajectory_rows.extend(
            collect_trajectories(
                spec,
                raw_dir,
                be_stance_source=be_stance_source,
            )
        )

    if summary_rows:
        write_csv(output_dir / "summary.csv", summary_rows)
        write_csv(output_dir / "stance_trajectories.csv", trajectory_rows)
        write_csv(output_dir / "topic_grid_stance_trajectories.csv", topic_grid_trajectory_rows)
        generate_figures(
            output_dir,
            summary_rows,
            trajectory_rows,
            conditions,
            topic_grid_trajectory_rows=topic_grid_trajectory_rows,
        )

    return summary_rows


def maybe_import_plotting():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return plt, np, pd


def split_final_parenthetical(label: str) -> Tuple[str, str]:
    label = str(label).strip()
    if label.endswith(")") and "(" in label:
        start = label.rfind("(")
        prefix = label[:start].strip()
        suffix = label[start:].strip()
        if prefix and suffix:
            return prefix, suffix
    return label, ""


def format_title_param(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def generate_figures(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    trajectory_rows: List[Dict[str, Any]],
    conditions: List[Condition],
    topic_grid_trajectory_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if not summary_rows or not trajectory_rows:
        return

    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    plt, np, pd = maybe_import_plotting()
    figures_dir = output_dir / "figures"
    topic_grid_dir = figures_dir / "topic_grids"
    topic_grid_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Avenir Next",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "font.size": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    trajectories = pd.DataFrame(topic_grid_trajectory_rows or trajectory_rows)
    summary = pd.DataFrame(summary_rows)
    stance_sources = set(str(source) for source in trajectories["stance_source"].dropna().unique())
    if "internal_belief_engine" in stance_sources:
        y_label = "Stance\n(BE internal; baselines judged)"
    else:
        y_label = "Externally judged stance"
    condition_order = [condition.name for condition in conditions]
    shared_models = {str(condition.model).strip() for condition in conditions if condition.model}
    label_parts = {
        condition.name: split_final_parenthetical(condition.label)
        for condition in conditions
    }
    shared_suffixes = {suffix for _label, suffix in label_parts.values() if suffix}
    title_model_label = ""
    if len(shared_models) == 1:
        title_model_label = f"Model: {next(iter(shared_models))}"
    be_parameter_settings = {
        (
            condition.memory_overrides.get("update_sensitivity"),
            condition.memory_overrides.get("anchor_sensitivity"),
        )
        for condition in conditions
        if condition.use_belief_engine
    }
    title_parameter_label = ""
    if len(be_parameter_settings) == 1:
        update_sensitivity, anchor_sensitivity = next(iter(be_parameter_settings))
        if update_sensitivity is not None and anchor_sensitivity is not None:
            title_parameter_label = (
                "Belief Engine UA profile: "
                rf"$(u,a)=({format_title_param(update_sensitivity)}, "
                rf"{format_title_param(anchor_sensitivity)})$"
            )
    if len(shared_models) == 1 and len(shared_suffixes) == 1:
        label_by_condition = {
            name: label for name, (label, _suffix) in label_parts.items()
        }
    else:
        label_by_condition = {condition.name: condition.label for condition in conditions}
    pro_color = "#19747e"
    con_color = "#b34a4a"
    text_color = "#293241"
    grid_color = "#e5e7eb"
    spine_color = "#d1d5db"
    max_round = (
        int(trajectories["round"].dropna().max())
        if not trajectories["round"].dropna().empty
        else DEFAULT_ROUNDS
    )
    topics_with_data = [
        topic for topic in TOPICS
        if topic in set(trajectories["topic"].dropna())
    ]
    if not topics_with_data:
        return

    for topic in topics_with_data:
        fig, axes = plt.subplots(
            1,
            len(condition_order),
            figsize=(2.85 * len(condition_order), 3.25),
            sharex=True,
            sharey=True,
            constrained_layout=False,
        )
        axes = np.atleast_1d(axes)
        for col_idx, condition_name in enumerate(condition_order):
            ax = axes[col_idx]
            subset = trajectories[
                (trajectories["topic"] == topic)
                & (trajectories["condition"] == condition_name)
            ]
            ax.axhspan(-0.1, 0.1, color="#f3f4f6", alpha=1.0, linewidth=0, zorder=0)
            ax.axhline(0, color="#9CA3AF", linestyle="--", linewidth=0.9, alpha=0.8, zorder=1)
            ax.grid(axis="y", color=grid_color, linewidth=0.75, alpha=0.78)
            ax.grid(axis="x", color=grid_color, linewidth=0.55, alpha=0.45)
            ax.set_axisbelow(True)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color(spine_color)
                ax.spines[spine].set_linewidth(0.8)
            ax.set_ylim(-1.08, 1.08)
            ax.set_xlim(-0.4, max_round + 0.6)

            for side, color in [("pro", pro_color), ("con", con_color)]:
                side_df = subset[subset["side"] == side].copy()
                side_df = side_df.dropna(subset=["stance"])
                if side_df.empty:
                    continue
                for _run_id, run_df in side_df.groupby("run_id"):
                    run_df = run_df.sort_values("round")
                    ax.plot(
                        run_df["round"],
                        run_df["stance"],
                        color=color,
                        linewidth=1.0,
                        alpha=0.34,
                        solid_capstyle="round",
                        zorder=2,
                    )
                grouped = (
                    side_df.groupby("round")["stance"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                    .sort_values("round")
                )
                grouped["std"] = grouped["std"].fillna(0.0)
                rounds = grouped["round"].to_numpy()
                mean = grouped["mean"].to_numpy()
                std = grouped["std"].to_numpy()
                ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.10, linewidth=0, zorder=1)
                ax.plot(rounds, mean, color=color, linewidth=2.45, solid_capstyle="round", label=f"{side.upper()} mean", zorder=3)

                if len(rounds) > 0:
                    ax.annotate(
                        f"{mean[0]:+.2f}",
                        xy=(rounds[0], mean[0]),
                        xytext=(4, 5 if side == "pro" else -10),
                        textcoords="offset points",
                        ha="left",
                        color=color,
                        fontsize=9.3,
                        fontweight=750,
                        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
                    )
                    ax.annotate(
                        f"{mean[-1]:+.2f}",
                        xy=(rounds[-1], mean[-1]),
                        xytext=(-4, 4 if side == "pro" else -11),
                        textcoords="offset points",
                        ha="right",
                        color=color,
                        fontsize=9.3,
                        fontweight=750,
                        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
                    )

            ax.set_title(
                label_by_condition.get(condition_name, condition_name),
                fontsize=13.4,
                fontweight=750,
                color=text_color,
                pad=8,
                linespacing=0.95,
            )
            if col_idx == 0:
                ax.set_ylabel(y_label, fontsize=10.5, fontweight=700, color=text_color)
            ax.set_xlabel("Round", fontsize=10.5, fontweight=700, color=text_color)
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.tick_params(axis="both", colors=text_color, width=0.8)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, -0.025),
                handlelength=1.8,
                prop={"weight": 650, "size": 9.4},
            )
        title_parts = [f"Topic: {topic}"]
        if title_model_label:
            title_parts.append(title_model_label)
        if title_parameter_label:
            title_parts.append(title_parameter_label)
        title = " | ".join(title_parts)
        fig.suptitle(title, fontsize=12.6, fontweight=700, color="#6b7280", y=0.965)
        fig.subplots_adjust(left=0.082, right=0.992, bottom=0.24, top=0.805, wspace=0.18)
        stem = f"{short_topic_slug(topic)}_two_agent_model_comparison_grid"
        fig.savefig(topic_grid_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(topic_grid_dir / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)

    # Compact overview: mean convergence by topic and condition.
    grouped = summary.groupby(["topic", "condition"])["convergence"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    x = np.arange(len(topics_with_data))
    width = min(0.18, 0.82 / max(1, len(condition_order)))
    colors = ["#19747e", "#5b7c99", "#7b6d8d", "#b34a4a"]
    for idx, condition_name in enumerate(condition_order):
        vals = []
        for topic in topics_with_data:
            match = grouped[(grouped["topic"] == topic) & (grouped["condition"] == condition_name)]
            vals.append(float(match["convergence"].iloc[0]) if not match.empty else np.nan)
        ax.bar(
            x + (idx - (len(condition_order) - 1) / 2) * width,
            vals,
            width=width,
            label=label_by_condition.get(condition_name, condition_name),
            color=colors[idx % len(colors)],
            alpha=0.92,
        )
    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    ax.set_ylabel("Mean convergence")
    ax.set_xticks(x)
    ax.set_xticklabels([short_topic_slug(topic).replace("_", " ") for topic in topics_with_data], rotation=20, ha="right")
    ax.grid(axis="y", color=grid_color, linewidth=0.75, alpha=0.78)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(figures_dir / "figure_two_agent_model_comparison_convergence.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "figure_two_agent_model_comparison_convergence.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def measurement_description(args: argparse.Namespace) -> str:
    if args.be_stance_source == "internal":
        return (
            "Belief Engine conditions are plotted and summarized in every output "
            "using the "
            "agents' internal Belief Engine stance after each round. Non-BE "
            f"conditions remain measured by the "
            f"external judge (`{args.judge_backend}` / `{args.judge_model}`). "
            "Round 0 is fixed to the designed starting stances (+0.75 pro, -0.75 con)."
        )
    return (
        "All summaries, trajectory files, and topic-grid figures use the same "
        f"external judge (`{args.judge_backend}` / `{args.judge_model}`) after "
        "each agent turn, including Belief Engine conditions. "
        "Round 0 is fixed to the designed starting stances (+0.75 pro, -0.75 con)."
    )


def reproduction_command(args: argparse.Namespace, output_dir: Path) -> str:
    """Return a shell-safe command containing all comparison-defining controls."""
    command = [
        "python",
        "validation/scripts/run_two_agent_model_comparison.py",
    ]
    if args.comparison_spec:
        command.extend(["--comparison-spec", display_path(Path(args.comparison_spec))])
    if args.topics:
        command.extend(["--topics", args.topics])
    if args.conditions:
        command.extend(["--conditions", args.conditions])
    command.extend(
        [
            "--rounds",
            str(args.rounds),
            "--trials",
            str(args.trials),
            "--seeding-limit",
            str(args.seeding_limit),
        ]
    )
    if args.be_update_sensitivity is not None:
        command.extend(
            ["--be-update-sensitivity", str(args.be_update_sensitivity)]
        )
    if args.be_anchor_sensitivity is not None:
        command.extend(
            ["--be-anchor-sensitivity", str(args.be_anchor_sensitivity)]
        )
    command.extend(
        [
            "--temperature",
            str(args.temperature),
            "--self-update-temperature",
            str(args.self_update_temperature),
            "--response-shell",
            args.response_shell,
            "--response-context-messages",
            str(args.response_context_messages),
            "--response-word-limit",
            str(args.response_word_limit),
            "--retrieval-strategy",
            args.retrieval_strategy,
            "--retrieval-k",
            str(args.retrieval_k),
            "--belief-semantics-mode",
            args.belief_semantics_mode,
            "--schedule-order",
            args.schedule_order,
            "--speaker-order",
            getattr(args, "speaker_order", "pro-first"),
            "--extraction-output-mode",
            getattr(args, "extraction_output_mode", "legacy_text_v1"),
            "--extraction-max-attempts",
            str(getattr(args, "extraction_max_attempts", 2)),
            "--extraction-max-tokens",
            str(getattr(args, "extraction_max_tokens", 2000)),
        ]
    )
    if args.self_update_context_messages is not None:
        command.extend(
            [
                "--self-update-context-messages",
                str(args.self_update_context_messages),
            ]
        )
    if args.allow_self_arguments is not None:
        command.append(
            "--allow-self-arguments"
            if args.allow_self_arguments
            else "--no-allow-self-arguments"
        )
    if getattr(args, "allow_missing_existing", False):
        command.append("--allow-missing-existing")
    command.extend(
        [
            "--judge-backend",
            args.judge_backend,
            "--judge-model",
            args.judge_model,
            "--judge-reasoning-effort",
            args.judge_reasoning_effort,
            "--judge-verbosity",
            args.judge_verbosity,
            "--judge-temperature",
            str(args.judge_temperature),
            "--be-stance-source",
            args.be_stance_source,
            "--artifact-mode",
            args.artifact_mode,
            "--workers",
            str(args.workers),
            "--output-dir",
            display_path(output_dir),
        ]
    )
    if args.strict_judge:
        command.append("--strict-judge")
    if getattr(args, "strict_extraction", False):
        command.append("--strict-extraction")
    if getattr(args, "strict_memory_ingestion", False):
        command.append("--strict-memory-ingestion")
    return shlex.join(command)


def write_manifest_and_readme(
    output_dir: Path,
    args: argparse.Namespace,
    conditions: List[Condition],
    specs: List[RunSpec],
    failures: List[Dict[str, Any]],
    skipped: Optional[List[Dict[str, Any]]] = None,
) -> None:
    skipped = skipped or []
    selected_topics = []
    for spec in specs:
        if spec.topic not in selected_topics:
            selected_topics.append(spec.topic)
    topic_grid_source_policy = (
        "external_judge_for_all_conditions"
        if args.be_stance_source == "external"
        else "belief_engine_internal_non_be_external_judge"
    )
    manifest = {
        "experiment": "two_agent_model_comparison",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "repository_layout": "repository_relative_v1",
        "git": git_metadata(),
        "args": vars(args),
        "topics": selected_topics,
        "initial_stances": {"pro": INITIAL_STANCE, "con": -INITIAL_STANCE},
        "open_minded_memory": dict(OPEN_MINDED_MEMORY),
        "vanilla_definition": VANILLA_DEFINITION,
        "prompt_self_update_definition": PROMPT_SELF_UPDATE_DEFINITION,
        "rag_memory_definition": RAG_MEMORY_DEFINITION,
        "rag_self_update_definition": RAG_SELF_UPDATE_DEFINITION,
        "conditions": [asdict(condition) for condition in conditions],
        "selected_be_stance_source": args.be_stance_source,
        "topic_grid_stance_source": topic_grid_source_policy,
        "evaluator_provenance": evaluator_provenance_from_args(args),
        "runs": [asdict(spec) for spec in specs],
        "failures": failures,
        "skipped": skipped,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)

    lines = [
        "# Two-Agent Model Comparison",
        "",
        "This run compares multiple LLMs under matched two-agent debate conditions.",
        "",
        "## Baseline Definitions",
        "",
        f"- Vanilla no-update: {VANILLA_DEFINITION}",
        f"- Prompt self-update: {PROMPT_SELF_UPDATE_DEFINITION}",
        f"- RAG-only: {RAG_MEMORY_DEFINITION}",
        f"- RAG + self-update: {RAG_SELF_UPDATE_DEFINITION}",
        "",
        "## Measurement",
        "",
        measurement_description(args),
        "",
        "## Matched controls",
        "",
        f"- Response shell: `{args.response_shell}`",
        f"- Response context: `{args.response_context_messages}` messages",
        "- Self-update context: "
        f"`{args.self_update_context_messages if args.self_update_context_messages is not None else ('response context' if args.response_shell == 'matched_v1' else 6)}`",
        f"- Response word limit: `{args.response_word_limit}`",
        f"- Retrieval: `{args.retrieval_strategy}`, `k={args.retrieval_k}`",
        f"- BE semantics: `{args.belief_semantics_mode}`",
        f"- Within-round speaker order: "
        f"`{getattr(args, 'speaker_order', 'pro-first')}`",
        "- Extraction: "
        f"`{getattr(args, 'extraction_output_mode', 'legacy_text_v1')}`, "
        f"strict=`{getattr(args, 'strict_extraction', False)}`, "
        f"attempts=`{getattr(args, 'extraction_max_attempts', 2)}`, "
        f"max tokens=`{getattr(args, 'extraction_max_tokens', 2000)}`",
        "Memory ingestion fail-closed: "
        f"`{getattr(args, 'strict_memory_ingestion', False)}`",
        f"- Self-authored arguments: `{args.allow_self_arguments}` "
        "(`None` means the condition default)",
        "",
        "### External evaluator provenance",
        "",
        "The complete machine-readable evaluator fingerprint is stored under "
        "`evaluator_provenance` in `manifest.json`, including backend, model, "
        "reasoning effort, verbosity, temperature, prompt path and SHA-256, "
        "evaluator-code SHA-256, runner-code SHA-256, and runner commit.",
        f"Live evaluator strict parsing: `{args.strict_judge}`.",
        "Extractor-backed successful runs store cumulative contract-health "
        "counters in `results.json` under "
        "`metadata.argument_extraction`; strict runs must account for every "
        "generated message.",
        "",
        "## Conditions",
        "",
        "| Condition | Backend | Model | Agent mode |",
        "| --- | --- | --- | --- |",
    ]
    for condition in conditions:
        label = condition.label.replace("\n", "<br>")
        mode = (
            "Belief Engine"
            if condition.use_belief_engine
            else f"{condition.agent_type}, memory={condition.use_memory}"
        )
        lines.append(
            f"| {label} | `{condition.backend_type}` | `{condition.model}` | "
            f"{mode} |"
        )
    lines.extend(
        [
            "",
            "## Topics",
            "",
            *[f"- {topic}" for topic in selected_topics],
            "",
            "## Files",
            "",
            "| Path | Contents |",
            "| --- | --- |",
            "| `configs/` | Generated YAML configs for each topic-condition-trial run. |",
            "| `raw_runs/` | Raw transcripts, memory reports, resolved configs, and `results.json`. |",
            "| `summary.csv` | One row per completed run with final stances and convergence. |",
            "| `stance_trajectories.csv` | Long-form stance trajectories matching the summary stance source. |",
            "| `topic_grid_stance_trajectories.csv` | Topic-grid trajectories using the selected `--be-stance-source`. |",
            "| `figures/topic_grids/` | Topic-grid trajectory figures in the two-agent style. |",
            "",
            "## Reproduce",
            "",
            "```bash",
            reproduction_command(args, output_dir),
            "```",
            "",
            "To compare other models, pass `--comparison-spec` with a YAML file "
            "that lists condition names, agent modes, labels, backends, and slugs. "
            "The exact resolved strings are "
            "saved in `manifest.json` and each generated config.",
        ]
    )
    if failures:
        lines.extend(["", "## Failures", "", "| Run | Error |", "| --- | --- |"])
        for failure in failures:
            lines.append(f"| `{failure['run_id']}` | {failure['error']} |")
    if skipped:
        lines.extend(["", "## Skipped", "", "| Run | Reason |", "| --- | --- |"])
        for item in skipped:
            lines.append(f"| `{item['run_id']}` | {item['reason']} |")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def write_campaign_heartbeat(
    output_dir: Path,
    specs: List[RunSpec],
    *,
    status: str,
    started_at: str,
    failures: List[Dict[str, Any]],
) -> None:
    """Publish compact parent-runner liveness/progress without raw content."""
    completed = [
        spec.run_id
        for spec in specs
        if (raw_dir_for(output_dir, spec) / "results.json").is_file()
    ]
    payload = {
        "schema_version": 1,
        "owner": "run_two_agent_model_comparison",
        "pid": os.getpid(),
        "runner_commit": git_metadata().get("commit", ""),
        "started_at": started_at,
        "updated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "status": status,
        "registered_runs": len(specs),
        "completed_runs": len(completed),
        "completed_run_ids": completed,
        "failed_run_ids": [
            item.get("run_id")
            for item in failures
            if isinstance(item, dict)
        ],
    }
    write_json_atomic(
        output_dir / "monitor" / "runner_heartbeat.json",
        payload,
    )


def run(args: argparse.Namespace) -> int:
    for name in (
        "response_context_messages",
        "response_word_limit",
        "retrieval_k",
        "extraction_max_attempts",
        "extraction_max_tokens",
    ):
        if int(
            getattr(
                args,
                name,
                2 if name == "extraction_max_attempts" else 2000,
            )
        ) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if (
        args.self_update_context_messages is not None
        and int(args.self_update_context_messages) < 1
    ):
        raise ValueError("--self-update-context-messages must be positive.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "validation" / "results" / f"two_agent_model_comparison_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_runs").mkdir(parents=True, exist_ok=True)

    conditions = make_conditions(args)
    topics = resolve_topics(args.topics)
    specs = build_specs(
        conditions,
        args.trials,
        topics,
        schedule_order=args.schedule_order,
    )
    if args.max_runs:
        specs = specs[: args.max_runs]

    failures: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for spec in specs:
        write_yaml(config_path_for(output_dir, spec), build_config(spec, args))

    if args.dry_run:
        write_manifest_and_readme(
            output_dir,
            args,
            conditions,
            specs,
            failures,
            skipped,
        )
        print(f"Wrote {len(specs)} configs to {display_path(output_dir)}")
        return 0

    tasks = [
        (spec, str(config_path_for(output_dir, spec)), str(raw_dir_for(output_dir, spec)))
        for spec in specs
    ]
    heartbeat_started_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    if args.rejudge_existing:
        if args.allow_missing_existing:
            tasks, skipped = partition_existing_tasks(tasks)
        backup = not args.no_rejudge_backup
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_map = {
                    executor.submit(rejudge_existing_run, config_path, raw_dir, backup): spec
                    for spec, config_path, raw_dir in tasks
                }
                for future in as_completed(future_map):
                    spec = future_map[future]
                    error = future.result()
                    if error:
                        failures.append({"run_id": spec.run_id, "error": error})
                        print(f"[REJUDGE FAIL] {spec.run_id}: {error}")
                    else:
                        print(f"[REJUDGED] {spec.run_id}")
        else:
            for idx, (spec, config_path, raw_dir) in enumerate(tasks, start=1):
                print(f"[{idx}/{len(tasks)}] Rejudging {spec.run_id}")
                error = rejudge_existing_run(config_path, raw_dir, backup)
                if error:
                    failures.append({"run_id": spec.run_id, "error": error})
                    print(f"[REJUDGE FAIL] {spec.run_id}: {error}")

        aggregate_outputs(output_dir, specs, conditions, be_stance_source=args.be_stance_source)
        write_manifest_and_readme(
            output_dir,
            args,
            conditions,
            specs,
            failures,
            skipped,
        )
        print(f"Rejudged and aggregated outputs in {display_path(output_dir)}")
        if failures:
            print(f"{len(failures)} runs failed to rejudge. See README.md and manifest.json.")
            return 1
        return 0

    if args.aggregate_only:
        aggregate_outputs(output_dir, specs, conditions, be_stance_source=args.be_stance_source)
        write_manifest_and_readme(
            output_dir,
            args,
            conditions,
            specs,
            failures,
            skipped,
        )
        print(f"Aggregated outputs in {display_path(output_dir)}")
        return 0

    if args.workers > 1:
        write_campaign_heartbeat(
            output_dir,
            specs,
            status="running",
            started_at=heartbeat_started_at,
            failures=failures,
        )
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(run_generated_config, config_path, raw_dir, args.resume): spec
                for spec, config_path, raw_dir in tasks
            }
            for future in as_completed(future_map):
                spec = future_map[future]
                error = future.result()
                if error:
                    failures.append({"run_id": spec.run_id, "error": error})
                    print(f"[FAIL] {spec.run_id}: {error}")
                else:
                    print(f"[OK] {spec.run_id}")
                write_campaign_heartbeat(
                    output_dir,
                    specs,
                    status="running",
                    started_at=heartbeat_started_at,
                    failures=failures,
                )
    else:
        write_campaign_heartbeat(
            output_dir,
            specs,
            status="running",
            started_at=heartbeat_started_at,
            failures=failures,
        )
        for idx, (spec, config_path, raw_dir) in enumerate(tasks, start=1):
            print(f"[{idx}/{len(tasks)}] Running {spec.run_id}")
            error = run_generated_config(config_path, raw_dir, args.resume)
            if error:
                failures.append({"run_id": spec.run_id, "error": error})
                print(f"[FAIL] {spec.run_id}: {error}")
            write_campaign_heartbeat(
                output_dir,
                specs,
                status="running",
                started_at=heartbeat_started_at,
                failures=failures,
            )

    aggregate_outputs(output_dir, specs, conditions, be_stance_source=args.be_stance_source)
    write_manifest_and_readme(
        output_dir,
        args,
        conditions,
        specs,
        failures,
        skipped,
    )
    write_campaign_heartbeat(
        output_dir,
        specs,
        status="failed" if failures else "complete",
        started_at=heartbeat_started_at,
        failures=failures,
    )

    print(f"Done. Results: {display_path(output_dir)}")
    if failures:
        print(f"{len(failures)} runs failed. See README.md and manifest.json.")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple LLMs as Belief Engine and/or vanilla two-agent debaters."
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for configs, raw runs, summaries, and figures.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Debate rounds per run.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Trials per topic-condition cell.")
    parser.add_argument(
        "--topics",
        default=None,
        help=(
            "Comma-separated topic slugs or exact topic strings. "
            "Example: social_media,journalism"
        ),
    )
    parser.add_argument("--seeding-limit", type=int, default=10, help="Seed arguments per side.")
    parser.add_argument(
        "--be-update-sensitivity",
        type=float,
        default=None,
        help="Override update_sensitivity for Belief Engine conditions.",
    )
    parser.add_argument(
        "--be-anchor-sensitivity",
        type=float,
        default=None,
        help="Override anchor_sensitivity for Belief Engine conditions.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Agent response generation temperature.")
    parser.add_argument(
        "--self-update-temperature",
        type=float,
        default=0.0,
        help="Temperature for prompt self-update JSON stance updates.",
    )
    parser.add_argument(
        "--response-shell",
        choices=RESPONSE_SHELLS,
        default="legacy",
        help="Versioned response-generation shell shared across selected conditions.",
    )
    parser.add_argument(
        "--response-context-messages",
        type=int,
        default=4,
        help="Recent transcript messages exposed during response generation.",
    )
    parser.add_argument(
        "--self-update-context-messages",
        type=int,
        default=None,
        help=(
            "Recent transcript messages exposed to prompt self-update. "
            "Defaults to four for matched_v1 and six for legacy."
        ),
    )
    parser.add_argument(
        "--response-word-limit",
        type=int,
        default=120,
        help="Common response word limit for generated conditions.",
    )
    parser.add_argument(
        "--retrieval-strategy",
        default="proportional",
        help=(
            "Belief Engine retrieval strategy for generated BE configs. "
            "Use `proportional` to match the focus10 two-agent artifact."
        ),
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="Common retrieval budget for memory-backed response conditions.",
    )
    parser.add_argument(
        "--belief-semantics-mode",
        choices=BELIEF_SEMANTICS_MODES,
        default="incremental_v1",
        help=(
            "Belief-state contract for BE conditions. Non-BE memory conditions "
            "always use incremental_v1."
        ),
    )
    parser.add_argument(
        "--extraction-output-mode",
        choices=EXTRACTION_OUTPUT_MODES,
        default="legacy_text_v1",
        help=(
            "Versioned extractor response contract. structured_json_v1 uses "
            "schema-constrained JSON and bounded whole-stage retries."
        ),
    )
    parser.add_argument(
        "--extraction-max-attempts",
        type=int,
        default=2,
        help="Maximum complete attempts per structured extraction stage.",
    )
    parser.add_argument(
        "--extraction-max-tokens",
        type=int,
        default=2000,
        help="Completion-token budget for each structured extraction attempt.",
    )
    parser.add_argument(
        "--strict-extraction",
        action="store_true",
        help=(
            "Fail the entire trial on an unrecovered extraction-contract "
            "failure or incomplete extraction-call audit."
        ),
    )
    parser.add_argument(
        "--strict-memory-ingestion",
        action="store_true",
        help=(
            "Fail the entire trial if an extracted record cannot be admitted "
            "to an intended agent memory."
        ),
    )
    parser.add_argument(
        "--allow-self-arguments",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override self-authored memory ingestion. Use "
            "--no-allow-self-arguments for a received-evidence-only comparison."
        ),
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument(
        "--schedule-order",
        choices=SCHEDULE_ORDERS,
        default="condition-major",
        help=(
            "Execution order only. paired interleaves conditions within each "
            "topic/trial to bound classifier memory and temporal drift."
        ),
    )
    parser.add_argument(
        "--speaker-order",
        choices=SPEAKER_ORDERS,
        default="pro-first",
        help=(
            "Within-round order for the two configured agents. pro-first "
            "preserves the historical Pro-then-Con default; con-first reverses "
            "it for the order-sensitivity experiment."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip runs that already have results.json.")
    parser.add_argument("--dry-run", action="store_true", help="Write configs and metadata without executing runs.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing raw runs and regenerate figures.")
    parser.add_argument(
        "--rejudge-existing",
        action="store_true",
        help="Recompute external stance fields in existing results.json files, then aggregate.",
    )
    parser.add_argument(
        "--no-rejudge-backup",
        action="store_true",
        help="Do not create results.strict_topic_judge_backup.json before rejudging.",
    )
    parser.add_argument(
        "--allow-missing-existing",
        action="store_true",
        help=(
            "With --rejudge-existing, record missing historical runs as skipped "
            "instead of failures. This never applies to fresh generation."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument(
        "--comparison-spec",
        dest="comparison_spec",
        type=str,
        default=str(REPO_ROOT / "validation/config/two_agent_model_comparison.yaml"),
        help=(
            "YAML file defining any number of independent two-agent conditions "
            "to compare."
        ),
    )
    parser.add_argument("--model-matrix", dest="comparison_spec", help=argparse.SUPPRESS)
    parser.add_argument(
        "--conditions",
        default=None,
        help=(
            "Comma-separated condition names to run. In the optional `models` "
            "shorthand, generated names use be_<key> and vanilla_<key> unless "
            "explicitly overridden."
        ),
    )
    parser.add_argument("--artifact-mode", choices=["compact", "full"], default="compact", help="Raw run artifact detail.")
    parser.add_argument(
        "--be-stance-source",
        choices=["external", "internal"],
        default="external",
        help=(
            "Stance source for Belief Engine conditions when aggregating and "
            "plotting. Vanilla conditions are always externally judged."
        ),
    )

    parser.add_argument("--gpt-backend", default="openai", help=argparse.SUPPRESS)
    parser.add_argument("--gpt-model", default="gpt-5.4-mini", help=argparse.SUPPRESS)
    parser.add_argument("--qwen-backend", default="openrouter", help=argparse.SUPPRESS)
    parser.add_argument("--qwen-model", default="qwen/qwen3.5-9b", help=argparse.SUPPRESS)
    parser.add_argument("--gemma-backend", default="openrouter", help=argparse.SUPPRESS)
    parser.add_argument("--gemma-model", default="google/gemma-4-e4b", help=argparse.SUPPRESS)
    parser.add_argument(
        "--vanilla-backend",
        "--baseline-backend",
        dest="baseline_backend",
        metavar="VANILLA_BACKEND",
        default="openai",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--vanilla-model",
        "--baseline-model",
        dest="baseline_model",
        metavar="VANILLA_MODEL",
        default="gpt-5.4-mini",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--judge-backend", default="openai", help="Shared external judge backend.")
    parser.add_argument("--judge-model", default="gpt-5.4-mini", help="Shared external judge model.")
    parser.add_argument(
        "--judge-reasoning-effort",
        default="none",
        help="Optional reasoning_effort for OpenAI judge models; use `none` to allow deterministic temperature.",
    )
    parser.add_argument("--judge-verbosity", default="low", help="Optional verbosity for OpenAI judge models.")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="External judge temperature.")
    parser.add_argument(
        "--strict-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail a generated run on malformed, non-finite, or out-of-range "
            "external stance output instead of using the legacy neutral fallback."
        ),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
