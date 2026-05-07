#!/usr/bin/env python3
"""Run controlled two-agent AQR debate experiments.

This script generates controlled two-agent designs used to demonstrate Belief
Engine controllability. It can either dry-run the configs or execute them
through the standard multi-agent simulation runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FULL_TOPICS = [
    {
        "topic_class": "Social/political",
        "topic": "Social media brings more harm than good",
    },
    {
        "topic_class": "Institutional policy",
        "topic": "We should introduce compulsory voting",
    },
    {
        "topic_class": "Factual/health-ish",
        "topic": "Homeopathy brings more harm than good",
    },
]


def load_focus_topics() -> List[Dict[str, str]]:
    focus_path = REPO_ROOT / "data" / "focus_topics.json"
    with focus_path.open() as f:
        topics = json.load(f)
    return [
        {
            "topic_class": "Focus topic",
            "topic": str(topic),
        }
        for topic in topics
    ]


CONTROLLED_DEBATER_PERSONA = (
    "You are a careful, concise debater. Defend your assigned side, engage "
    "directly with the opposing side's arguments, and stay focused on the topic."
)

FULL_PROFILES = {
    "Open": {
        "K": 0.30,
        "K_a": 0.20,
    },
    "Balanced": {
        "K": 0.15,
        "K_a": 0.80,
    },
    "Stubborn": {
        "K": 0.06,
        "K_a": 1.60,
    },
}

FULL_PAIRINGS = [
    ("Open/Open", "Open", "Open"),
    ("Balanced/Balanced", "Balanced", "Balanced"),
    ("Stubborn/Stubborn", "Stubborn", "Stubborn"),
    ("Open/Stubborn", "Open", "Stubborn"),
    ("Stubborn/Open", "Stubborn", "Open"),
    ("Balanced/Stubborn", "Balanced", "Stubborn"),
    ("Stubborn/Balanced", "Stubborn", "Balanced"),
]

SELF_CONDITIONS = [
    ("self_off", False, "pure opponent-exposure belief updating"),
    ("self_on", True, "full debate dynamics with self-reinforcement"),
]

SHOWCASE_TOPICS = [
    {
        "topic_class": "Institutional policy",
        "topic": "We should introduce compulsory voting",
    },
]

HUMAN_SHOWCASE_EXTRA_TOPICS = [
    {
        "topic_class": "Social/political",
        "topic": "Social media brings more harm than good",
    },
    {
        "topic_class": "Factual/health-ish",
        "topic": "Homeopathy brings more harm than good",
    },
]

SHOWCASE_PROFILES = {
    "Open": {
        "K": 0.30,
        "K_a": 0.15,
    },
    "Balanced": {
        "K": 0.15,
        "K_a": 0.40,
    },
    "Stubborn": {
        "K": 0.06,
        "K_a": 0.90,
    },
}

SHOWCASE_PAIRINGS = [
    ("Open/Open", "Open", "Open"),
    ("Balanced/Balanced", "Balanced", "Balanced"),
    ("Stubborn/Stubborn", "Stubborn", "Stubborn"),
    ("Open/Stubborn", "Open", "Stubborn"),
    ("Stubborn/Open", "Stubborn", "Open"),
]

HUMAN_SHOWCASE_PROFILES = {
    "Open": {
        "K": 0.40,
        "K_a": 0.20,
        "opinion_change_style": "human_like",
        "confirmation_bias": 0.00,
        "worldview_openness": "balanced",
        "argument_similarity_threshold": 0.60,
        "self_similarity_threshold": 0.45,
        "framework_axes": {
            "prior_strength": "weak",
            "evidence_responsiveness": "high",
            "selective_uptake": "classifier + vector dedup",
        },
    },
    "Committed": {
        "K": 0.10,
        "K_a": 0.80,
        "opinion_change_style": "human_like",
        "confirmation_bias": 0.00,
        "worldview_openness": "balanced",
        "argument_similarity_threshold": 0.60,
        "self_similarity_threshold": 0.45,
        "framework_axes": {
            "prior_strength": "strong",
            "evidence_responsiveness": "low",
            "selective_uptake": "classifier + vector dedup",
        },
    },
}

HUMAN_SHOWCASE_PAIRINGS = [
    (f"{pro}/{con}", pro, con)
    for pro in ("Open", "Committed")
    for con in ("Open", "Committed")
]

DISPLAY_PROFILE_LABELS = {
    "Open": "Open-minded",
    "Committed": "Stubborn",
}

ANCHOR_SHOWCASE_PROFILES = {
    "WeakAnchor": {
        "K": 0.15,
        "K_a": 0.15,
    },
    "MediumAnchor": {
        "K": 0.15,
        "K_a": 0.60,
    },
    "StrongAnchor": {
        "K": 0.15,
        "K_a": 1.60,
    },
}

ANCHOR_SHOWCASE_PAIRINGS = [
    ("WeakAnchor/WeakAnchor", "WeakAnchor", "WeakAnchor"),
    ("MediumAnchor/MediumAnchor", "MediumAnchor", "MediumAnchor"),
    ("StrongAnchor/StrongAnchor", "StrongAnchor", "StrongAnchor"),
]

PRESETS = {
    "full": {
        "topics": FULL_TOPICS,
        "profiles": FULL_PROFILES,
        "pairings": FULL_PAIRINGS,
        "trials": 1,
        "seeding_limit": 10,
        "seeding_strategy": "random",
        "initial_stance": 0.99,
        "initialization_mode": "target_controlled",
        "self_conditions": SELF_CONDITIONS,
        "description": "3-topic robustness design with 7 pairings and one replicate.",
    },
    "clean_showcase": {
        "topics": SHOWCASE_TOPICS,
        "profiles": SHOWCASE_PROFILES,
        "pairings": SHOWCASE_PAIRINGS,
        "trials": 3,
        "seeding_limit": 5,
        "seeding_strategy": "random",
        "initial_stance": 0.75,
        "initialization_mode": "target_controlled",
        "self_conditions": SELF_CONDITIONS,
        "description": "Topic5-style 1-topic showcase with 5 pairings, self-on/off, and 3 replicates.",
    },
    "human_showcase": {
        "topics": SHOWCASE_TOPICS,
        "profiles": HUMAN_SHOWCASE_PROFILES,
        "pairings": HUMAN_SHOWCASE_PAIRINGS,
        "trials": 3,
        "seeding_limit": 10,
        "seeding_strategy": "random",
        "seed_dedup": False,
        "initial_stance": 0.75,
        "initialization_mode": "target_controlled",
        "self_conditions": [("self_on", True, "full-dialogue updating with self arguments")],
        "description": (
            "Human-like 1-topic showcase with a 2x2 Open/Committed profile grid, "
            "3 replicates, self arguments enabled, classifier strength scoring, "
            "and vector deduplication."
        ),
    },
    "human_showcase_multi_topic": {
        "topics": FULL_TOPICS,
        "profiles": HUMAN_SHOWCASE_PROFILES,
        "pairings": HUMAN_SHOWCASE_PAIRINGS,
        "trials": 3,
        "seeding_limit": 10,
        "seeding_strategy": "random",
        "seed_dedup": False,
        "initial_stance": 0.75,
        "initialization_mode": "target_controlled",
        "self_conditions": [("self_on", True, "full-dialogue updating with self arguments")],
        "description": (
            "Human-like 3-topic showcase with a 2x2 Open/Committed profile grid, "
            "3 replicates per topic/pairing, self arguments enabled, classifier "
            "strength scoring, and vector deduplication."
        ),
    },
    "human_showcase_extra_topics": {
        "topics": HUMAN_SHOWCASE_EXTRA_TOPICS,
        "profiles": HUMAN_SHOWCASE_PROFILES,
        "pairings": HUMAN_SHOWCASE_PAIRINGS,
        "trials": 3,
        "seeding_limit": 10,
        "seeding_strategy": "random",
        "seed_dedup": False,
        "initial_stance": 0.75,
        "initialization_mode": "target_controlled",
        "self_conditions": [("self_on", True, "full-dialogue updating with self arguments")],
        "description": (
            "Human-like extra-topic showcase with a 2x2 Open/Committed profile "
            "grid on social-media and homeopathy topics, 3 replicates per "
            "topic/pairing, self arguments enabled, classifier strength scoring, "
            "and vector deduplication."
        ),
    },
    "human_showcase_focus10": {
        "topics": load_focus_topics(),
        "profiles": HUMAN_SHOWCASE_PROFILES,
        "pairings": HUMAN_SHOWCASE_PAIRINGS,
        "trials": 3,
        "seeding_limit": 10,
        "seeding_strategy": "random",
        "seed_dedup": False,
        "initial_stance": 0.75,
        "initialization_mode": "target_controlled",
        "self_conditions": [("self_on", True, "full-dialogue updating with self arguments")],
        "topic_separated": True,
        "description": (
            "Human-like 10-topic focus showcase with a 2x2 Open/Committed "
            "profile grid, 3 replicates per topic/pairing, self arguments "
            "enabled, classifier strength scoring, and vector deduplication."
        ),
    },
    "anchor_showcase": {
        "topics": SHOWCASE_TOPICS,
        "profiles": ANCHOR_SHOWCASE_PROFILES,
        "pairings": ANCHOR_SHOWCASE_PAIRINGS,
        "trials": 3,
        "seeding_limit": 5,
        "seeding_strategy": "random",
        "initial_stance": None,
        "initialization_mode": "seed_derived",
        "self_conditions": [("self_off", False, "pure opponent-exposure belief updating")],
        "description": (
            "Seed-derived prior showcase: K is fixed and K_a varies, with no "
            "target_initial_stance override, so seed arguments form the initial stance."
        ),
    },
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    topic: str
    topic_class: str
    pairing: str
    pro_profile: str
    con_profile: str
    self_condition: str
    allow_self_arguments: bool
    seed: int
    trial: int
    initialization_mode: str
    target_initial_stance: Optional[float]


def slugify(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:90]


def stable_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 1_000_000_000


def get_preset(name: str) -> Dict[str, Any]:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {', '.join(sorted(PRESETS))}")
    return PRESETS[name]


def display_profile_label(profile: str) -> str:
    return DISPLAY_PROFILE_LABELS.get(profile, profile)


def display_pairing_label(pairing: str) -> str:
    if "/" not in pairing:
        return display_profile_label(pairing)
    pro, con = pairing.split("/", 1)
    return f"{display_profile_label(pro)}/{display_profile_label(con)}"


def uses_topic_subdirs(preset_name: str) -> bool:
    return bool(get_preset(preset_name).get("topic_separated", False))


def topic_subdir(spec: RunSpec) -> str:
    return slugify(spec.topic)


def short_topic_slug(topic: str) -> str:
    labels = {
        "Social media brings more harm than good": "social_media",
        "Homeopathy brings more harm than good": "homeopathy",
        "Entrapment should be legalized": "entrapment",
        "We should adopt a zero-tolerance policy in schools": "zero_tolerance_schools",
        "We should introduce compulsory voting": "compulsory_voting",
        "We should adopt an austerity regime": "austerity",
        "We should legalize sex selection": "sex_selection",
        "We should adopt atheism": "atheism",
        "We should subsidize journalism": "journalism",
        "We should adopt libertarianism": "libertarianism",
    }
    if topic in labels:
        return labels[topic]
    return "_".join(slugify(topic).split("_")[:5])


def config_path_for(output_dir: Path, spec: RunSpec, preset_name: str) -> Path:
    configs_dir = output_dir / "configs"
    if uses_topic_subdirs(preset_name):
        configs_dir = configs_dir / topic_subdir(spec)
    return configs_dir / f"{spec.run_id}.yaml"


def raw_dir_for(output_dir: Path, spec: RunSpec, preset_name: str) -> Path:
    raw_runs_dir = output_dir / "raw_runs"
    if uses_topic_subdirs(preset_name):
        raw_runs_dir = raw_runs_dir / topic_subdir(spec)
    return raw_runs_dir / spec.run_id


def build_specs(preset_name: str = "full") -> List[RunSpec]:
    preset = get_preset(preset_name)
    specs: List[RunSpec] = []
    for topic_info in preset["topics"]:
        for pairing_name, pro_profile, con_profile in preset["pairings"]:
            for trial in range(1, int(preset["trials"]) + 1):
                base_seed = stable_seed(topic_info["topic"], pairing_name, f"replicate_{trial}")
                for self_condition, allow_self, _description in preset.get("self_conditions", SELF_CONDITIONS):
                    run_id = "__".join(
                        [
                            slugify(topic_info["topic"]),
                            slugify(pairing_name.replace("/", "_vs_")),
                            f"trial_{trial:02d}",
                            self_condition,
                        ]
                    )
                    specs.append(
                        RunSpec(
                            run_id=run_id,
                            topic=topic_info["topic"],
                            topic_class=topic_info["topic_class"],
                            pairing=pairing_name,
                            pro_profile=pro_profile,
                            con_profile=con_profile,
                            self_condition=self_condition,
                            allow_self_arguments=allow_self,
                            seed=base_seed,
                            trial=trial,
                            initialization_mode=preset.get("initialization_mode", "target_controlled"),
                            target_initial_stance=preset.get("initial_stance"),
                        )
                    )
    return specs


def profile_memory(profile_name: str, allow_self_arguments: bool, profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    profile = profiles[profile_name]
    memory = {
        "update_sensitivity": profile["K"],
        "anchor_sensitivity": profile["K_a"],
        "allow_self_arguments": allow_self_arguments,
    }
    for key in (
        "confirmation_bias",
        "worldview_openness",
        "argument_similarity_threshold",
        "self_similarity_threshold",
        "cross_polarity_dedup",
        "opinion_change_style",
    ):
        if key in profile:
            memory[key] = profile[key]
    return memory


def build_config(spec: RunSpec, backend_type: str, model: str, rounds: int, preset_name: str) -> Dict[str, Any]:
    preset = get_preset(preset_name)
    profiles = preset["profiles"]
    initialization_mode = preset.get("initialization_mode", "target_controlled")
    initial_stance = preset.get("initial_stance")
    is_human_showcase = preset_name.startswith("human_showcase")
    pro_agent = {
        "name": f"Pro_{spec.pro_profile}",
        "type": "subject",
        "seed_stance": "affirmative",
        "memory": profile_memory(spec.pro_profile, spec.allow_self_arguments, profiles),
        "persona": (
            f"You argue in favor of the proposition: '{spec.topic}'. "
            f"{CONTROLLED_DEBATER_PERSONA}"
        ),
    }
    con_agent = {
        "name": f"Con_{spec.con_profile}",
        "type": "subject",
        "seed_stance": "negative",
        "memory": profile_memory(spec.con_profile, spec.allow_self_arguments, profiles),
        "persona": (
            f"You argue against the proposition: '{spec.topic}'. "
            f"{CONTROLLED_DEBATER_PERSONA}"
        ),
    }
    if initialization_mode == "target_controlled":
        if initial_stance is None:
            raise ValueError(f"Preset '{preset_name}' uses target_controlled initialization but has no initial_stance.")
        initial_value = float(initial_stance)
        pro_agent["target_initial_stance"] = initial_value
        con_agent["target_initial_stance"] = -initial_value

    return {
        "backend": {
            "type": backend_type,
            "model": model,
        },
        "data": {
            "source": "unified_json",
            "json_path": "data/Argument-Quality-Ranking-30k/arguments.json",
        },
        "debate": {
            "topic": spec.topic,
            "rounds": rounds,
            "seeding_limit": int(preset["seeding_limit"]),
            "seeding_strategy": preset["seeding_strategy"],
            "seeding_random_seed": spec.seed,
            "seed_dedup": bool(preset.get("seed_dedup", True)),
        },
        "memory": {
            "storage_mode": "episodic" if is_human_showcase else "worldview",
            "stance_method": "bayesian",
            "belief_engine_type": "bayesian",
            "opinion_change_style": "human_like" if is_human_showcase else None,
            "worldview_openness": "balanced",
            "retrieval_strategy": "weighted",
            "argument_similarity_threshold": 0.60 if is_human_showcase else 0.80,
            "self_similarity_threshold": 0.45 if is_human_showcase else 0.50,
            "cross_polarity_dedup": False,
            "confirmation_bias": 0.0,
            "allow_self_arguments": spec.allow_self_arguments,
            "strength_estimation_method": "classifier",
            "classifier_model_path": "models/argument_quality_classifier",
        },
        "simulation": {
            "rounds": rounds,
            "agents": [pro_agent, con_agent],
        },
        "evaluation": {
            "enabled": False,
        },
        "logging": {
            "log_dir": "experiments/logs",
            "level": "INFO",
            "verbosity": 1,
        },
        "validation_metadata": {
            "experiment": "two_agent_controlled",
            "preset": preset_name,
            "run_id": spec.run_id,
            "topic_class": spec.topic_class,
            "pairing": spec.pairing,
            "pro_profile": spec.pro_profile,
            "con_profile": spec.con_profile,
            "self_condition": spec.self_condition,
            "allow_self_arguments": spec.allow_self_arguments,
            "seed": spec.seed,
            "trial": spec.trial,
            "initialization_mode": initialization_mode,
            "target_initial_stance": initial_stance,
        },
    }


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def display_path(path: Path) -> str:
    """Return a compact path when possible, while supporting /tmp smoke runs."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(spec: RunSpec, raw_dir: Path) -> Optional[Dict[str, Any]]:
    result_path = raw_dir / "results.json"
    if not result_path.exists():
        return None

    result = read_json(result_path)
    history = result.get("stance_history", [])
    if not history:
        return None

    pro_name = f"Pro_{spec.pro_profile}"
    con_name = f"Con_{spec.con_profile}"
    initial = history[0]
    final = history[-1]

    pro_initial = float(initial.get(pro_name, math.nan))
    con_initial = float(initial.get(con_name, math.nan))
    pro_final = float(final.get(pro_name, math.nan))
    con_final = float(final.get(con_name, math.nan))
    pro_shift = pro_final - pro_initial
    con_shift = con_final - con_initial
    initial_distance = abs(pro_initial - con_initial)
    final_distance = abs(pro_final - con_final)

    updates = result.get("argument_updates", [])
    internalized = [u for u in updates if u.get("status") != "ignored"]
    ignored_self = [
        u
        for u in updates
        if u.get("status") == "ignored"
        and u.get("receiver") == u.get("sender")
    ]

    return {
        "run_id": spec.run_id,
        "topic": spec.topic,
        "topic_class": spec.topic_class,
        "pairing": spec.pairing,
        "pro_profile": spec.pro_profile,
        "con_profile": spec.con_profile,
        "self_condition": spec.self_condition,
        "allow_self_arguments": spec.allow_self_arguments,
        "seed": spec.seed,
        "trial": spec.trial,
        "initialization_mode": spec.initialization_mode,
        "target_initial_stance": spec.target_initial_stance,
        "rounds_recorded": len(history),
        "pro_initial": pro_initial,
        "con_initial": con_initial,
        "pro_final": pro_final,
        "con_final": con_final,
        "pro_shift": pro_shift,
        "con_shift": con_shift,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "convergence": initial_distance - final_distance,
        "dominance": ((pro_final + con_final) / 2.0) - ((pro_initial + con_initial) / 2.0),
        "movement_asymmetry": abs(pro_shift) - abs(con_shift),
        "final_extremity_mean": (abs(pro_final) + abs(con_final)) / 2.0,
        "argument_updates": len(updates),
        "arguments_internalized": len(internalized),
        "self_arguments_ignored": len(ignored_self),
        "raw_dir": display_path(raw_dir),
    }


def collect_trajectories(spec: RunSpec, raw_dir: Path) -> List[Dict[str, Any]]:
    result_path = raw_dir / "results.json"
    if not result_path.exists():
        return []

    result = read_json(result_path)
    pro_name = f"Pro_{spec.pro_profile}"
    con_name = f"Con_{spec.con_profile}"
    rows = []
    for entry in result.get("stance_history", []):
        round_idx = entry.get("round")
        rows.append(
            {
                "run_id": spec.run_id,
                "topic": spec.topic,
                "pairing": spec.pairing,
                "self_condition": spec.self_condition,
                "trial": spec.trial,
                "initialization_mode": spec.initialization_mode,
                "agent": pro_name,
                "side": "pro",
                "profile": spec.pro_profile,
                "round": round_idx,
                "stance": entry.get(pro_name),
            }
        )
        rows.append(
            {
                "run_id": spec.run_id,
                "topic": spec.topic,
                "pairing": spec.pairing,
                "self_condition": spec.self_condition,
                "trial": spec.trial,
                "initialization_mode": spec.initialization_mode,
                "agent": con_name,
                "side": "con",
                "profile": spec.con_profile,
                "round": round_idx,
                "stance": entry.get(con_name),
            }
        )
    return rows


def collect_argument_updates(spec: RunSpec, raw_dir: Path) -> List[Dict[str, Any]]:
    result_path = raw_dir / "results.json"
    if not result_path.exists():
        return []

    result = read_json(result_path)
    rows = []
    for update in result.get("argument_updates", []):
        rows.append(
            {
                "run_id": spec.run_id,
                "topic": spec.topic,
                "pairing": spec.pairing,
                "self_condition": spec.self_condition,
                "trial": spec.trial,
                "initialization_mode": spec.initialization_mode,
                "receiver": update.get("receiver"),
                "sender": update.get("sender"),
                "round": update.get("round"),
                "status": update.get("status"),
                "reason": update.get("reason"),
                "strength": update.get("strength"),
                "claim": update.get("claim"),
            }
        )
    return rows


def compute_self_amplification(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for row in summary_rows:
        key = (
            row["topic"],
            row["pairing"],
            row["pro_profile"],
            row["con_profile"],
            row.get("trial", 1),
        )
        by_key.setdefault(key, {})[row["self_condition"]] = row

    rows = []
    metrics = [
        "convergence",
        "dominance",
        "final_extremity_mean",
        "pro_shift",
        "con_shift",
        "movement_asymmetry",
    ]
    for (topic, pairing, pro_profile, con_profile, trial), variants in sorted(by_key.items()):
        if "self_on" not in variants or "self_off" not in variants:
            continue
        row = {
            "topic": topic,
            "pairing": pairing,
            "pro_profile": pro_profile,
            "con_profile": con_profile,
            "trial": trial,
        }
        for metric in metrics:
            row[f"{metric}_self_on_minus_off"] = (
                float(variants["self_on"][metric]) - float(variants["self_off"][metric])
            )
        rows.append(row)
    return rows


def maybe_import_plotting():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return plt, np, pd


def generate_figures(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    trajectory_rows: List[Dict[str, Any]],
    preset_name: str,
) -> None:
    if not summary_rows or not trajectory_rows:
        return

    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    plt, np, pd = maybe_import_plotting()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

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

    summary = pd.DataFrame(summary_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    preset = get_preset(preset_name)
    profile_order = list(preset["profiles"].keys())
    pairing_order = [pairing[0] for pairing in preset["pairings"]]
    self_condition_order = [condition[0] for condition in preset.get("self_conditions", SELF_CONDITIONS)]
    cmap = plt.get_cmap("viridis")
    pro_color = "#19747e"
    con_color = "#b34a4a"
    text_color = "#293241"
    grid_color = "#e5e7eb"
    spine_color = "#d1d5db"

    # 1. Representative trajectory examples.
    first_topic = preset["topics"][0]["topic"]
    example_pairings = [
        p
        for p in [
            "Open/Open",
            "Leaning/Leaning",
            "Committed/Committed",
            "Open/Committed",
            "Committed/Open",
            "Stubborn/Stubborn",
            "Open/Stubborn",
            "Stubborn/Open",
        ]
        if p in pairing_order
    ]
    if example_pairings:
        fig, axes = plt.subplots(
            len(example_pairings),
            len(self_condition_order),
            figsize=(8.0, 8.5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.asarray(axes)
        if len(example_pairings) == 1 and len(self_condition_order) == 1:
            axes = axes.reshape(1, 1)
        elif len(example_pairings) == 1:
            axes = axes.reshape(1, len(self_condition_order))
        elif len(self_condition_order) == 1:
            axes = axes.reshape(len(example_pairings), 1)
        for row_idx, pairing in enumerate(example_pairings):
            for col_idx, self_condition in enumerate(self_condition_order):
                ax = axes[row_idx, col_idx]
                subset = trajectories[
                    (trajectories["topic"] == first_topic)
                    & (trajectories["pairing"] == pairing)
                    & (trajectories["self_condition"] == self_condition)
                ]
                for side, color in [("pro", "#2F6F4E"), ("con", "#8E3B46")]:
                    side_df = subset[subset["side"] == side].sort_values("round")
                    if not side_df.empty:
                        ax.plot(
                            side_df["round"],
                            side_df["stance"],
                            marker="o",
                            markersize=2.8,
                            linewidth=1.8,
                            color=color,
                            label=side.upper(),
                        )
                ax.axhline(0, color="#9CA3AF", linewidth=0.8)
                ax.set_ylim(-1.05, 1.05)
                if row_idx == 0:
                    ax.set_title("self off" if self_condition == "self_off" else "self on", fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(pairing)
                if row_idx == len(example_pairings) - 1:
                    ax.set_xlabel("round")
                if row_idx == 0 and col_idx == len(self_condition_order) - 1:
                    ax.legend(loc="upper right", frameon=False)
        fig.suptitle("Representative Two-Agent Belief Trajectories", fontweight="bold")
        fig.savefig(figures_dir / "figure_two_agent_trajectory_examples.pdf", bbox_inches="tight")
        fig.savefig(figures_dir / "figure_two_agent_trajectory_examples.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # 1b. Topic5-style condition grid: rows are self conditions, columns are pairings.
    if pairing_order:
        fig, axes = plt.subplots(
            len(self_condition_order),
            len(pairing_order),
            figsize=(2.65 * len(pairing_order), 2.85 * len(self_condition_order) + 0.25),
            sharex=True,
            sharey=True,
            constrained_layout=False,
        )
        axes = np.asarray(axes)
        if len(pairing_order) == 1 and len(self_condition_order) == 1:
            axes = axes.reshape(1, 1)
        elif len(pairing_order) == 1:
            axes = axes.reshape(len(self_condition_order), 1)
        elif len(self_condition_order) == 1:
            axes = axes.reshape(1, len(pairing_order))

        for row_idx, self_condition in enumerate(self_condition_order):
            for col_idx, pairing in enumerate(pairing_order):
                ax = axes[row_idx, col_idx]
                subset = trajectories[
                    (trajectories["topic"] == first_topic)
                    & (trajectories["pairing"] == pairing)
                    & (trajectories["self_condition"] == self_condition)
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
                ax.set_xlim(-0.4, 15.6)

                for side, color in [("pro", pro_color), ("con", con_color)]:
                    side_df = subset[subset["side"] == side].copy()
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
                            fontsize=8.8,
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
                            fontsize=8.8,
                            fontweight=750,
                            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
                        )

                if row_idx == 0:
                    pro_profile, con_profile = pairing.split("/", 1)
                    ax.set_title(
                        f"{display_profile_label(pro_profile)} PRO\n{display_profile_label(con_profile)} CON",
                        fontsize=10.8,
                        fontweight=750,
                        color=text_color,
                        pad=11,
                        linespacing=0.95,
                    )
                if col_idx == 0:
                    ax.set_ylabel("Stance", fontsize=10.5, fontweight=700, color=text_color)
                if row_idx == len(self_condition_order) - 1:
                    ax.set_xlabel("Round", fontsize=10.5, fontweight=700, color=text_color)
                ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                ax.tick_params(axis="both", colors=text_color, width=0.8)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, -0.02),
                handlelength=1.8,
                prop={"weight": 650, "size": 9.0},
            )
        fig.subplots_adjust(left=0.065, right=0.992, bottom=0.22, top=0.79, wspace=0.16)
        fig.savefig(figures_dir / "figure_two_agent_topic5_style_grid.pdf", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(figures_dir / "figure_two_agent_topic5_style_grid.png", dpi=240, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)

        topic_grid_dir = figures_dir / "topic_grids"
        topic_grid_dir.mkdir(parents=True, exist_ok=True)
        for old_path in topic_grid_dir.glob("figure_two_agent_topic5_style_grid_*"):
            old_path.unlink()
        for old_path in topic_grid_dir.glob(f"figure_two_agent_{slugify(preset_name)}_topic_*"):
            old_path.unlink()
        for old_path in topic_grid_dir.glob("*_two_agent_grid.*"):
            old_path.unlink()
        for old_path in topic_grid_dir.glob("*_2agent_belief_grid.*"):
            old_path.unlink()
        for topic_info in preset["topics"]:
            for old_path in topic_grid_dir.glob(f"{short_topic_slug(topic_info['topic'])}.*"):
                old_path.unlink()
        topic_grid_pairing_order = [
            pairing
            for pairing in ["Open/Open", "Committed/Open", "Committed/Committed"]
            if pairing in pairing_order
        ]
        if not topic_grid_pairing_order:
            topic_grid_pairing_order = pairing_order
        topic_grid_self_condition = (
            "self_on" if "self_on" in self_condition_order else self_condition_order[0]
        )
        topic_grid_row_labels = {
            "Open/Open": "Open-minded/Open-minded",
            "Committed/Open": "Stubborn/Open-minded",
            "Committed/Committed": "Stubborn/Stubborn",
        }
        label_min_y = -0.98
        label_max_y = 1.02
        endpoint_label_gap = 0.28

        def compute_topic_grid_side_shift_metrics(subset: "pd.DataFrame") -> Dict[str, float]:
            rows = []
            for _run_id, run_df in subset.groupby("run_id"):
                side_values: Dict[str, Tuple[float, float]] = {}
                for side in ["pro", "con"]:
                    side_df = run_df[run_df["side"] == side].sort_values("round")
                    if side_df.empty:
                        continue
                    side_values[side] = (
                        float(side_df["stance"].iloc[0]),
                        float(side_df["stance"].iloc[-1]),
                    )
                if "pro" not in side_values or "con" not in side_values:
                    continue
                pro_initial, pro_final = side_values["pro"]
                con_initial, con_final = side_values["con"]
                rows.append(
                    {
                        "pro_initial": pro_initial,
                        "con_initial": con_initial,
                        "pro_final": pro_final,
                        "con_final": con_final,
                    }
                )
            if not rows:
                return {}
            metrics = pd.DataFrame(rows)
            pro_initial = float(metrics["pro_initial"].mean())
            con_initial = float(metrics["con_initial"].mean())
            pro_final = float(metrics["pro_final"].mean())
            con_final = float(metrics["con_final"].mean())
            return {
                "pro_initial": pro_initial,
                "con_initial": con_initial,
                "pro_final": pro_final,
                "con_final": con_final,
                "pro_change": pro_final - pro_initial,
                "con_change": con_final - con_initial,
            }

        def draw_side_shift_rulers(ax: Any, metrics: Dict[str, float]) -> None:
            transform = ax.get_yaxis_transform()
            cap_width = 0.018
            ruler_x = 1.035

            def draw_ruler(
                initial: float,
                final: float,
                color: str,
                linewidth: float,
                alpha: float,
            ) -> float:
                low, high = sorted([initial, final])
                ax.plot(
                    [ruler_x, ruler_x],
                    [low, high],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="butt",
                    transform=transform,
                    clip_on=False,
                    zorder=6,
                )
                for y_value, tick_width in [
                    (low, cap_width),
                    ((low + high) / 2.0, cap_width * 0.58),
                    (high, cap_width),
                ]:
                    ax.plot(
                        [ruler_x - tick_width, ruler_x + tick_width],
                        [y_value, y_value],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        solid_capstyle="butt",
                        transform=transform,
                        clip_on=False,
                        zorder=6,
                    )
                return min(label_max_y, max(label_min_y, (initial + final) / 2.0))

            pro_label_y = draw_ruler(
                metrics["pro_initial"],
                metrics["pro_final"],
                pro_color,
                1.55,
                1.0,
            )
            con_label_y = draw_ruler(
                metrics["con_initial"],
                metrics["con_final"],
                con_color,
                1.55,
                1.0,
            )
            for label_y, change_key, color in [
                (pro_label_y, "pro_change", pro_color),
                (con_label_y, "con_change", con_color),
            ]:
                ax.text(
                    ruler_x + 0.032,
                    label_y,
                    f"{metrics[change_key]:+.2f}",
                    transform=transform,
                    ha="left",
                    va="center",
                    fontsize=7.8,
                    fontweight=760,
                    color=color,
                    clip_on=False,
                    zorder=7,
                    bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.80},
                )

        def spread_topic_endpoint_labels(values: List[float]) -> List[float]:
            if not values:
                return []
            gap = min(endpoint_label_gap, (label_max_y - label_min_y) / max(1, len(values) - 1))
            positions = [max(label_min_y, min(label_max_y, value)) for value in values]
            for idx in range(1, len(positions)):
                positions[idx] = max(positions[idx], positions[idx - 1] + gap)
            if positions[-1] > label_max_y:
                overflow = positions[-1] - label_max_y
                positions = [position - overflow for position in positions]
            for idx in range(len(positions) - 2, -1, -1):
                positions[idx] = min(positions[idx], positions[idx + 1] - gap)
            if positions[0] < label_min_y:
                underflow = label_min_y - positions[0]
                positions = [position + underflow for position in positions]
            return positions

        for topic in sorted(trajectories["topic"].dropna().unique()):
            fig, axes = plt.subplots(
                len(topic_grid_pairing_order),
                1,
                figsize=(3.18, 1.58 * len(topic_grid_pairing_order) + 0.14),
                sharex=True,
                sharey=True,
                constrained_layout=False,
            )
            axes = np.atleast_1d(axes)

            for row_idx, pairing in enumerate(topic_grid_pairing_order):
                ax = axes[row_idx]
                subset = trajectories[
                    (trajectories["topic"] == topic)
                    & (trajectories["pairing"] == pairing)
                    & (trajectories["self_condition"] == topic_grid_self_condition)
                ]
                side_shift_metrics = compute_topic_grid_side_shift_metrics(subset)
                ax.axhspan(-0.1, 0.1, color="#f3f4f6", alpha=1.0, linewidth=0, zorder=0)
                ax.axhline(0, color="#9CA3AF", linestyle="--", linewidth=0.9, alpha=0.8, zorder=1)
                ax.grid(axis="y", color=grid_color, linewidth=0.8, alpha=0.78)
                ax.grid(axis="x", color=grid_color, linewidth=0.58, alpha=0.45)
                ax.set_axisbelow(True)
                for spine in ["left", "bottom"]:
                    ax.spines[spine].set_color(spine_color)
                    ax.spines[spine].set_linewidth(0.9)
                ax.set_ylim(-1.08, 1.08)
                ax.set_xlim(-0.3, 15.55)

                endpoint_labels = []
                for side, color in [("pro", pro_color), ("con", con_color)]:
                    side_df = subset[subset["side"] == side].copy()
                    if side_df.empty:
                        continue
                    for _run_id, run_df in side_df.groupby("run_id"):
                        run_df = run_df.sort_values("round")
                        ax.plot(
                            run_df["round"],
                            run_df["stance"],
                            color=color,
                            linewidth=1.18,
                            alpha=0.42,
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
                    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
                    rounds = grouped["round"].to_numpy()
                    mean = grouped["mean"].to_numpy()
                    se = grouped["se"].to_numpy()
                    ax.fill_between(rounds, mean - se, mean + se, color=color, alpha=0.08, linewidth=0, zorder=1)
                    ax.plot(rounds, mean, color=color, linewidth=2.85, solid_capstyle="round", label=f"{side.upper()} mean", zorder=3)

                    if len(rounds) > 0:
                        endpoint_offset = 0.16 if side == "pro" else -0.16
                        preferred_label_y = max(label_min_y, min(label_max_y, float(mean[-1]) + endpoint_offset))
                        endpoint_labels.append(
                            (
                                preferred_label_y,
                                color,
                                float(rounds[-1]),
                                float(mean[-1]),
                                f"{mean[-1]:+.2f}",
                            )
                        )
                        ax.annotate(
                            f"{mean[0]:+.2f}",
                            xy=(rounds[0], mean[0]),
                            xytext=(5, 7 if side == "pro" else -13),
                            textcoords="offset points",
                            ha="left",
                            color=color,
                            fontsize=8.2,
                            fontweight=750,
                            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
                        )

                if endpoint_labels:
                    endpoint_labels.sort(key=lambda item: item[0])
                    label_positions = spread_topic_endpoint_labels(
                        [preferred_y for preferred_y, _color, _x_value, _y_value, _label in endpoint_labels]
                    )
                    for (_preferred_y, color, x_value, y_value, label), label_y in zip(endpoint_labels, label_positions):
                        ax.annotate(
                            label,
                            xy=(x_value, y_value),
                            xycoords="data",
                            xytext=(15.08, label_y),
                            textcoords=("data", "data"),
                            arrowprops=(
                                {
                                    "arrowstyle": "-",
                                    "color": color,
                                    "linewidth": 0.8,
                                    "alpha": 0.55,
                                    "shrinkA": 0,
                                    "shrinkB": 4,
                                }
                                if abs(label_y - y_value) > 0.035
                                else None
                            ),
                            annotation_clip=False,
                            clip_on=False,
                            ha="right",
                            va="center",
                            color=color,
                            fontsize=8.4,
                            fontweight=750,
                            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.58},
                            zorder=4,
                        )

                if side_shift_metrics:
                    draw_side_shift_rulers(ax, side_shift_metrics)

                ax.set_title(
                    topic_grid_row_labels.get(pairing, display_pairing_label(pairing)),
                    fontsize=10.0,
                    fontweight=780,
                    color=text_color,
                    pad=5,
                )
                if row_idx == len(topic_grid_pairing_order) - 1:
                    ax.set_xlabel("Round", fontsize=9.0, fontweight=720, color=text_color)
                else:
                    ax.tick_params(axis="x", bottom=False, labelbottom=False)
                ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                ax.set_xticks([0, 5, 10, 15])
                ax.tick_params(axis="x", colors=text_color, width=0.85, labelsize=8.8)
                ax.tick_params(axis="y", colors=text_color, width=0.85, labelsize=8.0)

            fig.subplots_adjust(
                left=0.12,
                right=0.69,
                bottom=0.10,
                top=0.965,
                hspace=0.34,
            )
            stem = f"{short_topic_slug(topic)}_2agent_belief_grid"
            fig.savefig(topic_grid_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
            fig.savefig(topic_grid_dir / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)

    # 2. Profile-pair heatmap for convergence.
    fig, axes = plt.subplots(1, len(self_condition_order), figsize=(3.8 * len(self_condition_order), 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, self_condition in zip(axes, self_condition_order):
        n_profiles = len(profile_order)
        mat = np.full((n_profiles, n_profiles), np.nan)
        sub = summary[summary["self_condition"] == self_condition]
        grouped = sub.groupby(["pro_profile", "con_profile"])["convergence"].mean()
        for i, pro in enumerate(profile_order):
            for j, con in enumerate(profile_order):
                if (pro, con) in grouped:
                    mat[i, j] = grouped[(pro, con)]
        if np.all(np.isnan(mat)):
            im = ax.imshow(np.zeros_like(mat), cmap=cmap, vmin=0, vmax=1, alpha=0.15)
        else:
            im = ax.imshow(mat, cmap=cmap, vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_xticks(range(n_profiles), [display_profile_label(p) for p in profile_order], rotation=25, ha="right")
        ax.set_yticks(range(n_profiles), [display_profile_label(p) for p in profile_order])
        ax.set_xlabel("Con profile")
        ax.set_ylabel("Pro profile")
        ax.set_title("opponent-only" if self_condition == "self_off" else "self-reinforcing", fontweight="bold")
        for i in range(n_profiles):
            for j in range(n_profiles):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontweight="bold", color="white")
        if not np.all(np.isnan(mat)):
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
            cbar.ax.set_ylabel("convergence", rotation=90)
    fig.suptitle("Profile-Pair Convergence", fontweight="bold")
    fig.savefig(figures_dir / "figure_two_agent_profile_pair_heatmap.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "figure_two_agent_profile_pair_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 3. Self-reinforcement comparison.
    self_amp = pd.DataFrame(compute_self_amplification(summary_rows))
    if not self_amp.empty:
        amp_by_pairing = self_amp.groupby("pairing")[
            ["convergence_self_on_minus_off", "final_extremity_mean_self_on_minus_off"]
        ].mean()
        fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
        x = np.arange(len(amp_by_pairing))
        width = 0.36
        ax.bar(
            x - width / 2,
            amp_by_pairing["convergence_self_on_minus_off"],
            width,
            label="convergence",
            color="#2F6F4E",
        )
        ax.bar(
            x + width / 2,
            amp_by_pairing["final_extremity_mean_self_on_minus_off"],
            width,
            label="final extremity",
            color="#A9793B",
        )
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks(x, [display_pairing_label(p) for p in amp_by_pairing.index], rotation=30, ha="right")
        ax.set_ylabel("self on - self off")
        ax.set_title("Self-Reinforcement Effect", fontweight="bold")
        ax.legend(frameon=False)
        fig.savefig(figures_dir / "figure_two_agent_self_reinforcement.pdf", bbox_inches="tight")
        fig.savefig(figures_dir / "figure_two_agent_self_reinforcement.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # 4. Topic robustness panel.
    metrics = ["convergence", "dominance", "final_extremity_mean"]
    metric_labels = ["convergence", "dominance", "final extremity"]
    topic_summary = summary.groupby(["topic_class", "self_condition"])[metrics].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), constrained_layout=True)
    for ax, metric, label in zip(axes, metrics, metric_labels):
        pivot = topic_summary.pivot(index="topic_class", columns="self_condition", values=metric)
        x = np.arange(len(pivot))
        width = 0.36
        ax.bar(x - width / 2, pivot.get("self_off", np.nan), width, label="self off", color="#577590")
        ax.bar(x + width / 2, pivot.get("self_on", np.nan), width, label="self on", color="#A9793B")
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x, pivot.index, rotation=30, ha="right")
    axes[0].set_ylabel("mean over pairings")
    axes[-1].legend(frameon=False)
    fig.suptitle("Topic Robustness", fontweight="bold")
    if len(topic_summary) > 0:
        fig.savefig(figures_dir / "figure_two_agent_topic_robustness.pdf", bbox_inches="tight")
        fig.savefig(figures_dir / "figure_two_agent_topic_robustness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    failed: List[Dict[str, Any]],
    backend_type: str,
    model: str,
    rounds: int,
    mode_label: str,
    preset_name: str,
) -> None:
    completed = len(summary_rows)
    preset = get_preset(preset_name)
    total = len(build_specs(preset_name))
    lines = [
        "# Controlled Two-Agent Debate Experiments",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {mode_label}",
        f"Preset: `{preset_name}`",
        f"Backend: `{backend_type}` / `{model}`",
        f"Rounds: {rounds}",
        f"Completed runs with results: {completed}/{total}",
        f"Failed runs: {len(failed)}",
        f"Initialization mode: `{preset.get('initialization_mode', 'target_controlled')}`",
        f"Target initial stance: `{preset.get('initial_stance')}`",
        "",
        "## Design",
        "",
        preset["description"],
        "",
        "AQR topics are used for controlled mechanics. DEBATE remains the human-grounded validation source.",
        "",
        "| Profile | Prior strength | Evidence responsiveness | Selective uptake | K update | K_a anchor | Confirmation bias | Self dedup |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for name, profile in preset["profiles"].items():
        axes = profile.get("framework_axes", {})
        lines.append(
            "| "
            f"{name} | "
            f"{axes.get('prior_strength', '')} | "
            f"{axes.get('evidence_responsiveness', '')} | "
            f"{axes.get('selective_uptake', '')} | "
            f"{profile['K']:.2f} | "
            f"{profile['K_a']:.2f} | "
            f"{profile.get('confirmation_bias', 0.0):.2f} | "
            f"{profile.get('self_similarity_threshold', '')} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "| File | Description |",
            "|---|---|",
            "| `summary.csv` | one row per completed debate |",
            "| `stance_trajectories.csv` | round-by-round stance trajectories |",
            "| `argument_uptake.csv` | argument extraction/internalization log |",
            "| `self_amplification.csv` | matched self-on minus self-off metrics |",
            "| `configs/` | generated YAML configs; topic-separated presets use `configs/<topic_slug>/` |",
            "| `raw_runs/` | raw transcripts, memory dumps, and `results.json`; topic-separated presets use `raw_runs/<topic_slug>/` |",
            "| `raw_runs/<run_id>/live_stance_trajectory.png` | optional live-updated trajectory when `--live-plot` is used |",
            "| `raw_runs/<run_id>/memory_reports/` | per-agent memory diff reports in compact artifact mode |",
            "| `figures/` | generated paper-facing figures |",
            "",
            "## Reproduce",
            "",
            "```bash",
            f"python validation/scripts/run_two_agent_controlled.py --preset {preset_name}",
            "```",
            "",
            "For a config-only dry run:",
            "",
            "```bash",
            f"python validation/scripts/run_two_agent_controlled.py --preset {preset_name} --dry-run",
            "```",
        ]
    )

    if failed:
        lines.extend(["", "## Failed Runs", "", "| run_id | error |", "|---|---|"])
        for row in failed:
            lines.append(f"| `{row['run_id']}` | {row['error']} |")

    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def aggregate_outputs(output_dir: Path, specs: Iterable[RunSpec], preset_name: str) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    uptake_rows: List[Dict[str, Any]] = []

    for spec in specs:
        raw_dir = raw_dir_for(output_dir, spec, preset_name)
        summary = summarize_run(spec, raw_dir)
        if summary:
            summary_rows.append(summary)
        trajectory_rows.extend(collect_trajectories(spec, raw_dir))
        uptake_rows.extend(collect_argument_updates(spec, raw_dir))

    if summary_rows:
        write_csv(output_dir / "summary.csv", summary_rows)
        write_csv(output_dir / "stance_trajectories.csv", trajectory_rows)
        write_csv(output_dir / "argument_uptake.csv", uptake_rows)
        write_csv(output_dir / "self_amplification.csv", compute_self_amplification(summary_rows))
        generate_figures(output_dir, summary_rows, trajectory_rows, preset_name)

    return summary_rows


def execute_config_run(config_path: str, raw_dir: str, resume: bool) -> Optional[str]:
    from src.simulation.run_simulation import run_simulation

    raw_path = Path(raw_dir)
    results_path = raw_path / "results.json"
    if resume and results_path.exists():
        return None
    if raw_path.exists() and not resume:
        shutil.rmtree(raw_path)
    raw_path.mkdir(parents=True, exist_ok=True)
    try:
        run_simulation(config_path, log_dir_override=raw_dir)
        return None
    except Exception as exc:
        return str(exc)


def run(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "validation" / "results" / f"two_agent_controlled_{args.preset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_runs").mkdir(parents=True, exist_ok=True)

    specs = build_specs(args.preset)
    if args.max_runs:
        specs = specs[: args.max_runs]
    if uses_topic_subdirs(args.preset):
        for spec in specs:
            config_path_for(output_dir, spec, args.preset).parent.mkdir(parents=True, exist_ok=True)
            raw_dir_for(output_dir, spec, args.preset).parent.mkdir(parents=True, exist_ok=True)

    failed: List[Dict[str, Any]] = []

    if args.aggregate_only:
        specs_for_aggregation = specs
        summary_rows = aggregate_outputs(output_dir, specs_for_aggregation, args.preset)
        write_readme(
            output_dir=output_dir,
            summary_rows=summary_rows,
            failed=failed,
            backend_type=args.backend_type,
            model=args.model,
            rounds=args.rounds,
            mode_label="aggregate-only over existing raw runs",
            preset_name=args.preset,
        )
        print(f"\nOutput directory: {output_dir}")
        print(f"Completed results: {len(summary_rows)}/{len(specs_for_aggregation)}")
        return 0

    if not args.dry_run and args.workers <= 1:
        from src.simulation.run_simulation import run_simulation

    run_jobs: List[Dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        config = build_config(spec, args.backend_type, args.model, args.rounds, args.preset)
        config.setdefault("logging", {})["artifact_mode"] = args.artifact_mode
        if args.live_plot:
            config["logging"]["live_plot"] = True
            config.setdefault("simulation", {})["live_plot"] = True
        config_path = config_path_for(output_dir, spec, args.preset)
        raw_dir = raw_dir_for(output_dir, spec, args.preset)
        write_yaml(config_path, config)

        print(f"[{index}/{len(specs)}] {spec.run_id}")
        if args.dry_run:
            continue

        results_path = raw_dir / "results.json"
        if args.resume and results_path.exists():
            print("  skipped existing results")
            continue

        if args.workers > 1:
            run_jobs.append(
                {
                    "run_id": spec.run_id,
                    "config_path": str(config_path),
                    "raw_dir": str(raw_dir),
                }
            )
            continue

        try:
            if raw_dir.exists() and not args.resume:
                shutil.rmtree(raw_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)
            run_simulation(str(config_path), log_dir_override=str(raw_dir))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append({"run_id": spec.run_id, "error": str(exc)})

    if run_jobs:
        print(f"Running {len(run_jobs)} configs with {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(
                    execute_config_run,
                    job["config_path"],
                    job["raw_dir"],
                    args.resume,
                ): job
                for job in run_jobs
            }
            completed_jobs = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                completed_jobs += 1
                try:
                    error = future.result()
                except Exception as exc:
                    error = str(exc)
                if error:
                    print(f"[{completed_jobs}/{len(run_jobs)}] {job['run_id']} ERROR: {error}")
                    failed.append({"run_id": job["run_id"], "error": error})
                else:
                    print(f"[{completed_jobs}/{len(run_jobs)}] {job['run_id']} done")

    summary_rows = aggregate_outputs(output_dir, specs, args.preset)
    write_readme(
        output_dir=output_dir,
        summary_rows=summary_rows,
        failed=failed,
        backend_type=args.backend_type,
        model=args.model,
        rounds=args.rounds,
        mode_label="dry-run config generation" if args.dry_run else "executed simulations",
        preset_name=args.preset,
    )

    if failed:
        write_csv(output_dir / "failed_runs.csv", failed, fieldnames=["run_id", "error"])

    print(f"\nOutput directory: {output_dir}")
    print(f"Completed results: {len(summary_rows)}/{len(specs)}")
    if failed:
        print(f"Failed runs: {len(failed)}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Output directory. Defaults to validation/results/two_agent_controlled_<timestamp>.")
    parser.add_argument("--backend-type", default="openai", choices=["openai", "openrouter", "local", "mock"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--preset", default="full", choices=sorted(PRESETS), help="Experiment preset to generate/run.")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and README without executing simulations.")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate existing raw runs without generating or executing runs.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of generated/executed runs for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Skip runs that already have raw results.")
    parser.add_argument("--live-plot", action="store_true", help="Update live_stance_history.csv/json and live_stance_trajectory.png during each run.")
    parser.add_argument("--artifact-mode", default="compact", choices=["compact", "full"], help="Use compact to skip redundant per-run artifacts; full preserves legacy outputs.")
    parser.add_argument("--workers", type=int, default=1, help="Number of independent runs to execute in parallel.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
