"""
Run LLM ArgumentExtractionAgent on the frozen AQR-30k judgement sample.

This is the API-backed companion to evaluate_judgement_layer_aqr.py. It uses
the same human-labelled Argument-Quality-Ranking-30k sample, calls the real
ArgumentExtractionAgent, caches every result, and evaluates:

- extraction rate for single-argument inputs,
- over/under-splitting,
- polarity accuracy against human stance_WA,
- strength reliability against WA / MACE-P quality labels.

The cache is keyed by topic + text, because the extraction prompt is
topic-aware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/review_artifact_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/review_artifact_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.agents.extractor_agent import ArgumentExtractionAgent
from src.backend.factory import create_backend
from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE


DEFAULT_SAMPLE = (
    "validation/results/judgement_layer_aqr/aqr_test_20260426_133422/"
    "sample_for_extractor.csv"
)
DEFAULT_OUTPUT_ROOT = "validation/results/judgement_layer_aqr_llm"
DEFAULT_CACHE_DIR = "validation/data/cache/extractions/aqr"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _cache_key(topic: str, text: str) -> str:
    return hashlib.sha256(f"{topic}|||{text}".encode("utf-8")).hexdigest()[:20]


def _load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1))


def _record_to_dict(record) -> dict:
    return {
        "topic_id": record.topic_id,
        "claim": record.claim,
        "polarity": record.polarity,
        "source_role": record.source_role,
        "support_strength": record.support_strength,
        "support_strength_llm": record.support_strength_llm,
        "support_strength_classifier": record.support_strength_classifier,
        "evidence_snippets": record.evidence_snippets,
        "category_tags": record.category_tags,
    }


def _safe_corr(df: pd.DataFrame, x: str, y: str, method: str) -> float:
    valid = df[[x, y]].dropna()
    if len(valid) < 2:
        return float("nan")
    return float(valid[x].corr(valid[y], method=method))


def _strength_metrics(df: pd.DataFrame, target: str, pred: str) -> dict:
    valid = df[[target, pred]].dropna()
    if valid.empty:
        return {
            "n": 0,
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
        }
    err = valid[pred] - valid[target]
    return {
        "n": int(len(valid)),
        "pearson_r": _safe_corr(valid, target, pred, "pearson"),
        "spearman_rho": _safe_corr(valid, target, pred, "spearman"),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
    }


def _classification_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    valid = df[(df["n_extracted"] > 0) & df[pred_col].isin(["affirmative", "negative"])]
    labels = ["affirmative", "negative"]
    if valid.empty:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "per_label": {},
        }

    per_label = {}
    f1s = []
    precisions = []
    recalls = []
    for label in labels:
        tp = int(((valid["human_polarity"] == label) & (valid[pred_col] == label)).sum())
        fp = int(((valid["human_polarity"] != label) & (valid[pred_col] == label)).sum())
        fn = int(((valid["human_polarity"] == label) & (valid[pred_col] != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision == precision and recall == recall and (precision + recall)
            else float("nan")
        )
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "n": int(len(valid)),
        "accuracy": float((valid[pred_col] == valid["human_polarity"]).mean()),
        "macro_precision": float(np.nanmean(precisions)),
        "macro_recall": float(np.nanmean(recalls)),
        "macro_f1": float(np.nanmean(f1s)),
        "per_label": per_label,
    }


def _extraction_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n": 0,
            "extraction_rate": float("nan"),
            "exact_one_rate": float("nan"),
            "oversplit_rate": float("nan"),
            "zero_extract_rate": float("nan"),
            "mean_extracted": float("nan"),
        }
    return {
        "n": int(len(df)),
        "extraction_rate": float((df["n_extracted"] > 0).mean()),
        "exact_one_rate": float((df["n_extracted"] == 1).mean()),
        "oversplit_rate": float((df["n_extracted"] > 1).mean()),
        "zero_extract_rate": float((df["n_extracted"] == 0).mean()),
        "mean_extracted": float(df["n_extracted"].mean()),
    }


def _summarize_extraction(row: pd.Series, extracted: list[dict]) -> dict:
    signed_pressure = 0.0
    strengths = []
    strongest = None
    for item in extracted:
        strength = float(item.get("support_strength_llm", item.get("support_strength", 0.0)))
        strengths.append(strength)
        if strongest is None or strength > float(
            strongest.get("support_strength_llm", strongest.get("support_strength", 0.0))
        ):
            strongest = item
        polarity = item.get("polarity")
        if polarity == POLARITY_AFFIRMATIVE:
            signed_pressure += strength
        elif polarity == POLARITY_NEGATIVE:
            signed_pressure -= strength

    if signed_pressure > 0:
        aggregate_polarity = "affirmative"
    elif signed_pressure < 0:
        aggregate_polarity = "negative"
    else:
        aggregate_polarity = ""

    strongest_polarity = strongest.get("polarity", "") if strongest else ""
    strongest_strength = (
        float(strongest.get("support_strength_llm", strongest.get("support_strength", np.nan)))
        if strongest
        else np.nan
    )

    return {
        "text_hash": row["text_hash"],
        "topic": row["topic"],
        "argument": row["argument"],
        "human_polarity": row["human_polarity"],
        "stance_WA": int(row["stance_WA"]),
        "stance_WA_conf": float(row["stance_WA_conf"]),
        "WA": float(row["WA"]),
        "MACE-P": float(row["MACE-P"]),
        "quality_target": float(row["quality_target"]),
        "n_extracted": len(extracted),
        "pred_polarity_aggregate": aggregate_polarity,
        "pred_polarity_strongest": strongest_polarity,
        "polarity_match_aggregate": aggregate_polarity == row["human_polarity"],
        "polarity_match_strongest": strongest_polarity == row["human_polarity"],
        "signed_pressure": signed_pressure,
        "mean_strength_llm": float(np.mean(strengths)) if strengths else np.nan,
        "max_strength_llm": float(np.max(strengths)) if strengths else np.nan,
        "strongest_strength_llm": strongest_strength,
        "claims": json.dumps([item.get("claim", "") for item in extracted], ensure_ascii=False),
        "polarities": json.dumps([item.get("polarity", "") for item in extracted]),
        "strengths_llm": json.dumps(strengths),
    }


def _write_plot(df: pd.DataFrame, output_dir: Path) -> None:
    valid = df.dropna(subset=["quality_target", "strongest_strength_llm"])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    colors = valid["human_polarity"].map({"affirmative": "#19747e", "negative": "#d8873a"})
    ax.scatter(
        valid["quality_target"],
        valid["strongest_strength_llm"],
        s=30,
        alpha=0.55,
        color=colors,
        edgecolor="none",
    )
    ax.plot([0, 1], [0, 1], color="#293241", linewidth=1.4, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("human quality target: mean(WA, MACE-P)")
    ax.set_ylabel("LLM strongest-claim strength")
    ax.set_title("LLM Judgement Strength on AQR-30k", weight="bold")
    ax.grid(True, color="#d8dee9", linewidth=0.8, alpha=0.75)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(output_dir / f"figure_llm_strength_reliability.{ext}", dpi=220)
    plt.close(fig)


def _write_report(output_dir: Path, args: argparse.Namespace, df: pd.DataFrame, metrics: dict) -> None:
    extraction = metrics["extraction"]
    aggregate = metrics["aggregate_polarity"]
    strongest = metrics["strongest_polarity"]
    strength = metrics["strength"]
    lines = [
        "# LLM Judgement Layer Evaluation: AQR-30k",
        "",
        "This run calls the real `ArgumentExtractionAgent` on the frozen",
        "Argument-Quality-Ranking-30k sample and compares outputs to human",
        "stance and quality labels.",
        "",
        "## Inputs",
        "",
        f"- Sample: `{args.sample_csv}`",
        f"- Backend: `{args.backend_type}`",
        f"- Model: `{args.model}`",
        f"- Rows evaluated in this run: `{len(df)}`",
        f"- Cache path: `{args.cache_path}`",
        "",
        "## Extraction And Polarity",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Extraction rate (`n_extracted > 0`) | {extraction['extraction_rate']:.3f} |",
        f"| Exact-one extraction rate | {extraction['exact_one_rate']:.3f} |",
        f"| Over-split rate (`n_extracted > 1`) | {extraction['oversplit_rate']:.3f} |",
        f"| Mean extracted claims | {extraction['mean_extracted']:.2f} |",
        "",
        "## Polarity Recovery",
        "",
        "| Polarity rule | N | Accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Aggregate signed pressure | {aggregate['n']} | {aggregate['accuracy']:.3f} | "
            f"{aggregate['macro_precision']:.3f} | {aggregate['macro_recall']:.3f} | "
            f"{aggregate['macro_f1']:.3f} |"
        ),
        (
            f"| Strongest extracted claim | {strongest['n']} | {strongest['accuracy']:.3f} | "
            f"{strongest['macro_precision']:.3f} | {strongest['macro_recall']:.3f} | "
            f"{strongest['macro_f1']:.3f} |"
        ),
        "",
        "## Strength Reliability",
        "",
        "| Target | N | Pearson r | Spearman rho | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target, vals in strength.items():
        lines.append(
            "| {target} | {n} | {pearson:.4f} | {spearman:.4f} | {mae:.4f} | {rmse:.4f} | {bias:.4f} |".format(
                target=target,
                n=vals["n"],
                pearson=vals["pearson_r"],
                spearman=vals["spearman_rho"],
                mae=vals["mae"],
                rmse=vals["rmse"],
                bias=vals["bias"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `llm_extraction_predictions.csv`: row-level predictions and extracted claims.",
            "- `llm_metrics.json`: machine-readable metric summary.",
            "- `figure_llm_strength_reliability.pdf`: LLM strength versus human quality.",
            "",
            "## Notes",
            "",
            "AQR-30k rows are already single human-written arguments, so exact-one",
            "extraction is desirable. DEBATE dialogue still needs a separate stress",
            "test for multi-claim messages and moderation/exception language.",
        ]
    )
    (output_dir / "llm_judgement_aqr_report.md").write_text("\n".join(lines))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", default=DEFAULT_SAMPLE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--backend-type", default="openrouter", choices=["openrouter", "openai", "local"])
    parser.add_argument("--model", default="anthropic/claude-sonnet-4")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    df = pd.read_csv(args.sample_csv)
    if args.limit and args.limit > 0:
        df = df.iloc[args.offset : args.offset + args.limit].copy()
    elif args.offset:
        df = df.iloc[args.offset :].copy()
    else:
        df = df.copy()

    model_slug = _slug(args.model)
    cache_path = Path(args.cache_dir) / f"aqr_extraction_cache_{model_slug}.json"
    args.cache_path = str(cache_path)
    cache = _load_cache(cache_path)
    config = {
        "backend": {
            "type": args.backend_type,
            "model": args.model,
        },
        "memory": {
            "strength_estimation_method": "llm",
            "classifier_model_path": "__disabled__",
        },
    }
    backend = create_backend(config)
    extractors: dict[str, ArgumentExtractionAgent] = {}
    rows = []
    errors = []

    total_rows = len(df)
    for row_number, (idx, row) in enumerate(df.iterrows(), start=1):
        topic = str(row["topic"])
        text = str(row["argument"])
        key = _cache_key(topic, text)
        if key not in cache:
            print(
                f"[{row_number}/{total_rows}] extracting {row['text_hash']} "
                f"({topic[:56]})",
                flush=True,
            )
            if topic not in extractors:
                extractors[topic] = ArgumentExtractionAgent(backend, topic_id=topic, config=config)
            try:
                records = extractors[topic].extract(text, source_role="aqr_argument")
                cache[key] = {
                    "topic": topic,
                    "text_hash": row["text_hash"],
                    "argument": text,
                    "records": [_record_to_dict(record) for record in records],
                    "error": "",
                }
            except Exception as exc:
                cache[key] = {
                    "topic": topic,
                    "text_hash": row["text_hash"],
                    "argument": text,
                    "records": [],
                    "error": repr(exc),
                }
                errors.append({"idx": int(idx), "topic": topic, "error": repr(exc)})
            if args.save_every <= 1 or len(cache) % args.save_every == 0:
                _save_cache(cache_path, cache)
        else:
            print(
                f"[{row_number}/{total_rows}] cached {row['text_hash']} "
                f"({topic[:56]})",
                flush=True,
            )

        cached = cache[key]
        rows.append(_summarize_extraction(row, cached.get("records", [])))

    _save_cache(cache_path, cache)
    pred_df = pd.DataFrame(rows)
    metrics = {
        "extraction": _extraction_metrics(pred_df),
        "aggregate_polarity": _classification_metrics(pred_df, "pred_polarity_aggregate"),
        "strongest_polarity": _classification_metrics(pred_df, "pred_polarity_strongest"),
        "strength": {
        "WA": _strength_metrics(pred_df, "WA", "strongest_strength_llm"),
        "MACE-P": _strength_metrics(pred_df, "MACE-P", "strongest_strength_llm"),
        "quality_target": _strength_metrics(
            pred_df, "quality_target", "strongest_strength_llm"
        ),
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"aqr_llm_{model_slug}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_dir / "llm_extraction_predictions.csv", index=False)
    (output_dir / "llm_metrics.json").write_text(json.dumps(metrics, indent=2))
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)
    _write_plot(pred_df, output_dir)
    _write_report(output_dir, args, pred_df, metrics)
    print(f"Saved LLM AQR judgement evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
