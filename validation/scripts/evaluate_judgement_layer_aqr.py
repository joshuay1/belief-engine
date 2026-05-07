"""
Evaluate judgement-layer strength scoring on Argument-Quality-Ranking-30k.

This script uses the human-annotated argument-quality dataset already present
in data/Argument-Quality-Ranking-30k. It deliberately separates what can be
validated locally from what requires an LLM extraction pass:

1. Local: strength scoring against human WA / MACE-P quality labels.
2. Local: frozen balanced sample for future polarity/extraction evaluation.
3. Local: coverage check for existing extraction caches.

The dataset provides one argument per row, so it is appropriate for strength
and stance/polarity validation once extractor outputs are available. It is not
a complete test of messy multi-argument dialogue extraction.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/review_artifact_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/review_artifact_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = "data/Argument-Quality-Ranking-30k"
DEFAULT_MODEL_DIR = "models/argument_quality_classifier"
DEFAULT_OUTPUT_ROOT = "validation/results/judgement_layer_aqr"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_split(data_dir: str, split: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)
    df = df.dropna(subset=["argument", "topic", "WA", "MACE-P", "stance_WA"])
    df["argument"] = df["argument"].astype(str)
    df["topic"] = df["topic"].astype(str)
    df["stance_WA"] = df["stance_WA"].astype(int)
    return df


def _balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    per_stance = n // 2
    parts = []
    for stance in [-1, 1]:
        sub = df[df["stance_WA"] == stance]
        take = min(per_stance, len(sub))
        parts.append(sub.sample(n=take, random_state=seed + stance + 2))
    out = pd.concat(parts, ignore_index=True)
    remaining = n - len(out)
    if remaining > 0:
        rest = df.drop(out.index, errors="ignore")
        out = pd.concat(
            [out, rest.sample(n=min(remaining, len(rest)), random_state=seed + 17)],
            ignore_index=True,
        )
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _predict_strengths(arguments: list[str], model_dir: str, batch_size: int) -> list[float]:
    import sys

    model_path = Path(model_dir).resolve()
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))
    from argument_quality_model import ArgumentQualityModel

    model = ArgumentQualityModel(str(model_path), device="cpu")
    preds: list[float] = []
    for start in range(0, len(arguments), batch_size):
        batch = arguments[start : start + batch_size]
        preds.extend(float(v) for v in model.predict_batch(batch))
    return preds


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "n": int(y_true.notna().sum()),
        "pearson_r": float(y_true.corr(y_pred, method="pearson")),
        "spearman_rho": float(y_true.corr(y_pred, method="spearman")),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
    }


def _group_metrics(df: pd.DataFrame, group_col: str, min_n: int) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col):
        if len(sub) < min_n:
            continue
        vals = _metrics(sub["quality_target"], sub["classifier_strength"])
        vals[group_col] = group
        rows.append(vals)
    if not rows:
        return pd.DataFrame()
    cols = [group_col, "n", "pearson_r", "spearman_rho", "mae", "rmse", "bias"]
    return pd.DataFrame(rows)[cols].sort_values("pearson_r", ascending=False)


def _cache_coverage(df: pd.DataFrame, cache_glob: str) -> pd.DataFrame:
    hashes = {_hash_text(text) for text in df["argument"]}
    rows = []
    for path in sorted(glob.glob(cache_glob)):
        with open(path) as f:
            cache = json.load(f)
        overlap = hashes & set(cache.keys())
        rows.append(
            {
                "cache_file": os.path.basename(path),
                "entries": len(cache),
                "overlap": len(overlap),
                "nonempty_overlap": sum(1 for key in overlap if cache.get(key)),
            }
        )
    return pd.DataFrame(rows)


def _write_strength_plot(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    colors = df["stance_WA"].map({1: "#19747e", -1: "#d8873a"})
    ax.scatter(
        df["quality_target"],
        df["classifier_strength"],
        s=22,
        alpha=0.42,
        color=colors,
        edgecolor="none",
    )
    lims = [0, 1]
    ax.plot(lims, lims, color="#293241", linewidth=1.4, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("human quality target: mean(WA, MACE-P)")
    ax.set_ylabel("local classifier strength")
    ax.set_title("Judgement Strength Reliability on AQR-30k", weight="bold")
    ax.grid(True, color="#d8dee9", linewidth=0.8, alpha=0.75)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(output_dir / f"figure_strength_reliability.{ext}", dpi=220)
    plt.close(fig)


def _write_report(
    output_dir: Path,
    args: argparse.Namespace,
    df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    cache_df: pd.DataFrame,
    topic_metrics: pd.DataFrame,
    stance_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Judgement Layer Evaluation: Argument-Quality-Ranking-30k",
        "",
        "This run evaluates the local argument-quality scorer against the",
        "human/crowd-labelled Argument-Quality-Ranking-30k dataset.",
        "",
        "## Inputs",
        "",
        f"- Data directory: `{args.data_dir}`",
        f"- Split: `{args.split}`",
        f"- Sample size: `{len(df)}`",
        f"- Seed: `{args.seed}`",
        f"- Classifier model: `{args.model_dir}`",
        "",
        "## Strength Reliability",
        "",
        "| Target | N | Pearson r | Spearman rho | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target, vals in metrics.items():
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
    stance_counts = df["stance_WA"].value_counts().sort_index()
    lines.extend(
        [
            "",
            "Primary target for paper reporting is `quality_target = mean(WA, MACE-P)`.",
            "The local classifier was originally trained on this dataset family, so",
            "these numbers validate in-domain scoring rather than out-of-domain",
            "generalization.",
            "",
            "## Sample Balance",
            "",
            "| Stance | N |",
            "|---:|---:|",
        ]
    )
    for stance, count in stance_counts.items():
        lines.append(f"| {stance} | {count} |")
    lines.extend(
        [
            "",
            f"- Unique topics in sample: `{df['topic'].nunique()}`",
            f"- Full split rows available: `{args.full_split_n}`",
            "",
            "## Topic And Stance Stability",
            "",
        ]
    )
    if not topic_metrics.empty:
        lines.extend(
            [
                f"Topic metrics are computed for topics with at least `{args.min_group_n}` sampled arguments.",
                "",
                "| Summary over topics | Pearson r | Spearman rho | MAE |",
                "|---|---:|---:|---:|",
                "| Mean | {:.4f} | {:.4f} | {:.4f} |".format(
                    topic_metrics["pearson_r"].mean(),
                    topic_metrics["spearman_rho"].mean(),
                    topic_metrics["mae"].mean(),
                ),
                "| Min | {:.4f} | {:.4f} | {:.4f} |".format(
                    topic_metrics["pearson_r"].min(),
                    topic_metrics["spearman_rho"].min(),
                    topic_metrics["mae"].max(),
                ),
                "| Max | {:.4f} | {:.4f} | {:.4f} |".format(
                    topic_metrics["pearson_r"].max(),
                    topic_metrics["spearman_rho"].max(),
                    topic_metrics["mae"].min(),
                ),
                "",
            ]
        )
    if not stance_metrics.empty:
        lines.extend(
            [
                "| Stance | N | Pearson r | Spearman rho | MAE |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in stance_metrics.to_dict("records"):
            lines.append(
                "| {stance} | {n} | {pearson:.4f} | {spearman:.4f} | {mae:.4f} |".format(
                    stance=row["stance_WA"],
                    n=int(row["n"]),
                    pearson=row["pearson_r"],
                    spearman=row["spearman_rho"],
                    mae=row["mae"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Existing Extraction Cache Coverage",
            "",
            "Existing DEBATE extraction caches do not currently cover AQR-30k rows.",
            "A future extractor pass should use `sample_for_extractor.csv` as the",
            "frozen benchmark input.",
            "",
            "| Cache | Entries | Overlap | Non-empty overlap |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in cache_df.to_dict("records"):
        lines.append(
            f"| `{row['cache_file']}` | {row['entries']} | {row['overlap']} | {row['nonempty_overlap']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `predictions.csv`: sample rows plus classifier predictions.",
            "- `sample_for_extractor.csv`: frozen sample for future LLM extraction/polarity evaluation.",
            "- `metrics.json`: machine-readable summary metrics.",
            "- `topic_metrics.csv`: strength reliability by topic.",
            "- `stance_metrics.csv`: strength reliability by pro/con stance.",
            "- `cache_coverage.csv`: extraction-cache overlap diagnostics.",
            "- `figure_strength_reliability.pdf`: classifier strength versus human quality target.",
            "",
            "## Interpretation",
            "",
            "This validates the strength-scoring component against existing human",
            "quality labels. It does not yet validate multi-claim extraction from",
            "messy dialogue turns or binary-polarity failures on moderation cases.",
            "Those should be handled by a smaller DEBATE-specific stress test.",
        ]
    )
    (output_dir / "judgement_layer_aqr_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", choices=["train", "dev", "test", "merged"], default="test")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-group-n", type=int, default=20)
    parser.add_argument(
        "--cache-glob",
        default="validation/data/cache/extractions/extraction_cache*.json",
    )
    args = parser.parse_args()

    if args.split == "merged":
        full_df = pd.read_csv(Path(args.data_dir) / "merged.csv")
    else:
        full_df = _load_split(args.data_dir, args.split)
    args.full_split_n = len(full_df)
    df = _balanced_sample(full_df, args.sample_size, args.seed)
    df = df.copy()
    df["quality_target"] = (df["WA"] + df["MACE-P"]) / 2.0
    df["support_strength_target"] = df["quality_target"] * df["stance_WA_conf"]
    df["human_polarity"] = df["stance_WA"].map({1: "affirmative", -1: "negative", 0: "neutral"})
    df["text_hash"] = df["argument"].map(_hash_text)
    df["classifier_strength"] = _predict_strengths(
        df["argument"].tolist(), args.model_dir, args.batch_size
    )

    metrics = {
        "WA": _metrics(df["WA"], df["classifier_strength"]),
        "MACE-P": _metrics(df["MACE-P"], df["classifier_strength"]),
        "quality_target": _metrics(df["quality_target"], df["classifier_strength"]),
        "support_strength_target": _metrics(
            df["support_strength_target"], df["classifier_strength"]
        ),
    }
    cache_df = _cache_coverage(df, args.cache_glob)
    topic_metrics = _group_metrics(df, "topic", args.min_group_n)
    stance_metrics = _group_metrics(df, "stance_WA", 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"aqr_{args.split}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "predictions.csv", index=False)
    topic_metrics.to_csv(output_dir / "topic_metrics.csv", index=False)
    stance_metrics.to_csv(output_dir / "stance_metrics.csv", index=False)
    df[
        [
            "text_hash",
            "topic",
            "argument",
            "human_polarity",
            "stance_WA",
            "stance_WA_conf",
            "WA",
            "MACE-P",
            "quality_target",
        ]
    ].to_csv(output_dir / "sample_for_extractor.csv", index=False)
    cache_df.to_csv(output_dir / "cache_coverage.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_strength_plot(df, output_dir)
    _write_report(output_dir, args, df, metrics, cache_df, topic_metrics, stance_metrics)
    print(f"Saved AQR judgement-layer evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
