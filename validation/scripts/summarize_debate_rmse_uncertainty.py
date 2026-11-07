#!/usr/bin/env python3
"""Summarize fold-level uncertainty for DEBATE held-out RMSE.

The canonical parameter recommendation runs save one out-of-fold prediction per
participant/model in ``individual_results.csv``.  This script collapses those
rows to fold-level RMSE, adds t confidence intervals across outer folds, and
runs paired fold-level tests against selected reference models.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_GROUP_DIR = Path(
    "validation/results/parameter_recommendations_group_response5_evidence13x13_quality"
)
DEFAULT_TOPIC_DIR = Path(
    "validation/results/parameter_recommendations_topic_response5_evidence13x13_quality"
)
DEFAULT_OUTPUT_DIR = Path(
    "validation/paper_artifacts/evidence_alignment"
)
DEFAULT_REFERENCES = ["prior_only_no_change", "preset_global"]

MODEL_LABELS = {
    "prior_only_no_change": "Prior only",
    "preset_global": "Global parameters",
    "preset_participant_type": "Participant-type parameters",
    "preset_topic_category": "Topic-category parameters",
    "preset_evidence_alignment": "Evidence-alignment parameters",
    "preset_response_type": "Response-type diagnostic",
}

MODEL_ORDER = [
    "prior_only_no_change",
    "preset_global",
    "preset_participant_type",
    "preset_topic_category",
    "preset_evidence_alignment",
    "preset_response_type",
]

REQUIRED_COLUMNS = {
    "fold",
    "model",
    "opinion_change_actual",
    "opinion_change_predicted",
}


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _model_rank(model: str) -> int:
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
    err = predicted.astype(float) - actual.astype(float)
    return float(np.sqrt(np.mean(np.square(err))))


def _mse(actual: pd.Series, predicted: pd.Series) -> float:
    err = predicted.astype(float) - actual.astype(float)
    return float(np.mean(np.square(err)))


def _mae(actual: pd.Series, predicted: pd.Series) -> float:
    err = predicted.astype(float) - actual.astype(float)
    return float(np.mean(np.abs(err)))


def _load_individual_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def fold_metrics(group_dir: Path, topic_dir: Path) -> pd.DataFrame:
    rows = []
    sources = [
        ("group-held-out", group_dir / "individual_results.csv"),
        ("topic-held-out", topic_dir / "individual_results.csv"),
    ]
    for split, path in sources:
        df = _load_individual_results(path)
        for (fold, model), group in df.groupby(["fold", "model"], dropna=False):
            actual = group["opinion_change_actual"]
            predicted = group["opinion_change_predicted"]
            rows.append(
                {
                    "split": split,
                    "fold": int(fold),
                    "model": str(model),
                    "model_label": _model_label(str(model)),
                    "n": int(len(group)),
                    "rmse": _rmse(actual, predicted),
                    "mse": _mse(actual, predicted),
                    "mae": _mae(actual, predicted),
                }
            )
    out = pd.DataFrame(rows)
    out["model_order"] = out["model"].map(_model_rank)
    out = out.sort_values(["split", "model_order", "fold"]).drop(columns=["model_order"])
    return out.reset_index(drop=True)


def confidence_intervals(
    fold_df: pd.DataFrame,
    ci_level: float,
) -> pd.DataFrame:
    rows = []
    alpha = 1.0 - ci_level
    for (split, model), group in fold_df.groupby(["split", "model"], dropna=False):
        ordered = group.sort_values("fold")
        values = ordered["rmse"].to_numpy(dtype=float)
        n_folds = int(len(values))
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if n_folds > 1 else math.nan
        se = float(sd / math.sqrt(n_folds)) if n_folds > 1 else math.nan
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, n_folds - 1)) if n_folds > 1 else math.nan
        half_width = t_crit * se if n_folds > 1 else math.nan
        weighted_mse = float(np.average(ordered["mse"], weights=ordered["n"]))
        rows.append(
            {
                "split": split,
                "model": model,
                "model_label": _model_label(model),
                "n_folds": n_folds,
                "n_observations": int(ordered["n"].sum()),
                "aggregate_rmse": math.sqrt(weighted_mse),
                "fold_mean_rmse": mean,
                "fold_sd_rmse": sd,
                "fold_se_rmse": se,
                "ci_level": ci_level,
                "fold_ci_low": mean - half_width if n_folds > 1 else math.nan,
                "fold_ci_high": mean + half_width if n_folds > 1 else math.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["model_order"] = out["model"].map(_model_rank)
    out = out.sort_values(["split", "model_order"]).drop(columns=["model_order"])
    return out.reset_index(drop=True)


def paired_fold_tests(
    fold_df: pd.DataFrame,
    reference_models: Iterable[str],
    ci_level: float,
) -> pd.DataFrame:
    rows = []
    alpha = 1.0 - ci_level
    references = list(dict.fromkeys(reference_models))
    for split, split_df in fold_df.groupby("split", dropna=False):
        pivot = split_df.pivot_table(index="fold", columns="model", values="rmse", aggfunc="first")
        for reference in references:
            if reference not in pivot.columns:
                continue
            for model in pivot.columns:
                if model == reference:
                    continue
                paired = pivot[[model, reference]].dropna()
                if paired.empty:
                    continue
                model_values = paired[model].to_numpy(dtype=float)
                reference_values = paired[reference].to_numpy(dtype=float)
                diff = model_values - reference_values
                n_folds = int(len(diff))
                mean_delta = float(np.mean(diff))
                sd_delta = float(np.std(diff, ddof=1)) if n_folds > 1 else math.nan
                se_delta = float(sd_delta / math.sqrt(n_folds)) if n_folds > 1 else math.nan
                t_crit = (
                    float(stats.t.ppf(1.0 - alpha / 2.0, n_folds - 1))
                    if n_folds > 1
                    else math.nan
                )
                half_width = t_crit * se_delta if n_folds > 1 else math.nan
                if n_folds > 1 and not np.allclose(diff, 0.0):
                    t_result = stats.ttest_rel(model_values, reference_values)
                    paired_t_statistic = float(t_result.statistic)
                    paired_t_p_value = float(t_result.pvalue)
                else:
                    paired_t_statistic = math.nan
                    paired_t_p_value = math.nan
                try:
                    if not np.allclose(diff, 0.0):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            wilcoxon_result = stats.wilcoxon(
                                model_values,
                                reference_values,
                                zero_method="wilcox",
                                alternative="two-sided",
                                # Preserve the original SciPy 1.10 auto policy.
                                # Newer versions use a different default for
                                # zero differences in these small fold samples.
                                method=("approx" if np.any(diff == 0.0)
                                        or n_folds > 50 else "exact"),
                            )
                        wilcoxon_statistic = float(wilcoxon_result.statistic)
                        wilcoxon_p_value = float(wilcoxon_result.pvalue)
                    else:
                        wilcoxon_statistic = math.nan
                        wilcoxon_p_value = math.nan
                except ValueError:
                    wilcoxon_statistic = math.nan
                    wilcoxon_p_value = math.nan
                rows.append(
                    {
                        "split": split,
                        "model": model,
                        "model_label": _model_label(str(model)),
                        "reference_model": reference,
                        "reference_label": _model_label(reference),
                        "n_folds": n_folds,
                        "model_fold_mean_rmse": float(np.mean(model_values)),
                        "reference_fold_mean_rmse": float(np.mean(reference_values)),
                        "mean_rmse_delta": mean_delta,
                        "delta_ci_level": ci_level,
                        "delta_ci_low": mean_delta - half_width if n_folds > 1 else math.nan,
                        "delta_ci_high": mean_delta + half_width if n_folds > 1 else math.nan,
                        "paired_t_statistic": paired_t_statistic,
                        "paired_t_p_value": paired_t_p_value,
                        "wilcoxon_statistic": wilcoxon_statistic,
                        "wilcoxon_p_value": wilcoxon_p_value,
                        "folds_model_lower_rmse": int(np.sum(diff < 0)),
                        "folds_reference_lower_rmse": int(np.sum(diff > 0)),
                        "folds_tied": int(np.sum(np.isclose(diff, 0.0))),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["model_order"] = out["model"].map(_model_rank)
    out["reference_order"] = out["reference_model"].map(_model_rank)
    out = out.sort_values(["split", "reference_order", "model_order"]).drop(
        columns=["model_order", "reference_order"]
    )
    return out.reset_index(drop=True)


def _format_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _key in columns) + " |"
    divider = "| " + " | ".join("---" for _label, _key in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for _label, key in columns) + " |")
    return "\n".join([header, divider, *body])


def _ci_text(row: pd.Series) -> str:
    return f"[{_format_float(row['fold_ci_low'], 4)}, {_format_float(row['fold_ci_high'], 4)}]"


def _delta_ci_text(row: pd.Series) -> str:
    return f"[{_format_float(row['delta_ci_low'], 4)}, {_format_float(row['delta_ci_high'], 4)}]"


def _primary_reading_lines(
    ci_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    ci_level: float,
    primary_model: str,
) -> list[str]:
    """Create a compact interpretation from the computed CSV rows."""
    if ci_df.empty:
        return ["_No confidence-interval rows were available._"]

    ci_lookup = {
        (str(row["split"]), str(row["model"])): row
        for _, row in ci_df.iterrows()
    }
    primary_tests = tests_df[tests_df["model"] == primary_model].copy()
    primary_label = _model_label(primary_model)
    percent = int(ci_level * 100)
    lines = [
        "Generated from the confidence-interval and paired-test tables above:",
        "",
    ]

    for split in sorted(ci_df["split"].unique()):
        primary_row = ci_lookup.get((str(split), primary_model))
        if primary_row is None:
            continue
        lines.append(
            f"- `{split}`: `{primary_label}` has aggregate RMSE "
            f"`{_format_float(primary_row['aggregate_rmse'], 4)}` and fold-mean RMSE "
            f"`{_format_float(primary_row['fold_mean_rmse'], 4)}` "
            f"({percent}% fold CI `{_ci_text(primary_row)}`)."
        )
        split_tests = primary_tests[primary_tests["split"] == split].copy()
        for _, test_row in split_tests.iterrows():
            lines.append(
                f"- `{split}` vs `{test_row['reference_label']}`: mean fold RMSE delta "
                f"`{_format_float(test_row['mean_rmse_delta'], 4)}` "
                f"({percent}% delta CI `{_delta_ci_text(test_row)}`), with lower RMSE in "
                f"`{int(test_row['folds_model_lower_rmse'])}/{int(test_row['n_folds'])}` folds."
            )

        split_ci = ci_df[ci_df["split"] == split].copy()
        best = split_ci.loc[split_ci["aggregate_rmse"].idxmin()]
        if str(best["model"]) != primary_model:
            caveat = ""
            if str(best["model"]) == "preset_response_type":
                caveat = (
                    " Treat this as diagnostic because response-type presets are based on "
                    "observed outcome classes rather than pre-debate deployable groups."
                )
            lines.append(
                f"- `{split}`: the lowest aggregate RMSE row is `{best['model_label']}` "
                f"at `{_format_float(best['aggregate_rmse'], 4)}`.{caveat}"
            )

    lines.extend(
        [
            "",
            "The paired t-test columns quantify the fold-level mean deltas. The Wilcoxon",
            "column is intentionally retained as a small-sample sign/rank check; with",
            "five folds, all folds moving in the same direction can still yield a",
            "two-sided Wilcoxon p-value of `0.0625`.",
        ]
    )
    return lines


def write_report(
    output_dir: Path,
    fold_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    ci_level: float,
    primary_model: str,
) -> Path:
    output_path = output_dir / "debate_rmse_uncertainty_report.md"
    ci_rows = []
    for _, row in ci_df.iterrows():
        ci_rows.append(
            {
                "Split": row["split"],
                "Model": row["model_label"],
                "Aggregate RMSE": _format_float(row["aggregate_rmse"], 4),
                "Fold mean RMSE": _format_float(row["fold_mean_rmse"], 4),
                f"{int(ci_level * 100)}% fold CI": (
                    f"[{_format_float(row['fold_ci_low'], 4)}, "
                    f"{_format_float(row['fold_ci_high'], 4)}]"
                ),
                "Folds": int(row["n_folds"]),
            }
        )
    primary_tests = tests_df[tests_df["model"] == primary_model].copy()
    test_rows = []
    for _, row in primary_tests.iterrows():
        test_rows.append(
            {
                "Split": row["split"],
                "Comparison": f"{row['model_label']} vs {row['reference_label']}",
                "Mean RMSE delta": _format_float(row["mean_rmse_delta"], 4),
                f"{int(ci_level * 100)}% delta CI": (
                    f"[{_format_float(row['delta_ci_low'], 4)}, "
                    f"{_format_float(row['delta_ci_high'], 4)}]"
                ),
                "paired t p": _format_p(row["paired_t_p_value"]),
                "Wilcoxon p": _format_p(row["wilcoxon_p_value"]),
                "Lower-RMSE folds": (
                    f"{int(row['folds_model_lower_rmse'])}/"
                    f"{int(row['n_folds'])}"
                ),
            }
        )
    lines = [
        "# DEBATE RMSE Fold-Level Uncertainty",
        "",
        "Computed from the canonical quality-filtered DEBATE parameter recommendation",
        "`individual_results.csv` files. Each model has one held-out RMSE per",
        "outer fold for the group-held-out split and one per outer fold for the",
        "topic-held-out split.",
        "",
        "Confidence intervals are t intervals over the five outer folds. Paired",
        "tests compare fold RMSE values within the same split and fold. With only",
        "five folds, these tests should be read as fold-level robustness checks,",
        "not as a substitute for a participant-level bootstrap or a new rerun.",
        "",
        "Negative RMSE deltas mean the first model has lower RMSE than the",
        "reference model.",
        "",
        "## Data-Driven Reading",
        "",
        *_primary_reading_lines(ci_df, tests_df, ci_level, primary_model),
        "",
        "## Fold-Level RMSE Confidence Intervals",
        "",
        _markdown_table(
            ci_rows,
            [
                ("Split", "Split"),
                ("Model", "Model"),
                ("Aggregate RMSE", "Aggregate RMSE"),
                ("Fold mean RMSE", "Fold mean RMSE"),
                (f"{int(ci_level * 100)}% fold CI", f"{int(ci_level * 100)}% fold CI"),
                ("Folds", "Folds"),
            ],
        ),
        "",
        f"## Primary Fold-Level Tests: {_model_label(primary_model)}",
        "",
        _markdown_table(
            test_rows,
            [
                ("Split", "Split"),
                ("Comparison", "Comparison"),
                ("Mean RMSE delta", "Mean RMSE delta"),
                (f"{int(ci_level * 100)}% delta CI", f"{int(ci_level * 100)}% delta CI"),
                ("paired t p", "paired t p"),
                ("Wilcoxon p", "Wilcoxon p"),
                ("Lower-RMSE folds", "Lower-RMSE folds"),
            ],
        ),
        "",
        "## Files",
        "",
        "- `debate_rmse_fold_metrics.csv`",
        "- `debate_rmse_confidence_intervals.csv`",
        "- `debate_rmse_fold_tests.csv`",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def build_outputs(
    group_dir: Path | str = DEFAULT_GROUP_DIR,
    topic_dir: Path | str = DEFAULT_TOPIC_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    ci_level: float = 0.95,
    reference_models: Iterable[str] = DEFAULT_REFERENCES,
    primary_model: str = "preset_evidence_alignment",
) -> dict[str, Path]:
    group_dir = Path(group_dir)
    topic_dir = Path(topic_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = fold_metrics(group_dir, topic_dir)
    intervals = confidence_intervals(folds, ci_level)
    tests = paired_fold_tests(folds, reference_models, ci_level)

    fold_path = output_dir / "debate_rmse_fold_metrics.csv"
    ci_path = output_dir / "debate_rmse_confidence_intervals.csv"
    tests_path = output_dir / "debate_rmse_fold_tests.csv"
    folds.to_csv(fold_path, index=False)
    intervals.to_csv(ci_path, index=False)
    tests.to_csv(tests_path, index=False)
    report_path = write_report(output_dir, folds, intervals, tests, ci_level, primary_model)

    return {
        "fold_metrics": fold_path,
        "confidence_intervals": ci_path,
        "fold_tests": tests_path,
        "report": report_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute fold-level confidence intervals and paired tests for DEBATE RMSE."
    )
    parser.add_argument("--group-dir", type=Path, default=DEFAULT_GROUP_DIR)
    parser.add_argument("--topic-dir", type=Path, default=DEFAULT_TOPIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--reference-model",
        action="append",
        dest="reference_models",
        default=None,
        help="Reference model for paired tests. Can be supplied multiple times.",
    )
    parser.add_argument("--primary-model", default="preset_evidence_alignment")
    args = parser.parse_args()

    outputs = build_outputs(
        group_dir=args.group_dir,
        topic_dir=args.topic_dir,
        output_dir=args.output_dir,
        ci_level=args.ci_level,
        reference_models=args.reference_models or DEFAULT_REFERENCES,
        primary_model=args.primary_model,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
