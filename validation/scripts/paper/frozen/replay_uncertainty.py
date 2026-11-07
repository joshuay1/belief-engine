"""Original saved-prediction bootstrap functions; see provenance.json."""
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
REFERENCE_MODEL="classifier_log1p"
FACTORIAL_MODELS=("classifier_log1p","uniform1_log1p","classifier_stance_linear","uniform1_stance_linear")
BOOTSTRAP_REPLICATES=10000
RANDOM_SEED=42
class ContractError(ValueError): pass
def contrast_definitions(models: Iterable[str]) -> dict[str, dict[str, float]]:
    available = set(models)
    if REFERENCE_MODEL not in available:
        raise ContractError("Reference model is absent from contrasts.")
    contrasts = {
        f"{model}_minus_{REFERENCE_MODEL}": {
            model: 1.0,
            REFERENCE_MODEL: -1.0,
        }
        for model in sorted(available - {REFERENCE_MODEL})
    }
    factorial = set(FACTORIAL_MODELS)
    if factorial.issubset(available):
        contrasts["strength_by_updater_interaction"] = {
            "uniform1_stance_linear": 1.0,
            "classifier_stance_linear": -1.0,
            "uniform1_log1p": -1.0,
            "classifier_log1p": 1.0,
        }
    return contrasts

def paired_cluster_bootstrap(
    predictions: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Bootstrap registered metric contrasts within held-out-fold clusters."""
    if replicates < 1:
        raise ContractError("Bootstrap replicates must be positive.")
    output_rows: list[dict[str, Any]] = []
    for split in ("group", "topic"):
        frame = predictions[predictions["split"] == split].copy()
        cluster_column = "group_file" if split == "group" else "topic"
        models = sorted(frame["model"].unique())
        definitions = contrast_definitions(models)
        model_index = {model: index for index, model in enumerate(models)}

        # Check exact row-level pairing before aggregation.
        verify_exact_prediction_pairing(frame)

        fold_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _fold, fold_frame in frame.groupby("fold", sort=True):
            cluster_rows: list[tuple[str, np.ndarray, np.ndarray, int]] = []
            for cluster, cluster_frame in fold_frame.groupby(cluster_column, sort=True):
                sq = np.zeros(len(models), dtype=float)
                absolute = np.zeros(len(models), dtype=float)
                counts: list[int] = []
                for model, model_frame in cluster_frame.groupby("model", sort=True):
                    predicted = model_frame["opinion_change_predicted"].to_numpy(float)
                    actual = model_frame["opinion_change_actual"].to_numpy(float)
                    errors = predicted - actual
                    idx = model_index[str(model)]
                    sq[idx] = float(np.square(errors).sum())
                    absolute[idx] = float(np.abs(errors).sum())
                    counts.append(len(errors))
                if len(counts) != len(models) or len(set(counts)) != 1:
                    raise ContractError(f"{split}/{cluster} model rows are unpaired.")
                cluster_rows.append((str(cluster), sq, absolute, counts[0]))
            if not cluster_rows:
                raise ContractError(f"{split} contains an empty held-out fold.")
            fold_arrays.append((
                np.stack([row[1] for row in cluster_rows]),
                np.stack([row[2] for row in cluster_rows]),
                np.asarray([row[3] for row in cluster_rows], dtype=float),
            ))

        rng = np.random.default_rng(seed)
        contrast_draws = {
            (contrast, metric): np.empty(replicates, dtype=float)
            for contrast in definitions
            for metric in ("rmse", "mae")
        }
        chunk_size = min(500, replicates)
        for start in range(0, replicates, chunk_size):
            stop = min(start + chunk_size, replicates)
            width = stop - start
            total_sq = np.zeros((width, len(models)), dtype=float)
            total_abs = np.zeros((width, len(models)), dtype=float)
            total_n = np.zeros(width, dtype=float)
            for sq, absolute, counts in fold_arrays:
                n_clusters = len(counts)
                choices = rng.integers(0, n_clusters, size=(width, n_clusters))
                total_sq += sq[choices].sum(axis=1)
                total_abs += absolute[choices].sum(axis=1)
                total_n += counts[choices].sum(axis=1)
            rmse = np.sqrt(total_sq / total_n[:, None])
            mae = total_abs / total_n[:, None]
            for contrast, coefficients in definitions.items():
                coefficient_vector = np.zeros(len(models), dtype=float)
                for model, coefficient in coefficients.items():
                    coefficient_vector[model_index[model]] = coefficient
                contrast_draws[(contrast, "rmse")][start:stop] = rmse @ coefficient_vector
                contrast_draws[(contrast, "mae")][start:stop] = mae @ coefficient_vector

        pooled, _fold = summarize_metrics(frame)
        for metric in ("rmse", "mae"):
            values = pooled.set_index("model")[metric].astype(float).to_dict()
            for contrast, coefficients in definitions.items():
                draws = contrast_draws[(contrast, metric)]
                observed = sum(c * values[m] for m, c in coefficients.items())
                lower, upper = np.quantile(draws, [0.025, 0.975])
                p_value = min(
                    1.0, 2.0 * min(float(np.mean(draws <= 0.0)),
                                   float(np.mean(draws >= 0.0))),
                )
                output_rows.append({
                    "split": split, "cluster": cluster_column,
                    "contrast": contrast, "metric": metric,
                    "observed_delta": observed, "bootstrap_mean": float(draws.mean()),
                    "ci_lower": float(lower), "ci_upper": float(upper),
                    "p_two_sided": p_value, "replicates": replicates, "seed": seed,
                })
    return pd.DataFrame(output_rows)

# Compact adapters for released prediction columns (the original runner used
# in-memory records and the names group_file/topic before export).
def verify_exact_prediction_pairing(frame):
    keys=['split','participant_record_id']
    assert not frame.duplicated(keys+['model']).any()
    assert frame.groupby(keys).model.nunique().eq(frame.model.nunique()).all()
    assert frame.groupby(keys).opinion_change_actual.nunique().eq(1).all()
    assert frame.groupby(keys).fold.nunique().eq(1).all()

def summarize_metrics(frame):
    rows=[]
    for (split,model),d in frame.groupby(['split','model'],sort=True):
        error=d.opinion_change_predicted-d.opinion_change_actual
        rows.append(dict(split=split,model=model,n=len(d),rmse=float(np.sqrt((error**2).mean())),mae=float(abs(error).mean())))
    return pd.DataFrame(rows),None
