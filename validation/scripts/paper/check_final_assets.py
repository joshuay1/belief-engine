"""Check final figure files and the recovered profile/sweep summaries offline."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def main():
    index = json.loads((ROOT / "validation/camera_ready/figures/index.json").read_text())
    for item in index["figures"]:
        assert hashlib.sha256((ROOT / item["release_file"]).read_bytes()).hexdigest() == item["sha256"], item["paper_file"]

    profiles = ROOT / "validation/camera_ready/profiles"
    summary = pd.read_csv(profiles / "summary.csv")
    paths = pd.read_csv(profiles / "stance_trajectories.csv")
    assert len(summary) == 12 and len(paths) == 384
    assert summary.groupby("pairing").size().eq(3).all()
    assert paths.groupby(["run_id", "side"]).size().eq(16).all()
    for row in summary.itertuples():
        for side in ("pro", "con"):
            p = paths[(paths.run_id == row.run_id) & (paths.side == side)].sort_values("round")
            np.testing.assert_allclose(p.stance.iloc[[0, -1]], [getattr(row, side + "_initial"), getattr(row, side + "_final")], atol=1e-12)
    endpoints = summary.groupby("pairing")[["pro_final", "con_final"]].mean()
    expected = {"Open/Open": [.26, .30], "Committed/Open": [.81, .74], "Committed/Committed": [.38, -.37]}
    for pairing, values in expected.items():
        np.testing.assert_allclose(endpoints.loc[pairing], values, atol=.005)

    sweeps = ROOT / "validation/results/model_sensitivity_comparison"
    runs = pd.read_csv(sweeps / "model_sensitivity_runs.csv")
    rounds = pd.read_csv(sweeps / "model_sensitivity_rounds.csv")
    runs = runs[runs.agent == "CoreBot"]
    rounds = rounds[rounds.agent == "CoreBot"]
    assert len(runs) == 150 and len(rounds) == 2400
    assert runs.groupby(["model", "sweep", "parameter_value"]).size().eq(5).all()
    for row in runs.itertuples():
        p = rounds[rounds.run_folder == row.run_folder].sort_values("round")
        np.testing.assert_allclose(p.subject_stance.iloc[[0, -1]], [row.initial_stance, row.final_stance], atol=1e-12)
    print(f"Verified {len(index['figures'])} final assets, 12 profile runs and 150 BE sweep runs.")
    print("Profile final stance means:\n", endpoints.round(4).to_string())


if __name__ == "__main__":
    main()
