"""Rebuild the released fold statistics from their saved prediction inputs."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from validation.scripts.summarize_debate_rmse_uncertainty import build_outputs

ROOT = Path(__file__).resolve().parents[1]


class ReplayRebuildTests(unittest.TestCase):
    def test_saved_predictions_reproduce_all_three_released_tables(self):
        with tempfile.TemporaryDirectory() as output:
            paths = build_outputs(
                group_dir=ROOT / "validation/results/parameter_recommendations_group_response5_evidence13x13_quality",
                topic_dir=ROOT / "validation/results/parameter_recommendations_topic_response5_evidence13x13_quality",
                output_dir=Path(output),
            )
            for key in ("fold_metrics", "confidence_intervals", "fold_tests"):
                with self.subTest(table=key):
                    saved = ROOT / "validation/paper_artifacts/evidence_alignment" / paths[key].name
                    pd.testing.assert_frame_equal(
                        pd.read_csv(paths[key]), pd.read_csv(saved),
                        check_exact=False, rtol=1e-12, atol=1e-12,
                    )
