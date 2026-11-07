"""Offline regression tests for release loading, score semantics and fail-fast behavior."""

import importlib.util
import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.classifiers.argument_quality import ArgumentQualityModel
from src.agents.extractor_agent import ArgumentExtractionAgent


class ReleaseTests(unittest.TestCase):
    def test_regression_is_clipped_without_sigmoid_and_singleton_is_list(self):
        scorer = ArgumentQualityModel.__new__(ArgumentQualityModel)
        scorer.max_length = 128
        scorer.device = torch.device("cpu")
        scorer.tokenizer = Mock(return_value={"input_ids": torch.tensor([[1]])})
        scorer.model = Mock(return_value=Mock(logits=torch.tensor([[.2]])))
        self.assertAlmostEqual(scorer.predict_batch(["claim"])[0], .2)
        self.assertAlmostEqual(scorer.predict("claim"), .2)
        self.assertEqual(scorer.tokenizer.call_args.kwargs["max_length"], 128)
        self.assertEqual(scorer.tokenizer.call_args.kwargs["padding"], "max_length")
        scorer.model.return_value.logits = torch.tensor([[-.1], [1.2]])
        self.assertEqual(scorer.predict_batch(["a", "b"]), [0., 1.])
        scorer.tokenizer.reset_mock()
        self.assertEqual(scorer.predict_batch([]), [])
        scorer.tokenizer.assert_not_called()

    def test_lfs_pointer_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                ArgumentQualityModel(folder)

    def test_loading_failure_does_not_switch_requested_scorer(self):
        with patch.object(ArgumentExtractionAgent, "_load_classifier", side_effect=OSError("missing weights")):
            with self.assertRaisesRegex(RuntimeError, "refusing to change"):
                ArgumentExtractionAgent(None, "topic", {"memory": {"strength_estimation_method": "classifier"}})

    def test_explicit_fallback_and_llm_only_mode(self):
        with patch.object(ArgumentExtractionAgent, "_load_classifier", side_effect=OSError("missing weights")):
            agent = ArgumentExtractionAgent(None, "topic", {"memory": {
                "strength_estimation_method": "classifier", "classifier_allow_llm_fallback": True}})
            self.assertEqual(agent.strength_estimation_method, "llm")
            agent = ArgumentExtractionAgent(None, "topic", {"memory": {
                "strength_estimation_method": "llm", "classifier_model_path": "disabled"}})
            self.assertIsNone(agent.classifier_model)

    def test_disabled_classifier_is_an_error_when_required(self):
        with self.assertRaises(RuntimeError):
            ArgumentExtractionAgent(None, "topic", {"memory": {
                "strength_estimation_method": "classifier", "classifier_model_path": "disabled"}})

    def test_hub_id_and_revision_are_not_rewritten_as_local_path(self):
        with patch("src.classifiers.argument_quality.ArgumentQualityModel") as constructor:
            ArgumentExtractionAgent._load_classifier({"classifier_model_path": "release-owner/quality-model",
                "classifier_kind": "argument_quality", "classifier_revision": "a" * 40})
            constructor.assert_called_once_with("release-owner/quality-model", revision="a" * 40, local_files_only=False)

    def test_missing_local_model_is_not_sent_to_hub(self):
        with self.assertRaises(FileNotFoundError):
            ArgumentExtractionAgent._load_classifier({"classifier_model_path": "models/does-not-exist"})

    def test_classifier_scores_reach_records_in_both_extraction_modes(self):
        claim = "Public transit reduces congestion."
        for mode in ("legacy_text_v1", "structured_json_v1"):
            with self.subTest(mode=mode):
                backend = Mock()
                backend.chat.side_effect = [
                    json.dumps([{"claim": claim, "evidence_snippets": [claim]}]),
                    json.dumps([{"polarity": "affirmative", "support_strength": .9,
                                 "category_tags": []}]),
                ]
                backend.get_completion.side_effect = [
                    json.dumps({"arguments": [{"claim": claim, "evidence_snippets": [claim]}]}),
                    json.dumps({"classifications": [{"claim_index": 0, "claim": claim,
                        "polarity": "affirmative", "support_strength": .9, "category_tags": []}]}),
                ]
                scorer = Mock()
                scorer.predict.return_value = .2
                with patch.object(ArgumentExtractionAgent, "_load_classifier", return_value=scorer):
                    agent = ArgumentExtractionAgent(backend, "We should fund public transit", {
                        "memory": {"strength_estimation_method": "classifier"},
                        "extraction": {"output_mode": mode},
                    })
                records = agent.extract(claim, "opponent")
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0].support_strength, .2)
                self.assertEqual(records[0].support_strength_llm, .9)
                scorer.predict.assert_called_once_with(claim)



if __name__ == "__main__":
    unittest.main()
