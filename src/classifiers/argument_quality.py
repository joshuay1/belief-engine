# SPDX-License-Identifier: MIT
"""WA regression scorer, also shipped standalone as argument_quality_model.py."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ArgumentQualityModel:
    """Predict WA quality scores; clip raw regression outputs without a sigmoid."""

    def __init__(self, model_dir: str, device: str = None, *, revision: str = None,
                 local_files_only: bool = False):
        self.model_dir = str(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = 128
        weights = Path(model_dir) / "model.safetensors"
        if weights.is_file():
            with weights.open("rb") as stream:
                if stream.read(50).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError("Weights are a Git LFS pointer. Download the release or run git lfs pull.")
        options = {"revision": revision, "local_files_only": local_files_only}
        # DeBERTa does not use the Mistral regex. Passing fix_mistral_regex raises
        # TypeError in Transformers 5.1.0; the original tokenizer needs no patch.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, **options)
        self.model, loading = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, output_loading_info=True, **options
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete or incompatible argument-quality checkpoint: {loading}")
        if self.model.config.num_labels != 1 or self.model.config.problem_type != "regression":
            raise ValueError("Expected a single-output regression checkpoint.")
        self.model.to(self.device).eval()

    def predict(self, argument: str, max_length: int = None) -> float:
        return self.predict_batch([argument], max_length=max_length)[0]

    def predict_batch(self, arguments: list[str], max_length: int = None) -> list[float]:
        """Return one score per argument, including a list for a one-item batch."""
        if not arguments:
            return []
        inputs = self.tokenizer(
            arguments, truncation=True, padding="max_length",
            max_length=self.max_length if max_length is None else max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            scores = self.model(**inputs).logits.squeeze(-1).clamp(0.0, 1.0)
        return scores.cpu().tolist()
