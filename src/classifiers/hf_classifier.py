import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class HuggingFaceClassifier:
    """
    Wrapper for HuggingFace Sequence Classification models (e.g., Reward Models).
    Output: Normalized score (0.0 to 1.0).
    Supports optional z-score calibration for cross-model comparability.
    """
    def __init__(self, model_name_or_path: str, device: str = None, calibration_path: str = "config/calibration.json"):
        self.model_name = model_name_or_path
        print(f"   (Classifier) Loading model: {self.model_name}...")
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"   (Classifier) Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"   (Classifier) Error loading model: {e}")
            raise e
        
        # Load calibration if available
        self.calibration = None
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    all_calibration = json.load(f)
                if self.model_name in all_calibration:
                    self.calibration = all_calibration[self.model_name]
                    print(f"   (Classifier) Loaded calibration: mean={self.calibration['mean']:.4f}, std={self.calibration['std']:.4f}")
            except Exception as e:
                print(f"   (Classifier) Warning: Could not load calibration: {e}")

    def predict(self, text: str) -> float:
        """
        Predicts the 'quality' or 'reward' score of the text.
        Applies z-score normalization if calibration is available, then clips to [0, 1].
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Handling different output shapes
            if logits.shape[-1] == 1:
                # Regression/Reward Model (Single scalar)
                raw_score = logits.item()
                # Apply sigmoid for initial normalization
                normalized_score = 1.0 / (1.0 + np.exp(-raw_score))
            else:
                # Classification (Binary/Multi-class) - Take positive class prob
                probs = torch.softmax(logits, dim=-1)
                normalized_score = probs[0][1].item()
        
        # Apply z-score calibration if available
        if self.calibration:
            mean = self.calibration["mean"]
            std = self.calibration["std"]
            if std > 0:
                # Z-score: how many stds from mean
                z = (normalized_score - mean) / std
                # Map z-score to [0, 1] using sigmoid (z=0 -> 0.5, z=+2 -> ~0.88, z=-2 -> ~0.12)
                calibrated_score = 1.0 / (1.0 + np.exp(-z))
                return float(np.clip(calibrated_score, 0.0, 1.0))
        
        return float(np.clip(normalized_score, 0.0, 1.0))

