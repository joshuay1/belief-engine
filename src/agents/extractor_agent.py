import json
import os
from typing import List, Optional, Dict
from ..backend.backend_base import LLMBackend
from ..memory.argument_record import ArgumentRecord

class ArgumentExtractionAgent:
    """
    Extracts structured arguments from a debate turn using the LLM.
    Can use either LLM or a classifier model to estimate support_strength.
    """
    def __init__(self, backend: LLMBackend, topic_id: str, config: Optional[Dict] = None):
        self.backend = backend
        self.topic_id = topic_id
        self.config = config or {}

        memory_config = self.config.get("memory", {})
        self.strength_estimation_method = memory_config.get("strength_estimation_method", "llm")
        self.classifier_model = None

        # Try to load the classifier so both scores are available, unless a
        # caller explicitly disables it for LLM-only extraction runs.
        classifier_model_path = memory_config.get(
            "classifier_model_path", "models/argument_quality_classifier"
        )
        if str(classifier_model_path).lower() in {"", "none", "disabled", "__disabled__"}:
            return

        if not os.path.isabs(classifier_model_path):
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../")
            )
            classifier_model_path = os.path.join(project_root, classifier_model_path)

        is_hf_model = "/" in classifier_model_path and not os.path.exists(classifier_model_path)

        if is_hf_model:
            try:
                from ..classifiers.hf_classifier import HuggingFaceClassifier
                self.classifier_model = HuggingFaceClassifier(classifier_model_path)
                print(f"   (Extractor) Loaded HF argument quality classifier: {classifier_model_path}")
            except Exception as e:
                print(f"   (Extractor) Warning: Failed to load HF classifier: {e}")
        elif os.path.exists(classifier_model_path):
            try:
                import sys
                import importlib.util

                model_dir = os.path.abspath(classifier_model_path)
                if model_dir not in sys.path:
                    sys.path.insert(0, model_dir)

                model_file_path = os.path.join(model_dir, "argument_quality_model.py")
                if not os.path.exists(model_file_path):
                    raise ImportError(f"Model file not found: {model_file_path}")

                spec = importlib.util.spec_from_file_location(
                    "argument_quality_model", model_file_path
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load module from: {model_file_path}")

                argument_quality_model = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(argument_quality_model)
                ArgumentQualityModel = argument_quality_model.ArgumentQualityModel

                self.classifier_model = ArgumentQualityModel(classifier_model_path)
                print(f"   (Extractor) Loaded local argument quality classifier from: {classifier_model_path}")
            except Exception as e:
                print(f"   (Extractor) Warning: Failed to load local classifier model: {e}")

    def extract(self, text: str, source_role: str, expected_polarity: str = None) -> List[ArgumentRecord]:
        """
        Parses the text and returns a list of ArgumentRecords.
        Uses a two-pass pipeline: Pass 1 distills claims, Pass 2 classifies them.
        """
        from ..constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE, POLARITY_MAPPING

        # --- PASS 1: DISTILLATION ---
        distill_system_prompt = f"""
        You are an expert conversational argument extractor.
        Your task is to identify and extract distinct persuasive points from a casual chat debate regarding the topic: "{self.topic_id}".

        Extraction rules:
        - Extract ANY stated reason, belief, anecdote, concern, rhetorical question, or personal feeling that acts as persuasion.
        - Only ignore pure filler ("hello", "how are you", "I have to go").
        - If the message contains multiple distinct points, extract them separately.
        - Re-phrase rhetorical questions or anecdotes into clear, standalone claims.

        For each argument, identify:
        1. The canonical claim: A clear, standalone sentence summarizing the point.
        2. Evidence snippets: Specific quotes from the text that support this claim.

        Output must be a valid JSON list of objects.
        Example:
        [
            {{
                "claim": "Intermittent fasting helps control weight by limiting eating windows.",
                "evidence_snippets": ["I would fast one day a week and be off of on the other six days"]
            }}
        ]
        """

        distill_prompt = f"Topic: {self.topic_id}\n\nText to analyze:\n{text}"

        response_1 = self.backend.chat(
            messages=[{"role": "user", "content": distill_prompt}],
            system_prompt=distill_system_prompt,
            temperature=0.0, # Deterministic for extraction
            max_tokens=2000
        )

        try:
            cleaned_1 = response_1.replace("```json", "").replace("```", "").strip()
            distilled_data = json.loads(cleaned_1)
            if not distilled_data:
                return []
        except Exception as e:
            print(f"Error parsing distillation response: {e}")
            print(f"Raw response: {response_1}")
            return []

        # Ensure we have a list of strings for claims
        claims_only = []
        for d in distilled_data:
            if isinstance(d, dict) and "claim" in d:
                claims_only.append(d["claim"])

        if not claims_only:
             return []

        # --- PASS 2: CLASSIFICATION ---
        classify_system_prompt = f"""
        You are an expert propositional logic classifier.
        Your task is to classify a list of distinct claims regarding the topic: "{self.topic_id}".

        For each claim, identify:
        1. The polarity: Must be exact string "affirmative" or "negative".
           - "affirmative": Supports, agrees with, or justifies the topic claim.
           - "negative": Opposes, disagrees with, or raises concerns about the topic claim.
           - Do NOT use synonyms. Use EXACTLY "affirmative" or "negative".
           CRITICAL: Do NOT confuse emotional tone with propositional polarity.
           A claim with an angry or negative tone might still factually agree with the topic, and a polite claim might disagree.
           Ask yourself: "Does this claim's FACTUAL CONTENT agree or disagree with the topic?" Ignore all emotional language.
        2. Category tags: High-level themes (e.g., "humanitarian", "economic", "personal_experience").
        3. A support strength score (0.0 to 1.0). You MUST strictly anchor your scoring to this rubric:
           - 0.1 - 0.3: Bare assertions, unsupported opinions, purely emotional appeals, or rhetorical questions.
           - 0.4 - 0.6: Standard opinions with basic logical reasoning or hypothetical examples.
           - 0.7 - 0.8: Strong arguments backed by highly specific real-world examples, logical rigor, or established facts.
           - 0.9 - 1.0: Exceptional arguments backed by cited empirical data or mathematically irrefutable proof.

        Output must be a valid JSON list of objects IN THE EXACT SAME ORDER as the input claims.
        Example:
        [
            {{
                "polarity": "affirmative",
                "support_strength": 0.4,
                "category_tags": ["health", "diet"]
            }}
        ]
        """

        claims_json = json.dumps(claims_only, indent=2)
        classify_prompt = f"Topic: {self.topic_id}\n\nClaims to classify:\n{claims_json}"

        response_2 = self.backend.chat(
            messages=[{"role": "user", "content": classify_prompt}],
            system_prompt=classify_system_prompt,
            temperature=0.0,
            max_tokens=2000
        )

        try:
            cleaned_2 = response_2.replace("```json", "").replace("```", "").strip()
            classified_data = json.loads(cleaned_2)

            # Merge Pass 1 and Pass 2
            records = []
            # We iterate up to the min length just in case the LLM messes up the array size
            for i in range(min(len(distilled_data), len(classified_data))):
                item_1 = distilled_data[i]
                item_2 = classified_data[i]

                claim_text = item_1.get("claim", "")
                llm_strength = float(item_2.get("support_strength", 0.5))

                if self.classifier_model:
                    classifier_strength = self.classifier_model.predict(claim_text)
                else:
                    classifier_strength = llm_strength

                if self.strength_estimation_method == "classifier" and self.classifier_model:
                    active_strength = classifier_strength
                else:
                    active_strength = llm_strength

                # Robust Polarity mapping
                raw_polarity = item_2.get("polarity", "").lower().strip()
                mapped_polarity = POLARITY_MAPPING.get(raw_polarity, None)

                # Override if expected_polarity is forced
                final_polarity = expected_polarity if expected_polarity else mapped_polarity

                # Skip if polarity is invalid/unknown
                if not final_polarity:
                    print(f"   (Extractor) Warning: Skipping argument with unknown polarity: '{raw_polarity}'")
                    continue

                record = ArgumentRecord(
                    topic_id=self.topic_id,
                    claim=claim_text,
                    polarity=final_polarity,
                    source_role=source_role,
                    support_strength=active_strength,
                    support_strength_llm=llm_strength,
                    support_strength_classifier=classifier_strength,
                    evidence_snippets=item_1.get("evidence_snippets", []),
                    category_tags=item_2.get("category_tags", [])
                )
                records.append(record)
            return records

        except Exception as e:
            print(f"Error parsing classification response: {e}")
            print(f"Raw response: {response_2}")
            return []
