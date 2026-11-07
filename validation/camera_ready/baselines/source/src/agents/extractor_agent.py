import json
import math
import os
from typing import Any, Callable, Dict, List, Optional
from ..backend.backend_base import LLMBackend
from ..memory.argument_record import ArgumentRecord


LEGACY_EXTRACTION_OUTPUT_MODE = "legacy_text_v1"
STRUCTURED_EXTRACTION_OUTPUT_MODE = "structured_json_v1"
SUPPORTED_EXTRACTION_OUTPUT_MODES = frozenset(
    {
        LEGACY_EXTRACTION_OUTPUT_MODE,
        STRUCTURED_EXTRACTION_OUTPUT_MODE,
    }
)


class StructuredExtractionError(RuntimeError):
    """Raised when strict structured extraction exhausts its retry budget."""

    def __init__(self, stage: str, attempts: int, last_error: Exception):
        self.stage = stage
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Structured extraction stage {stage!r} failed after "
            f"{attempts} attempt(s): {last_error}"
        )


class _StructuredValidationError(ValueError):
    """Internal distinction between JSON parsing and contract validation."""


class ArgumentExtractionAgent:
    """
    Extracts structured arguments from a debate turn using the LLM.
    Can use either LLM or a classifier model to estimate support_strength.
    """
    def __init__(self, backend: LLMBackend, topic_id: str, config: Optional[Dict] = None):
        self.backend = backend
        self.topic_id = topic_id
        self.config = config or {}

        extraction_config = self.config.get("extraction", {})
        self.output_mode = extraction_config.get(
            "output_mode",
            LEGACY_EXTRACTION_OUTPUT_MODE,
        )
        if self.output_mode not in SUPPORTED_EXTRACTION_OUTPUT_MODES:
            supported = ", ".join(sorted(SUPPORTED_EXTRACTION_OUTPUT_MODES))
            raise ValueError(
                f"Unknown extraction.output_mode={self.output_mode!r}; "
                f"expected one of: {supported}"
            )

        self.strict_parsing = extraction_config.get("strict_parsing", True)
        if not isinstance(self.strict_parsing, bool):
            raise ValueError("extraction.strict_parsing must be a boolean")

        self.max_attempts = extraction_config.get("max_attempts", 2)
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("extraction.max_attempts must be a positive integer")

        self.max_tokens = extraction_config.get("max_tokens", 2000)
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError("extraction.max_tokens must be a positive integer")

        self._audit = {
            "extract_calls": 0,
            "calls": 0,
            "passes": 0,
            "retries": 0,
            "parse_failures": 0,
            "api_failures": 0,
            "validation_failures": 0,
            "valid_empty": 0,
        }

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

    def audit_summary(self) -> Dict[str, int]:
        """Return cumulative structured-extraction health counters.

        ``calls`` and ``passes`` count backend stage attempts and successfully
        validated stage responses, respectively. ``extract_calls`` counts
        top-level structured ``extract`` invocations.
        """
        return dict(self._audit)

    def extract(self, text: str, source_role: str, expected_polarity: str = None) -> List[ArgumentRecord]:
        """
        Parses the text and returns a list of ArgumentRecords.
        Uses a two-pass pipeline: Pass 1 distills claims, Pass 2 classifies them.
        """
        if self.output_mode == STRUCTURED_EXTRACTION_OUTPUT_MODE:
            self._audit["extract_calls"] += 1
            return self._extract_structured(
                text,
                source_role,
                expected_polarity=expected_polarity,
            )

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

    def _extract_structured(
        self,
        text: str,
        source_role: str,
        expected_polarity: Optional[str] = None,
    ) -> List[ArgumentRecord]:
        distillation_prompt = f"""
You are an expert conversational argument extractor.
Identify distinct persuasive points in a casual debate about:
"{self.topic_id}"

Rules:
- Extract any stated reason, belief, anecdote, concern, rhetorical question,
  or personal feeling that acts as persuasion.
- Ignore only pure conversational filler.
- Split distinct points into separate arguments in source order, up to 12.
- Rewrite rhetorical questions and anecdotes as standalone claims.
- Evidence snippets must be verbatim substrings from the input when possible.

Return only the object required by the supplied JSON schema.

Text to analyze:
{text}
""".strip()
        distilled = self._run_structured_stage(
            stage="distillation",
            prompt=distillation_prompt,
            response_format=self._distillation_response_format(),
            validator=self._validate_distillation,
        )
        if distilled is None:
            return []

        if not distilled:
            self._audit["valid_empty"] += 1
            return []

        claims = [item["claim"] for item in distilled]
        classification_prompt = f"""
You are an expert propositional-logic classifier.
Classify each claim relative to this proposition:
"{self.topic_id}"

Requirements:
- Return exactly one classification for every input claim.
- Preserve the exact input order.
- Copy each claim and its zero-based claim_index exactly.
- polarity must be exactly "affirmative" or "negative".
- support_strength must be in [0.0, 1.0], using this rubric:
  0.1-0.3 bare assertion/emotional appeal;
  0.4-0.6 ordinary reasoning or hypothetical example;
  0.7-0.8 specific evidence or strong logical support;
  0.9-1.0 cited empirical data or irrefutable proof.
- category_tags must contain only high-level theme strings.

Return only the object required by the supplied JSON schema.

Claims:
{json.dumps(claims, ensure_ascii=False, indent=2)}
""".strip()
        classified = self._run_structured_stage(
            stage="classification",
            prompt=classification_prompt,
            response_format=self._classification_response_format(),
            validator=lambda payload: self._validate_classification(
                payload,
                claims,
            ),
        )
        if classified is None:
            return []

        return self._build_structured_records(
            distilled,
            classified,
            source_role=source_role,
            expected_polarity=expected_polarity,
        )

    def _run_structured_stage(
        self,
        *,
        stage: str,
        prompt: str,
        response_format: Dict[str, Any],
        validator: Callable[[Any], Any],
    ):
        last_error: Optional[Exception] = None
        retry_feedback: Optional[str] = None
        for attempt_index in range(self.max_attempts):
            if attempt_index > 0:
                self._audit["retries"] += 1
            self._audit["calls"] += 1
            attempt_prompt = prompt
            if attempt_index > 0 and retry_feedback:
                attempt_prompt += (
                    "\n\nThe previous attempt violated the output contract: "
                    f"{retry_feedback}. Correct that failure and return a complete "
                    "schema-valid object."
                )

            try:
                response = self.backend.get_completion(
                    prompt=attempt_prompt,
                    response_format=response_format,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
            except Exception as error:
                self._audit["api_failures"] += 1
                last_error = error
                retry_feedback = "the API call did not complete"
                continue

            try:
                payload = json.loads(response)
            except (json.JSONDecodeError, TypeError) as error:
                self._audit["parse_failures"] += 1
                last_error = error
                retry_feedback = "the response was incomplete or invalid JSON"
                continue

            try:
                validated = validator(payload)
            except _StructuredValidationError as error:
                self._audit["validation_failures"] += 1
                last_error = error
                retry_feedback = str(error)
                continue

            self._audit["passes"] += 1
            return validated

        if last_error is None:
            last_error = RuntimeError("structured extraction failed unexpectedly")
        if self.strict_parsing:
            raise StructuredExtractionError(
                stage,
                self.max_attempts,
                last_error,
            ) from last_error
        return None

    @staticmethod
    def _distillation_response_format() -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "argument_distillation_v1",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "arguments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim": {"type": "string"},
                                    "evidence_snippets": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "claim",
                                    "evidence_snippets",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["arguments"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _classification_response_format() -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "argument_classification_v1",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "classifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim_index": {"type": "integer"},
                                    "claim": {"type": "string"},
                                    "polarity": {
                                        "type": "string",
                                        "enum": [
                                            "affirmative",
                                            "negative",
                                        ],
                                    },
                                    "support_strength": {
                                        "type": "number",
                                    },
                                    "category_tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "claim_index",
                                    "claim",
                                    "polarity",
                                    "support_strength",
                                    "category_tags",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["classifications"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _validate_distillation(payload: Any) -> List[Dict[str, Any]]:
        if type(payload) is not dict or set(payload) != {"arguments"}:
            raise _StructuredValidationError(
                "distillation must be an exact top-level object with "
                "the key 'arguments'"
            )

        arguments = payload["arguments"]
        if type(arguments) is not list:
            raise _StructuredValidationError("arguments must be a list")
        if len(arguments) > 12:
            raise _StructuredValidationError(
                f"arguments must contain at most 12 items, got {len(arguments)}"
            )

        expected_fields = {"claim", "evidence_snippets"}
        validated = []
        for index, item in enumerate(arguments):
            if type(item) is not dict or set(item) != expected_fields:
                raise _StructuredValidationError(
                    f"argument {index} has incorrect fields"
                )
            claim = item["claim"]
            snippets = item["evidence_snippets"]
            if type(claim) is not str or not claim.strip():
                raise _StructuredValidationError(
                    f"argument {index} claim must be a non-empty string"
                )
            if type(snippets) is not list or any(
                type(snippet) is not str or not snippet.strip()
                for snippet in snippets
            ):
                raise _StructuredValidationError(
                    f"argument {index} evidence_snippets must contain strings"
                )
            validated.append(
                {
                    "claim": claim,
                    "evidence_snippets": list(snippets),
                }
            )
        return validated

    @staticmethod
    def _validate_classification(
        payload: Any,
        claims: List[str],
    ) -> List[Dict[str, Any]]:
        if type(payload) is not dict or set(payload) != {"classifications"}:
            raise _StructuredValidationError(
                "classification must be an exact top-level object with "
                "the key 'classifications'"
            )

        classifications = payload["classifications"]
        if type(classifications) is not list:
            raise _StructuredValidationError("classifications must be a list")
        if len(classifications) != len(claims):
            raise _StructuredValidationError(
                "classification length mismatch: "
                f"expected {len(claims)}, got {len(classifications)}"
            )

        expected_fields = {
            "claim_index",
            "claim",
            "polarity",
            "support_strength",
            "category_tags",
        }
        validated = []
        for index, (item, expected_claim) in enumerate(
            zip(classifications, claims)
        ):
            if type(item) is not dict or set(item) != expected_fields:
                raise _StructuredValidationError(
                    f"classification {index} has incorrect fields"
                )
            if type(item["claim_index"]) is not int:
                raise _StructuredValidationError(
                    f"classification {index} claim_index must be an integer"
                )
            if item["claim_index"] != index or item["claim"] != expected_claim:
                raise _StructuredValidationError(
                    f"classification {index} does not preserve claim order"
                )
            if item["polarity"] not in {"affirmative", "negative"}:
                raise _StructuredValidationError(
                    f"classification {index} has invalid polarity"
                )

            strength = item["support_strength"]
            if (
                isinstance(strength, bool)
                or not isinstance(strength, (int, float))
                or not math.isfinite(float(strength))
                or not 0.0 <= float(strength) <= 1.0
            ):
                raise _StructuredValidationError(
                    f"classification {index} has invalid support_strength"
                )

            tags = item["category_tags"]
            if type(tags) is not list or any(
                type(tag) is not str or not tag.strip()
                for tag in tags
            ):
                raise _StructuredValidationError(
                    f"classification {index} category_tags must contain strings"
                )
            validated.append(
                {
                    "claim_index": index,
                    "claim": expected_claim,
                    "polarity": item["polarity"],
                    "support_strength": float(strength),
                    "category_tags": list(tags),
                }
            )
        return validated

    def _build_structured_records(
        self,
        distilled: List[Dict[str, Any]],
        classified: List[Dict[str, Any]],
        *,
        source_role: str,
        expected_polarity: Optional[str],
    ) -> List[ArgumentRecord]:
        from ..constants import (
            POLARITY_AFFIRMATIVE,
            POLARITY_MAPPING,
            POLARITY_NEGATIVE,
        )

        forced_polarity = None
        if expected_polarity:
            forced_polarity = POLARITY_MAPPING.get(
                expected_polarity,
                expected_polarity,
            )
            if forced_polarity not in {
                POLARITY_AFFIRMATIVE,
                POLARITY_NEGATIVE,
            }:
                raise _StructuredValidationError(
                    f"Invalid expected_polarity={expected_polarity!r}"
                )

        records = []
        for distilled_item, classified_item in zip(distilled, classified):
            claim = distilled_item["claim"]
            llm_strength = classified_item["support_strength"]
            if self.classifier_model:
                classifier_strength = self.classifier_model.predict(claim)
            else:
                classifier_strength = llm_strength

            if (
                self.strength_estimation_method == "classifier"
                and self.classifier_model
            ):
                active_strength = classifier_strength
            else:
                active_strength = llm_strength

            records.append(
                ArgumentRecord(
                    topic_id=self.topic_id,
                    claim=claim,
                    polarity=(
                        forced_polarity
                        if forced_polarity
                        else classified_item["polarity"]
                    ),
                    source_role=source_role,
                    support_strength=active_strength,
                    support_strength_llm=llm_strength,
                    support_strength_classifier=classifier_strength,
                    evidence_snippets=distilled_item["evidence_snippets"],
                    category_tags=classified_item["category_tags"],
                )
            )
        return records
