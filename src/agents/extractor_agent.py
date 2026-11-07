import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from ..backend.backend_base import LLMBackend
from ..memory.argument_record import ArgumentRecord


LEGACY_EXTRACTION_OUTPUT_MODE = "legacy_text_v1"
STRUCTURED_EXTRACTION_OUTPUT_MODE = "structured_json_v1"
WHOLE_MESSAGE_EXTRACTION_OUTPUT_MODE = "whole_message_json_v1"
SUPPORTED_EXTRACTION_OUTPUT_MODES = frozenset(
    {
        LEGACY_EXTRACTION_OUTPUT_MODE,
        STRUCTURED_EXTRACTION_OUTPUT_MODE,
        WHOLE_MESSAGE_EXTRACTION_OUTPUT_MODE,
    }
)
SUPPORTED_STRENGTH_ESTIMATION_METHODS = frozenset(
    {"classifier", "llm", "uniform_fixed"}
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
            "records_emitted": 0,
            "zero_record_extract_calls": 0,
            "single_record_extract_calls": 0,
            "multi_record_extract_calls": 0,
            "distillation_passes": 0,
            "classification_passes": 0,
            "uniform_strength_records": 0,
            "nonuniform_strength_records": 0,
            "classifier_strength_records": 0,
        }
        self._active_strength_min: Optional[float] = None
        self._active_strength_max: Optional[float] = None
        self._classifier_strength_min: Optional[float] = None
        self._classifier_strength_max: Optional[float] = None
        # Report-only manipulation telemetry. These numeric scores and
        # one-way record fingerprints are never consulted by extraction,
        # memory ingestion, response generation, or campaign continuation.
        self._manipulation_events: List[Dict[str, Any]] = []

        memory_config = self.config.get("memory", {})
        self.strength_estimation_method = memory_config.get(
            "strength_estimation_method",
            "llm",
        )
        if (
            self.strength_estimation_method
            not in SUPPORTED_STRENGTH_ESTIMATION_METHODS
        ):
            supported = ", ".join(
                sorted(SUPPORTED_STRENGTH_ESTIMATION_METHODS)
            )
            raise ValueError(
                "Unknown memory.strength_estimation_method="
                f"{self.strength_estimation_method!r}; expected one of: "
                f"{supported}"
            )
        self.uniform_strength_value = memory_config.get(
            "uniform_strength_value"
        )
        if self.strength_estimation_method == "uniform_fixed":
            if (
                isinstance(self.uniform_strength_value, bool)
                or not isinstance(self.uniform_strength_value, (int, float))
                or not math.isfinite(float(self.uniform_strength_value))
                or not 0.0 <= float(self.uniform_strength_value) <= 1.0
            ):
                raise ValueError(
                    "memory.uniform_strength_value must be a finite number "
                    "in [0,1] for uniform_fixed scoring"
                )
            self.uniform_strength_value = float(
                self.uniform_strength_value
            )
        self.classifier_model = None

        # LLM-only runs may collect a supplementary classifier score. A run
        # requesting classifier strength must never silently become an LLM run.
        reference = memory_config.get("classifier_model_path", "models/argument_quality_classifier")
        disabled = str(reference).lower() in {"", "none", "disabled", "__disabled__"}
        try:
            if disabled:
                raise ValueError("Classifier is disabled in configuration.")
            self.classifier_model = self._load_classifier(memory_config)
        except Exception as exc:
            if self.strength_estimation_method == "classifier":
                if not memory_config.get("classifier_allow_llm_fallback", False):
                    raise RuntimeError(
                        "Argument classifier failed to load; refusing to change the scoring method. "
                        "Install the classifier dependencies and verify the model download."
                    ) from exc
                self.strength_estimation_method = "llm"
                print(f"   (Extractor) Explicitly enabled LLM fallback: {exc}")
            elif not disabled:
                print(f"   (Extractor) Supplementary classifier unavailable: {exc}")

    @staticmethod
    def _load_classifier(memory_config):
        reference = str(memory_config.get("classifier_model_path", "models/argument_quality_classifier"))
        candidate = Path(reference).expanduser()
        project_candidate = Path(__file__).resolve().parents[2] / candidate
        is_local = (candidate.exists() or project_candidate.exists() or candidate.is_absolute()
                    or reference.startswith(("./", "../", "models/", "~")))
        if is_local:
            candidate = candidate if candidate.exists() or candidate.is_absolute() else project_candidate
            if not candidate.is_dir():
                raise FileNotFoundError(f"Classifier directory does not exist: {candidate}")
            reference = str(candidate.resolve())

        # Preserve the previous local-regressor/remote-reward distinction.
        # Set argument_quality explicitly for our Hub-hosted model.
        kind = memory_config.get("classifier_kind", "auto")
        if kind == "auto":
            kind = "argument_quality" if is_local else "reward"
        if kind == "argument_quality":
            from ..classifiers.argument_quality import ArgumentQualityModel
            return ArgumentQualityModel(
                reference, revision=memory_config.get("classifier_revision"),
                local_files_only=is_local,
            )
        if kind == "reward":
            from ..classifiers.hf_classifier import HuggingFaceClassifier
            return HuggingFaceClassifier(reference)
        raise ValueError(f"Unknown classifier_kind: {kind}")

    def audit_summary(
        self,
        include_manipulation_telemetry: bool = True,
    ) -> Dict[str, Any]:
        """Return cumulative structured-extraction health counters.

        ``calls`` and ``passes`` count backend stage attempts and successfully
        validated stage responses, respectively. ``extract_calls`` counts
        top-level structured ``extract`` invocations.
        """
        summary = dict(self._audit)
        summary.update(
            {
                "output_mode": self.output_mode,
                "strict_parsing": self.strict_parsing,
                "strength_estimation_method": (
                    self.strength_estimation_method
                ),
                "uniform_strength_value": self.uniform_strength_value,
                "classifier_model_available": self.classifier_model is not None,
                "active_strength_min": self._active_strength_min,
                "active_strength_max": self._active_strength_max,
                "classifier_strength_min": self._classifier_strength_min,
                "classifier_strength_max": self._classifier_strength_max,
            }
        )
        if include_manipulation_telemetry:
            summary.update(
                {
                    "manipulation_events": [
                        {
                            **event,
                            "record_fingerprints": list(
                                event["record_fingerprints"]
                            ),
                            "classifier_strengths": list(
                                event["classifier_strengths"]
                            ),
                            "active_strengths": list(
                                event["active_strengths"]
                            ),
                        }
                        for event in self._manipulation_events
                    ],
                    "manipulation_telemetry_use": (
                        "report_only_never_controls_generation_or_continuation"
                    ),
                }
            )
        return summary

    def extract(self, text: str, source_role: str, expected_polarity: str = None) -> List[ArgumentRecord]:
        """
        Parses the text and returns a list of ArgumentRecords.
        Uses a two-pass pipeline: Pass 1 distills claims, Pass 2 classifies them.
        """
        if self.output_mode in {
            STRUCTURED_EXTRACTION_OUTPUT_MODE,
            WHOLE_MESSAGE_EXTRACTION_OUTPUT_MODE,
        }:
            self._audit["extract_calls"] += 1
            if self.output_mode == STRUCTURED_EXTRACTION_OUTPUT_MODE:
                records = self._extract_structured(
                    text,
                    source_role,
                    expected_polarity=expected_polarity,
                )
            else:
                records = self._extract_whole_message_structured(
                    text,
                    source_role,
                    expected_polarity=expected_polarity,
                )
            record_count = len(records)
            self._audit["records_emitted"] += record_count
            record_fingerprints = []
            classifier_strengths = []
            active_strengths = []
            for record in records:
                strength = float(record.support_strength)
                active_strengths.append(strength)
                self._active_strength_min = (
                    strength
                    if self._active_strength_min is None
                    else min(self._active_strength_min, strength)
                )
                self._active_strength_max = (
                    strength
                    if self._active_strength_max is None
                    else max(self._active_strength_max, strength)
                )
                key = "nonuniform_strength_records"
                if (
                    self.strength_estimation_method == "uniform_fixed"
                    and abs(
                        strength - float(self.uniform_strength_value)
                    )
                    <= 1e-12
                ):
                    key = "uniform_strength_records"
                self._audit[key] += 1
                classifier_strength = float(
                    record.support_strength_classifier
                )
                classifier_strengths.append(classifier_strength)
                self._audit["classifier_strength_records"] += 1
                self._classifier_strength_min = (
                    classifier_strength
                    if self._classifier_strength_min is None
                    else min(
                        self._classifier_strength_min,
                        classifier_strength,
                    )
                )
                self._classifier_strength_max = (
                    classifier_strength
                    if self._classifier_strength_max is None
                    else max(
                        self._classifier_strength_max,
                        classifier_strength,
                    )
                )
                identity_payload = json.dumps(
                    {
                        "claim": str(record.claim),
                        "polarity": str(record.polarity),
                        "source_role": str(record.source_role),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                record_fingerprints.append(
                    hashlib.sha256(identity_payload).hexdigest()
                )
            self._manipulation_events.append(
                {
                    "call_index": len(self._manipulation_events) + 1,
                    "output_mode": self.output_mode,
                    "source_role": str(source_role),
                    "records_emitted": record_count,
                    "record_fingerprints": record_fingerprints,
                    "record_identity_sequence_sha256": hashlib.sha256(
                        json.dumps(
                            record_fingerprints,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "classifier_strengths": classifier_strengths,
                    "active_strengths": active_strengths,
                }
            )
            if record_count == 0:
                self._audit["zero_record_extract_calls"] += 1
            elif record_count == 1:
                self._audit["single_record_extract_calls"] += 1
            else:
                self._audit["multi_record_extract_calls"] += 1
            return records

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

                active_strength = self._active_strength(
                    llm_strength,
                    classifier_strength,
                )

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

    def _extract_whole_message_structured(
        self,
        text: str,
        source_role: str,
        expected_polarity: Optional[str] = None,
    ) -> List[ArgumentRecord]:
        """Represent one complete generated message as one verbatim record.

        The flat-memory ablation removes claim distillation and decomposition,
        while retaining the minimum proposition-relative polarity and strength
        metadata required by the unchanged Belief Engine update rule.
        """
        if type(text) is not str or not text.strip():
            raise _StructuredValidationError(
                "whole_message_json_v1 requires a non-empty message string"
            )

        claims = [text]
        classification_prompt = f"""
You are an expert propositional-logic classifier.
Classify the complete debate message below as one indivisible evidence record
relative to this proposition:
"{self.topic_id}"

Requirements:
- Return exactly one classification.
- Copy the complete input message and zero-based claim_index exactly.
- polarity must be exactly "affirmative" or "negative".
- Judge polarity from the message's overall factual direction relative to the
  proposition, not its emotional tone.
- support_strength must be in [0.0, 1.0], using this rubric:
  0.1-0.3 bare assertion/emotional appeal;
  0.4-0.6 ordinary reasoning or hypothetical example;
  0.7-0.8 specific evidence or strong logical support;
  0.9-1.0 cited empirical data or irrefutable proof.
- category_tags must contain only high-level theme strings.
- Do not split, summarize, rewrite, or omit any part of the message.

Return only the object required by the supplied JSON schema.

Complete message:
{json.dumps(text, ensure_ascii=False)}
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
            [{"claim": text, "evidence_snippets": [text]}],
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
            stage_pass_key = f"{stage}_passes"
            if stage_pass_key in self._audit:
                self._audit[stage_pass_key] += 1
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
                            "maxItems": 12,
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

            active_strength = self._active_strength(
                llm_strength,
                classifier_strength,
            )

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

    def _active_strength(
        self,
        llm_strength: float,
        classifier_strength: float,
    ) -> float:
        """Select the update-time score without changing claim or polarity."""
        if self.strength_estimation_method == "uniform_fixed":
            return float(self.uniform_strength_value)
        if (
            self.strength_estimation_method == "classifier"
            and self.classifier_model
        ):
            return classifier_strength
        return llm_strength
