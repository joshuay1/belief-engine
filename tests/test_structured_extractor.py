"""Offline contract tests for opt-in structured argument extraction."""

from __future__ import annotations

import json
import unittest

from src.agents.extractor_agent import (
    ArgumentExtractionAgent,
    StructuredExtractionError,
)


class _ScriptedBackend:
    def __init__(self, *, completions=(), chats=()):
        self.completions = list(completions)
        self.chats = list(chats)
        self.completion_calls = []
        self.chat_calls = []

    def get_completion(
        self,
        prompt,
        response_format,
        temperature=0.7,
        max_tokens=None,
    ):
        self.completion_calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self.completions:
            raise AssertionError("unexpected get_completion call")
        response = self.completions.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def chat(
        self,
        messages,
        system_prompt=None,
        temperature=0.7,
        max_tokens=1000,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self.chats:
            raise AssertionError("unexpected chat call")
        response = self.chats.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _ScriptedClassifier:
    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def predict(self, _claim):
        return next(self.predictions)


def _config(
    *,
    structured=True,
    output_mode="structured_json_v1",
    strict=True,
    max_attempts=2,
    max_tokens=2000,
    strength_estimation_method="llm",
    uniform_strength_value=None,
):
    config = {
        "memory": {
            "classifier_model_path": "disabled",
            "strength_estimation_method": strength_estimation_method,
            "uniform_strength_value": uniform_strength_value,
        }
    }
    if structured:
        config["extraction"] = {
            "output_mode": output_mode,
            "strict_parsing": strict,
            "max_attempts": max_attempts,
            "max_tokens": max_tokens,
        }
    return config


def _distillation(*claims):
    return json.dumps(
        {
            "arguments": [
                {
                    "claim": claim,
                    "evidence_snippets": [f"evidence {index}"],
                }
                for index, claim in enumerate(claims)
            ]
        }
    )


def _classification(*items):
    return json.dumps({"classifications": list(items)})


def _classified(index, claim, polarity="affirmative", strength=0.6):
    return {
        "claim_index": index,
        "claim": claim,
        "polarity": polarity,
        "support_strength": strength,
        "category_tags": ["test"],
    }


class StructuredExtractorTests(unittest.TestCase):
    def test_distillation_schema_enforces_registered_twelve_item_cap(self):
        response_format = (
            ArgumentExtractionAgent._distillation_response_format()
        )
        arguments_schema = response_format["json_schema"]["schema"][
            "properties"
        ]["arguments"]

        self.assertEqual(arguments_schema["type"], "array")
        self.assertEqual(arguments_schema["maxItems"], 12)

    def test_truncated_distillation_retries_then_succeeds(self):
        claim = "Public transit reduces urban congestion."
        backend = _ScriptedBackend(
            completions=[
                '{"arguments": [',
                _distillation(claim),
                _classification(_classified(0, claim, strength=0.7)),
            ]
        )
        extractor = ArgumentExtractionAgent(
            backend,
            topic_id="Cities should invest in public transit",
            config=_config(),
        )

        records = extractor.extract(
            "Buses reduce congestion.",
            source_role="opponent",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].claim, claim)
        self.assertEqual(records[0].support_strength, 0.7)
        self.assertEqual(len(backend.completion_calls), 3)
        self.assertFalse(backend.chat_calls)
        for call in backend.completion_calls:
            self.assertEqual(call["max_tokens"], 2000)
            self.assertEqual(call["temperature"], 0.0)
            response_format = call["response_format"]
            self.assertEqual(response_format["type"], "json_schema")
            self.assertTrue(response_format["json_schema"]["strict"])
            self.assertEqual(
                response_format["json_schema"]["schema"]["type"],
                "object",
            )

        audit = extractor.audit_summary()
        manipulation_events = audit.pop("manipulation_events")
        telemetry_use = audit.pop("manipulation_telemetry_use")
        self.assertEqual(
            audit,
            {
                "extract_calls": 1,
                "calls": 3,
                "passes": 2,
                "retries": 1,
                "parse_failures": 1,
                "api_failures": 0,
                "validation_failures": 0,
                "valid_empty": 0,
                "records_emitted": 1,
                "zero_record_extract_calls": 0,
                "single_record_extract_calls": 1,
                "multi_record_extract_calls": 0,
                "distillation_passes": 1,
                "classification_passes": 1,
                "uniform_strength_records": 0,
                "nonuniform_strength_records": 1,
                "classifier_strength_records": 1,
                "output_mode": "structured_json_v1",
                "strict_parsing": True,
                "strength_estimation_method": "llm",
                "uniform_strength_value": None,
                "classifier_model_available": False,
                "active_strength_min": 0.7,
                "active_strength_max": 0.7,
                "classifier_strength_min": 0.7,
                "classifier_strength_max": 0.7,
            },
        )
        self.assertEqual(len(manipulation_events), 1)
        self.assertEqual(
            manipulation_events[0]["classifier_strengths"],
            [0.7],
        )
        self.assertEqual(
            manipulation_events[0]["active_strengths"],
            [0.7],
        )
        self.assertEqual(
            telemetry_use,
            "report_only_never_controls_generation_or_continuation",
        )

    def test_exhausted_strict_parsing_raises_dedicated_error(self):
        backend = _ScriptedBackend(
            completions=['{"arguments": [', "not-json"]
        )
        extractor = ArgumentExtractionAgent(
            backend,
            topic_id="Test proposition",
            config=_config(max_attempts=2),
        )

        with self.assertRaises(StructuredExtractionError) as caught:
            extractor.extract("Some evidence.", source_role="opponent")

        self.assertEqual(caught.exception.stage, "distillation")
        self.assertEqual(caught.exception.attempts, 2)
        audit = extractor.audit_summary()
        self.assertEqual(audit.pop("manipulation_events"), [])
        self.assertEqual(
            audit.pop("manipulation_telemetry_use"),
            "report_only_never_controls_generation_or_continuation",
        )
        self.assertEqual(
            audit,
            {
                "extract_calls": 1,
                "calls": 2,
                "passes": 0,
                "retries": 1,
                "parse_failures": 2,
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
                "output_mode": "structured_json_v1",
                "strict_parsing": True,
                "strength_estimation_method": "llm",
                "uniform_strength_value": None,
                "classifier_model_available": False,
                "active_strength_min": None,
                "active_strength_max": None,
                "classifier_strength_min": None,
                "classifier_strength_max": None,
            },
        )

    def test_valid_empty_distillation_returns_empty_without_classification(self):
        backend = _ScriptedBackend(
            completions=[json.dumps({"arguments": []})]
        )
        extractor = ArgumentExtractionAgent(
            backend,
            topic_id="Test proposition",
            config=_config(),
        )

        records = extractor.extract("Hello there.", source_role="opponent")

        self.assertEqual(records, [])
        self.assertEqual(len(backend.completion_calls), 1)
        self.assertEqual(extractor.audit_summary()["passes"], 1)
        self.assertEqual(extractor.audit_summary()["valid_empty"], 1)

    def test_api_failure_is_counted_and_retried(self):
        claim = "Reliable extraction should survive a transient API error."
        backend = _ScriptedBackend(
            completions=[
                RuntimeError("transient"),
                _distillation(claim),
                _classification(_classified(0, claim)),
            ]
        )
        extractor = ArgumentExtractionAgent(
            backend,
            topic_id="Test proposition",
            config=_config(),
        )

        records = extractor.extract(
            "Transient failures happen.",
            source_role="opponent",
        )

        self.assertEqual(len(records), 1)
        summary = extractor.audit_summary()
        self.assertEqual(summary["api_failures"], 1)
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["calls"], 3)
        self.assertEqual(summary["passes"], 2)

    def test_classification_length_mismatch_retries_exact_order(self):
        claims = [
            "The policy lowers costs.",
            "The policy improves access.",
        ]
        backend = _ScriptedBackend(
            completions=[
                _distillation(*claims),
                _classification(_classified(0, claims[0])),
                _classification(
                    _classified(0, claims[0], strength=0.4),
                    _classified(
                        1,
                        claims[1],
                        polarity="negative",
                        strength=0.8,
                    ),
                ),
            ]
        )
        extractor = ArgumentExtractionAgent(
            backend,
            topic_id="Test proposition",
            config=_config(),
        )

        records = extractor.extract(
            "The policy lowers costs but may reduce access.",
            source_role="opponent",
        )

        self.assertEqual([record.claim for record in records], claims)
        self.assertEqual(
            [record.polarity for record in records],
            ["affirmative", "negative"],
        )
        summary = extractor.audit_summary()
        self.assertEqual(summary["calls"], 3)
        self.assertEqual(summary["passes"], 2)
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["validation_failures"], 1)
        self.assertEqual(summary["parse_failures"], 0)

    def test_whole_message_mode_emits_exactly_one_unchanged_record(self):
        message = (
            "The reform could reduce costs, although implementation may be "
            "difficult."
        )
        backend = _ScriptedBackend(
            completions=[
                _classification(
                    _classified(
                        0,
                        message,
                        polarity="affirmative",
                        strength=0.65,
                    )
                )
            ]
        )
        extractor = ArgumentExtractionAgent(
            backend,
            topic_id="We should adopt the reform",
            config=_config(output_mode="whole_message_json_v1"),
        )

        records = extractor.extract(message, source_role="opponent")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].claim, message)
        self.assertEqual(records[0].evidence_snippets, [message])
        self.assertEqual(records[0].polarity, "affirmative")
        self.assertEqual(records[0].support_strength, 0.65)
        self.assertEqual(len(backend.completion_calls), 1)
        prompt = backend.completion_calls[0]["prompt"]
        self.assertIn("one indivisible evidence record", prompt)
        self.assertIn(json.dumps(message, ensure_ascii=False), prompt)
        summary = extractor.audit_summary()
        self.assertEqual(summary["extract_calls"], 1)
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["passes"], 1)
        self.assertEqual(summary["distillation_passes"], 0)
        self.assertEqual(summary["classification_passes"], 1)
        self.assertEqual(summary["records_emitted"], 1)
        self.assertEqual(summary["single_record_extract_calls"], 1)
        self.assertEqual(summary["zero_record_extract_calls"], 0)
        self.assertEqual(summary["multi_record_extract_calls"], 0)

    def test_whole_message_mode_retries_schema_failure_without_distillation(self):
        message = "A complete message."
        backend = _ScriptedBackend(
            completions=[
                _classification(),
                _classification(_classified(0, message, strength=0.4)),
            ]
        )
        extractor = ArgumentExtractionAgent(
            backend,
            topic_id="Test proposition",
            config=_config(output_mode="whole_message_json_v1"),
        )

        records = extractor.extract(message, source_role="opponent")

        self.assertEqual(len(records), 1)
        summary = extractor.audit_summary()
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["validation_failures"], 1)
        self.assertEqual(summary["classification_passes"], 1)
        self.assertEqual(summary["distillation_passes"], 0)

    def test_uniform_strength_preserves_claim_and_polarity(self):
        fixed_strength = 0.8475946340113878
        claims = (
            "The policy improves access.",
            "The policy increases costs.",
        )
        backend = _ScriptedBackend(
            completions=[
                _distillation(*claims),
                _classification(
                    _classified(
                        0,
                        claims[0],
                        polarity="negative",
                        strength=0.23,
                    ),
                    _classified(
                        1,
                        claims[1],
                        polarity="affirmative",
                        strength=0.91,
                    ),
                ),
            ]
        )
        extractor = ArgumentExtractionAgent(
            backend,
            topic_id="Test proposition",
            config=_config(
                strength_estimation_method="uniform_fixed",
                uniform_strength_value=fixed_strength,
            ),
        )
        extractor.classifier_model = _ScriptedClassifier((0.31, 0.78))

        records = extractor.extract(
            " ".join(claims),
            source_role="opponent",
        )

        self.assertEqual([record.claim for record in records], list(claims))
        self.assertEqual(records[0].polarity, "negative")
        self.assertEqual(records[1].polarity, "affirmative")
        self.assertEqual(
            [record.support_strength for record in records],
            [fixed_strength, fixed_strength],
        )
        self.assertEqual(records[0].support_strength_llm, 0.23)
        self.assertEqual(
            [record.support_strength_classifier for record in records],
            [0.31, 0.78],
        )
        summary = extractor.audit_summary()
        self.assertEqual(summary["strength_estimation_method"], "uniform_fixed")
        self.assertEqual(summary["uniform_strength_value"], fixed_strength)
        self.assertTrue(summary["classifier_model_available"])
        self.assertEqual(summary["uniform_strength_records"], 2)
        self.assertEqual(summary["nonuniform_strength_records"], 0)
        self.assertEqual(summary["classifier_strength_records"], 2)
        self.assertEqual(summary["active_strength_min"], fixed_strength)
        self.assertEqual(summary["active_strength_max"], fixed_strength)
        self.assertEqual(summary["classifier_strength_min"], 0.31)
        self.assertEqual(summary["classifier_strength_max"], 0.78)

    def test_whole_message_mode_rejects_empty_input(self):
        extractor = ArgumentExtractionAgent(
            _ScriptedBackend(),
            topic_id="Test proposition",
            config=_config(output_mode="whole_message_json_v1"),
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires a non-empty message",
        ):
            extractor.extract("   ", source_role="opponent")

    def test_legacy_default_keeps_chat_list_pipeline_unchanged(self):
        claim = "Legacy extraction remains available."
        backend = _ScriptedBackend(
            chats=[
                json.dumps(
                    [
                        {
                            "claim": claim,
                            "evidence_snippets": ["legacy evidence"],
                        }
                    ]
                ),
                json.dumps(
                    [
                        {
                            "polarity": "affirmative",
                            "support_strength": 0.55,
                            "category_tags": ["legacy"],
                        }
                    ]
                ),
            ]
        )
        extractor = ArgumentExtractionAgent(
            backend,
            topic_id="Test proposition",
            config=_config(structured=False),
        )

        records = extractor.extract(
            "Legacy evidence.",
            source_role="opponent",
        )

        self.assertEqual(extractor.output_mode, "legacy_text_v1")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].claim, claim)
        self.assertEqual(records[0].support_strength, 0.55)
        self.assertEqual(len(backend.chat_calls), 2)
        self.assertFalse(backend.completion_calls)
        self.assertEqual(
            [call["max_tokens"] for call in backend.chat_calls],
            [2000, 2000],
        )
        audit = extractor.audit_summary()
        self.assertEqual(audit.pop("manipulation_events"), [])
        self.assertEqual(
            audit.pop("manipulation_telemetry_use"),
            "report_only_never_controls_generation_or_continuation",
        )
        self.assertEqual(
            audit,
            {
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
                "output_mode": "legacy_text_v1",
                "strict_parsing": True,
                "strength_estimation_method": "llm",
                "uniform_strength_value": None,
                "classifier_model_available": False,
                "active_strength_min": None,
                "active_strength_max": None,
                "classifier_strength_min": None,
                "classifier_strength_max": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
