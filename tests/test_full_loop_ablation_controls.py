from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.factory import create_agent
from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE
from src.memory.argument_record import ArgumentRecord
from src.memory.memory_store import MemoryStore
from src.simulation.simulation_runner import SimulationRunner
from validation.scripts import run_two_agent_model_comparison as comparison


def _record(
    argument_id: str,
    *,
    source_role: str,
    polarity: str = POLARITY_AFFIRMATIVE,
    strength: float = 0.7,
) -> ArgumentRecord:
    record = ArgumentRecord(
        id=argument_id,
        topic_id="Test proposition",
        claim=f"Claim {argument_id}",
        polarity=polarity,
        source_role=source_role,
        support_strength=strength,
    )
    record.credence_relevant = True
    return record


def _memory(directory: str, horizon: str) -> MemoryStore:
    return MemoryStore(
        storage_mode="episodic",
        retrieval_strategy="proportional",
        worldview_openness="balanced",
        stance_method="bayesian",
        update_sensitivity=0.2,
        anchor_sensitivity=0.4,
        confirmation_bias=0.0,
        allow_self_arguments=False,
        belief_engine_type="bayesian",
        belief_semantics_mode="active_explicit_prior_v1",
        dynamic_evidence_horizon=horizon,
        initial_stance=0.75,
        memory_dir=directory,
        agent_id="Test",
    )


class _Backend:
    def chat(self, **_kwargs):
        return "A deterministic response."


class _EmptyExtractor:
    topic_id = "Test proposition"

    def __init__(self):
        self.calls = 0

    def extract(self, **_kwargs):
        self.calls += 1
        return []

    def audit_summary(self):
        return {"extract_calls": self.calls, "valid_empty": self.calls}


class _RecordExtractor(_EmptyExtractor):
    def extract(self, **_kwargs):
        self.calls += 1
        return [_record("extracted", source_role="opponent")]


class _TrackingMemory:
    def __init__(self, agent_id, events):
        self.agent_id = agent_id
        self.arguments = {}
        self.events = events
        self._turns = []

    def prepare_for_opponent_evidence_turn(self, *, round_index, sender_name):
        self.events.append(("prepare", self.agent_id, sender_name, round_index))
        self._turns.append(
            {
                "round": round_index,
                "sender": sender_name,
                "finalized": False,
            }
        )
        return len(self._turns) - 1

    def finalize_opponent_evidence_turn(
        self,
        event_index,
        *,
        extracted_records,
        accepted_records,
        status_counts,
    ):
        self.events.append(
            (
                "finalize",
                self.agent_id,
                self._turns[event_index]["sender"],
                extracted_records,
            )
        )
        self._turns[event_index]["finalized"] = True

    def abort_opponent_evidence_turn(self, event_index, *, reason):
        self.events.append(("abort", self.agent_id, reason))
        self._turns[event_index].update(
            {"finalized": True, "aborted": True}
        )

    def evidence_horizon_audit_summary(self):
        completed = sum(
            1
            for turn in self._turns
            if turn["finalized"] and not turn.get("aborted")
        )
        return {
            "dynamic_evidence_horizon": "latest_opponent_turn",
            "prepare_calls": len(self._turns),
            "completed_turns": completed,
            "aborted_turns": sum(
                1 for turn in self._turns if turn.get("aborted")
            ),
            "open_turns": sum(
                1 for turn in self._turns if not turn["finalized"]
            ),
            "reset_calls": len(self._turns),
            "evicted_records": 0,
            "extracted_records": 0,
            "accepted_records": 0,
            "prior_reconstruction_error_max": 0.0,
            "active_state_recomputation_error_max": 0.0,
            "post_batch_dynamic_records_max": 0,
            "events": list(self._turns),
        }

    def get_all_arguments(self):
        return []


class _FailingTrackingMemory(_TrackingMemory):
    def add_argument(self, _argument, _backend):
        raise OSError("strict ingestion failed")

    def add_arguments_batch(self, _arguments, _backend):
        raise OSError("strict ingestion failed")


class _TrackingAgent:
    type = "subject"

    def __init__(self, name, events):
        self.name = name
        self.memory = _TrackingMemory(name, events)

    def get_current_stance(self):
        return 0.75 if self.name == "Pro" else -0.75

    def generate_reply(self, _transcript):
        return f"{self.name} reply", []


class _FailingTrackingAgent(_TrackingAgent):
    def __init__(self, name, events):
        self.name = name
        self.memory = _FailingTrackingMemory(name, events)


def _matched_args() -> argparse.Namespace:
    return argparse.Namespace(
        rounds=15,
        seeding_limit=10,
        temperature=0.7,
        self_update_temperature=0.0,
        response_shell="matched_v1",
        response_context_messages=4,
        self_update_context_messages=4,
        response_word_limit=120,
        speaker_order="pro-first",
        retrieval_strategy="proportional",
        retrieval_k=10,
        belief_semantics_mode="active_explicit_prior_v1",
        extraction_output_mode="structured_json_v1",
        strict_extraction=True,
        strict_memory_ingestion=True,
        extraction_max_attempts=2,
        extraction_max_tokens=2000,
        allow_self_arguments=False,
        judge_backend="openai",
        judge_model="gpt-5.4-mini",
        judge_reasoning_effort="none",
        judge_verbosity="low",
        judge_temperature=0.0,
        strict_judge=True,
        artifact_mode="compact",
    )


class FullLoopAblationControlTests(unittest.TestCase):
    def test_latest_turn_rejects_non_explicit_prior_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "active_explicit_prior_v1",
            ):
                MemoryStore(
                    belief_semantics_mode="incremental_v1",
                    dynamic_evidence_horizon="latest_opponent_turn",
                    memory_dir=directory,
                )

    def test_latest_turn_preserves_seeds_evicts_dynamic_and_rebuilds_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(directory, "latest_opponent_turn")
            seed = _record("seed", source_role="seed")
            old_pro = _record("old-pro", source_role="opponent")
            old_con = _record(
                "old-con",
                source_role="opponent",
                polarity=POLARITY_NEGATIVE,
                strength=0.3,
            )
            memory.arguments = {
                seed.id: seed,
                old_pro.id: old_pro,
                old_con.id: old_con,
            }
            memory.belief_engine.recompute_from_active_records(
                memory.arguments.values(),
                persist=False,
            )

            event_index = memory.prepare_for_opponent_evidence_turn(
                round_index=2,
                sender_name="Con",
            )
            memory.finalize_opponent_evidence_turn(
                event_index,
                extracted_records=0,
                accepted_records=0,
                status_counts={},
            )

            self.assertEqual(list(memory.arguments), ["seed"])
            self.assertAlmostEqual(memory.get_stance(verbose=False), 0.75)
            summary = memory.evidence_horizon_audit_summary()
            self.assertEqual(summary["prepare_calls"], 1)
            self.assertEqual(summary["reset_calls"], 1)
            self.assertEqual(summary["evicted_records"], 2)
            self.assertEqual(summary["post_batch_dynamic_records_max"], 0)
            self.assertTrue(summary["all_seed_records_preserved"])
            self.assertTrue(summary["all_latest_turn_bounds_ok"])
            self.assertLessEqual(
                summary["prior_reconstruction_error_max"],
                1e-12,
            )
            self.assertLessEqual(
                summary["active_state_recomputation_error_max"],
                1e-12,
            )

    def test_latest_turn_restores_every_mutable_seed_field(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(directory, "latest_opponent_turn")
            seed = _record("seed", source_role="seed")
            seed.tags = ["baseline"]
            seed.links = [{"kind": "baseline"}]
            memory.arguments = {seed.id: seed}

            first_event = memory.prepare_for_opponent_evidence_turn(
                round_index=1,
                sender_name="Con",
            )
            memory.finalize_opponent_evidence_turn(
                first_event,
                extracted_records=0,
                accepted_records=0,
                status_counts={},
            )
            pristine_state = copy.deepcopy(vars(memory.arguments[seed.id]))

            mutated = memory.arguments[seed.id]
            mutated.credence_relevant = False
            mutated.support_strength = 0.01
            mutated.retrieval_count = 99
            mutated.tags.append("cross-turn-mutation")
            mutated.links[0]["kind"] = "mutated"
            memory.arguments["dynamic"] = _record(
                "dynamic",
                source_role="opponent",
            )

            second_event = memory.prepare_for_opponent_evidence_turn(
                round_index=2,
                sender_name="Con",
            )
            memory.finalize_opponent_evidence_turn(
                second_event,
                extracted_records=0,
                accepted_records=0,
                status_counts={},
            )

            self.assertEqual(set(memory.arguments), {seed.id})
            self.assertEqual(vars(memory.arguments[seed.id]), pristine_state)
            summary = memory.evidence_horizon_audit_summary()
            self.assertTrue(summary["all_seed_state_restorations_ok"])
            self.assertEqual(summary["seed_state_restored_records"], 1)
            second = summary["events"][1]
            self.assertNotEqual(
                second["seed_state_before_reset_fingerprint"],
                second["seed_state_baseline_fingerprint"],
            )
            self.assertEqual(
                second["seed_state_after_reset_fingerprint"],
                second["seed_state_baseline_fingerprint"],
            )

    def test_persistent_horizon_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(directory, "persistent")
            seed = _record("seed", source_role="seed")
            dynamic = _record("dynamic", source_role="opponent")
            memory.arguments = {seed.id: seed, dynamic.id: dynamic}
            memory.belief_engine.recompute_from_active_records(
                memory.arguments.values(),
                persist=False,
            )
            stance_before = memory.get_stance(verbose=False)

            event_index = memory.prepare_for_opponent_evidence_turn(
                round_index=2,
                sender_name="Con",
            )
            memory.finalize_opponent_evidence_turn(
                event_index,
                extracted_records=0,
                accepted_records=0,
                status_counts={},
            )

            self.assertEqual(set(memory.arguments), {"seed", "dynamic"})
            self.assertEqual(memory.get_stance(verbose=False), stance_before)
            summary = memory.evidence_horizon_audit_summary()
            self.assertEqual(summary["reset_calls"], 0)
            self.assertEqual(summary["evicted_records"], 0)

    def test_latest_turn_bound_counts_ignored_and_replaced_records(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(directory, "latest_opponent_turn")
            memory.arguments = {
                "seed": _record("seed", source_role="seed"),
                "stale": _record("stale", source_role="opponent"),
            }
            memory.belief_engine.recompute_from_active_records(
                memory.arguments.values(),
                persist=False,
            )
            ignored = _record("ignored", source_role="opponent")
            replaced = _record("replaced", source_role="opponent")

            def process(argument, _arguments, _backend):
                if argument.id == "ignored":
                    argument.credence_relevant = False
                    return "ignored", None, "duplicate", argument
                argument.credence_relevant = True
                return "replaced", "old", "stronger", argument

            event_index = memory.prepare_for_opponent_evidence_turn(
                round_index=3,
                sender_name="Con",
            )
            with patch.object(
                memory.judgement_strategy,
                "process",
                side_effect=process,
            ):
                results = memory.add_arguments_batch([ignored, replaced])
            statuses = [result[0] for result in results]
            memory.finalize_opponent_evidence_turn(
                event_index,
                extracted_records=2,
                accepted_records=sum(
                    status != "ignored" for status in statuses
                ),
                status_counts={
                    status: statuses.count(status) for status in set(statuses)
                },
            )

            event = memory.evidence_horizon_audit_summary()["events"][0]
            self.assertEqual(event["after_batch"]["dynamic"], 2)
            self.assertEqual(event["extracted_records"], 2)
            self.assertEqual(event["accepted_records"], 1)
            self.assertEqual(
                event["status_counts"],
                {"ignored": 1, "replaced": 1},
            )
            self.assertTrue(event["latest_turn_bound_ok"])

    def test_valid_empty_extraction_prepares_receiver_but_never_sender(self):
        events = []
        agents = [
            _TrackingAgent("Pro", events),
            _TrackingAgent("Con", events),
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            result = SimulationRunner(
                agents,
                rounds=1,
                extractor=_EmptyExtractor(),
                output_dir=output_dir,
                artifact_mode="compact",
                extraction_strict=True,
                memory_strict=True,
            ).run()

        prepares = [event for event in events if event[0] == "prepare"]
        self.assertEqual(
            prepares,
            [
                ("prepare", "Con", "Pro", 1),
                ("prepare", "Pro", "Con", 1),
            ],
        )
        self.assertTrue(all(receiver != sender for _, receiver, sender, _ in prepares))
        audit = result["metadata"]["dynamic_evidence_horizon"]
        self.assertEqual(audit["totals"]["prepare_calls"], 2)
        self.assertEqual(audit["totals"]["completed_turns"], 2)
        self.assertEqual(audit["totals"]["aborted_turns"], 0)
        self.assertEqual(audit["totals"]["open_turns"], 0)

    def test_strict_ingestion_failure_aborts_all_prepared_horizon_events(self):
        events = []
        pro = _TrackingAgent("Pro", events)
        pro.memory = None
        con = _FailingTrackingAgent("Con", events)
        with tempfile.TemporaryDirectory() as output_dir:
            runner = SimulationRunner(
                [pro, con],
                rounds=1,
                extractor=_RecordExtractor(),
                output_dir=output_dir,
                artifact_mode="compact",
                extraction_strict=True,
                memory_strict=True,
            )
            with self.assertRaisesRegex(OSError, "strict ingestion failed"):
                runner.run()

        summary = con.memory.evidence_horizon_audit_summary()
        self.assertEqual(summary["prepare_calls"], 1)
        self.assertEqual(summary["completed_turns"], 0)
        self.assertEqual(summary["aborted_turns"], 1)
        self.assertEqual(summary["open_turns"], 0)

    def test_factory_threads_both_ablation_controls(self):
        global_config = {
            "backend": {"type": "mock", "model": "mock"},
            "debate": {"topic": "Test proposition", "seeding_limit": 0},
            "generation": {
                "temperature": 0.0,
                "response_shell": "matched_v1",
            },
            "memory": {
                "storage_mode": "episodic",
                "stance_method": "bayesian",
                "belief_engine_type": "bayesian",
                "belief_semantics_mode": "active_explicit_prior_v1",
                "response_retrieval_enabled": False,
                "dynamic_evidence_horizon": "latest_opponent_turn",
            },
        }
        with tempfile.TemporaryDirectory() as log_dir:
            agent = create_agent(
                {
                    "type": "subject",
                    "name": "Pro",
                    "persona": "Test persona",
                    "target_initial_stance": 0.75,
                },
                global_config,
                backend=_Backend(),
                log_dir=log_dir,
            )

        self.assertFalse(agent.response_retrieval_enabled)
        self.assertEqual(
            agent.memory.dynamic_evidence_horizon,
            "latest_opponent_turn",
        )

    def test_comparison_config_serializes_ablation_controls(self):
        condition = comparison.Condition(
            name="latest",
            label="Latest",
            backend_type="openai",
            model="gpt-5.4-mini",
            use_belief_engine=True,
            note="test",
            memory_overrides={
                **comparison.OPEN_MINDED_MEMORY,
                "response_retrieval_enabled": False,
                "dynamic_evidence_horizon": "latest_opponent_turn",
            },
            backend_config={"reasoning_effort": "none", "verbosity": "low"},
        )
        spec = comparison.build_specs(
            [condition],
            trials=1,
            topics=["We should adopt atheism"],
        )[0]
        config = comparison.build_config(spec, _matched_args())

        for container in (
            config["memory"],
            config["simulation"]["agents"][0]["memory"],
            config["validation_metadata"],
            config["validation_metadata"]["effective_memory"],
        ):
            self.assertFalse(container["response_retrieval_enabled"])
            self.assertEqual(
                container["dynamic_evidence_horizon"],
                "latest_opponent_turn",
            )


if __name__ == "__main__":
    unittest.main()
