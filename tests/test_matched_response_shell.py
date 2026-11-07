from __future__ import annotations

import tempfile
import unittest

from src.agents.factory import create_agent
from src.agents.self_updating_agent import SelfUpdatingSubjectAgent
from src.agents.shared_instructions import (
    LEGACY_RESPONSE_SHELL,
    MATCHED_RESPONSE_SHELL,
    build_matched_response_call,
)
from src.agents.subject_agent import SubjectAgent
from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE
from src.memory.argument_record import ArgumentRecord


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def chat(self, messages, system_prompt=None, temperature=0.7, max_tokens=1000):
        self.calls.append(
            {
                "messages": messages,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return "Recorded response."

    @property
    def last_call(self):
        return self.calls[-1]


class StubBeliefEngine:
    def get_belief_description(self):
        raise AssertionError("matched_v1 must not use mechanism-specific guidance")

    def get_explicit_stance_instruction(self, topic_placeholder="the proposition"):
        raise AssertionError("matched_v1 must not use mechanism-specific guidance")


class RecordingMemory:
    def __init__(self, stance, arguments):
        self.stance = stance
        self.arguments = {argument.id: argument for argument in arguments}
        self.stance_method = "bayesian"
        self.belief_engine = StubBeliefEngine()
        self.retrieve_calls = []

    def retrieve(self, k=3, **kwargs):
        self.retrieve_calls.append({"k": k, **kwargs})
        return list(self.arguments.values())[:k]

    def get_stance(self, verbose=False):
        del verbose
        return self.stance


def argument(argument_id, polarity, strength):
    record = ArgumentRecord(
        id=argument_id,
        topic_id="Test proposition",
        claim=f"Evidence {argument_id}",
        polarity=polarity,
        source_role="opponent",
        support_strength=strength,
    )
    record.credence_relevant = True
    return record


def six_message_history():
    return [
        {"role": "assistant", "agent_name": "Debater", "content": "old-self-1"},
        {"role": "user", "agent_name": "Opponent", "content": "old-opponent-2"},
        {"role": "assistant", "agent_name": "Debater", "content": "recent-self-3"},
        {"role": "user", "agent_name": "Opponent", "content": "recent-opponent-4"},
        {"role": "assistant", "agent_name": "Debater", "content": "recent-self-5"},
        {"role": "user", "agent_name": "Opponent", "content": "recent-opponent-6"},
    ]


class MatchedResponseShellTests(unittest.TestCase):
    def test_factory_threads_matched_shell_and_context_controls(self):
        global_config = {
            "backend": {"type": "mock", "model": "mock"},
            "debate": {
                "topic": "Test proposition",
                "seeding_limit": 0,
            },
            "generation": {
                "temperature": 0.3,
                "self_update_temperature": 0.0,
                "response_shell": MATCHED_RESPONSE_SHELL,
                "response_context_messages": 4,
                "response_word_limit": 99,
                "self_update_context_messages": 4,
            },
            "memory": {
                "storage_mode": "episodic",
                "stance_method": "bayesian",
                "belief_engine_type": "bayesian",
                "belief_semantics_mode": "active_explicit_prior_v1",
                "confirmation_bias": 0.0,
                "retrieval_strategy": "proportional",
                "retrieval_k": 7,
            },
        }
        with tempfile.TemporaryDirectory() as log_dir:
            subject = create_agent(
                {
                    "type": "subject",
                    "name": "Subject",
                    "persona": "A concise test persona.",
                    "target_initial_stance": 0.75,
                },
                global_config,
                backend=RecordingBackend(),
                log_dir=log_dir,
            )
            self_updater = create_agent(
                {
                    "type": "self_update",
                    "name": "Self updater",
                    "persona": "A concise test persona.",
                    "target_initial_stance": 0.75,
                    "use_memory": False,
                },
                global_config,
                backend=RecordingBackend(),
                log_dir=log_dir,
            )

            for agent in (subject, self_updater):
                self.assertEqual(agent.response_shell, MATCHED_RESPONSE_SHELL)
                self.assertEqual(agent.response_context_messages, 4)
                self.assertEqual(agent.response_word_limit, 99)
                self.assertEqual(agent.retrieval_k, 7)
            self.assertEqual(self_updater.update_context_messages, 4)

    def test_equal_inputs_produce_byte_identical_subject_and_self_update_calls(self):
        records = [
            argument("pro", POLARITY_AFFIRMATIVE, 0.75),
            argument("con", POLARITY_NEGATIVE, 0.65),
        ]
        subject_memory = RecordingMemory(stance=0.55, arguments=records)
        self_update_memory = RecordingMemory(stance=0.0, arguments=records)
        subject_backend = RecordingBackend()
        self_update_backend = RecordingBackend()

        subject = SubjectAgent(
            backend=subject_backend,
            memory=subject_memory,
            persona="A concise test persona.",
            name="Debater",
            temperature=0.3,
            retrieval_k=10,
            response_shell=MATCHED_RESPONSE_SHELL,
        )
        self_updater = SelfUpdatingSubjectAgent(
            backend=self_update_backend,
            persona="A concise test persona.",
            name="Debater",
            topic="Test proposition",
            initial_stance=0.55,
            memory=self_update_memory,
            retrieval_k=10,
            generation_temperature=0.3,
            response_shell=MATCHED_RESPONSE_SHELL,
        )

        subject.generate_reply(six_message_history())
        self_updater.generate_reply(six_message_history())

        self.assertEqual(
            subject_backend.last_call["system_prompt"],
            self_update_backend.last_call["system_prompt"],
        )
        self.assertEqual(
            subject_backend.last_call["messages"],
            self_update_backend.last_call["messages"],
        )
        self.assertEqual(subject_backend.last_call["temperature"], 0.3)
        self.assertEqual(self_update_backend.last_call["temperature"], 0.3)
        self.assertEqual(len(subject_memory.retrieve_calls), 1)
        self.assertEqual(len(self_update_memory.retrieve_calls), 1)
        self.assertEqual(subject_memory.retrieve_calls[0]["k"], 10)
        self.assertEqual(self_update_memory.retrieve_calls[0]["k"], 10)

    def test_response_retrieval_off_skips_memory_and_keeps_matched_shell(self):
        records = [argument("pro", POLARITY_AFFIRMATIVE, 0.75)]
        memory = RecordingMemory(stance=0.55, arguments=records)
        backend = RecordingBackend()
        subject = SubjectAgent(
            backend=backend,
            memory=memory,
            persona="A concise test persona.",
            name="Debater",
            temperature=0.3,
            retrieval_k=10,
            response_shell=MATCHED_RESPONSE_SHELL,
            response_retrieval_enabled=False,
        )

        _response, retrieved = subject.generate_reply(six_message_history())

        self.assertEqual(retrieved, [])
        self.assertEqual(memory.retrieve_calls, [])
        self.assertIn(
            "Retrieved Evidence:\nNo retrieved arguments are available.",
            backend.last_call["system_prompt"],
        )
        audit = subject.response_retrieval_audit_summary()
        events = audit.pop("events")
        self.assertEqual(
            audit,
            {
                "response_retrieval_enabled": False,
                "reply_calls": 1,
                "retrieval_calls": 0,
                "retrieval_skipped_calls": 1,
                "records_returned_total": 0,
                "records_returned_max": 0,
                "memory_context_tokens_total": 0,
                "memory_context_tokens_max": 0,
                "memory_context_budget_saturated_calls": 0,
            },
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["call_index"], 1)
        self.assertFalse(events[0]["retrieval_enabled"])
        self.assertEqual(events[0]["records_returned"], 0)
        for key in (
            "rendered_context_sha256",
            "retrieved_record_identity_sequence_sha256",
            "retrieved_record_identity_multiset_sha256",
            "retrieved_active_score_sequence_sha256",
        ):
            self.assertEqual(len(events[0][key]), 64)

    def test_retrieved_evidence_is_exactly_token_budgeted(self):
        long_record = ArgumentRecord(
            id="long",
            topic_id="Test proposition",
            claim=" ".join(["substantive"] * 300),
            polarity=POLARITY_AFFIRMATIVE,
            source_role="opponent",
            support_strength=0.75,
        )
        long_record.credence_relevant = True
        memory = RecordingMemory(stance=0.55, arguments=[long_record])
        backend = RecordingBackend()
        subject = SubjectAgent(
            backend=backend,
            memory=memory,
            persona="A concise test persona.",
            name="Debater",
            temperature=0.3,
            retrieval_k=10,
            response_shell=MATCHED_RESPONSE_SHELL,
            retrieved_evidence_token_budget=32,
            retrieved_evidence_encoding="cl100k_base",
        )

        subject.generate_reply(six_message_history())

        audit = subject.response_retrieval_audit_summary()
        self.assertEqual(audit["memory_context_tokens_total"], 32)
        self.assertEqual(audit["memory_context_tokens_max"], 32)
        self.assertEqual(
            audit["memory_context_budget_saturated_calls"],
            1,
        )
        self.assertEqual(len(audit["events"]), 1)
        self.assertEqual(audit["events"][0]["memory_context_tokens"], 32)
        self.assertTrue(
            audit["events"][0]["memory_context_budget_saturated"]
        )
        self.assertEqual(
            len(audit["events"][0]["rendered_context_sha256"]),
            64,
        )
        self.assertEqual(
            len(
                audit["events"][0][
                    "retrieved_record_identity_sequence_sha256"
                ]
            ),
            64,
        )
        self.assertEqual(
            len(
                audit["events"][0][
                    "retrieved_record_identity_multiset_sha256"
                ]
            ),
            64,
        )
        self.assertEqual(
            len(
                audit["events"][0][
                    "retrieved_active_score_sequence_sha256"
                ]
            ),
            64,
        )

    def test_matched_shell_contains_only_last_four_messages(self):
        system_prompt, messages = build_matched_response_call(
            name="Debater",
            persona="A concise test persona.",
            topic="Test proposition",
            stance_score=0.55,
            retrieved_arguments=[],
            conversation_history=six_message_history(),
        )

        self.assertNotIn("old-self-1", system_prompt)
        self.assertNotIn("old-opponent-2", system_prompt)
        for expected in (
            "recent-self-3",
            "recent-opponent-4",
            "recent-self-5",
            "recent-opponent-6",
        ):
            self.assertIn(expected, system_prompt)
        self.assertLess(
            system_prompt.index("recent-self-3"),
            system_prompt.index("recent-opponent-4"),
        )
        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": "The debate continues. You are Debater. Respond now.",
                }
            ],
        )

    def test_matched_generation_does_not_leak_update_provenance(self):
        system_prompt, _messages = build_matched_response_call(
            name="Debater",
            persona="A concise test persona.",
            topic="Test proposition",
            stance_score=-0.85,
            retrieved_arguments=[],
            conversation_history=six_message_history(),
        )

        self.assertIn("Near Certainty OPPOSE", system_prompt)
        self.assertNotIn("Bayesian", system_prompt)
        self.assertNotIn("self-reported", system_prompt)
        self.assertNotIn("Last stance update reason", system_prompt)

    def test_neutral_bin_does_not_map_zero_to_opposition(self):
        system_prompt, _messages = build_matched_response_call(
            name="Debater",
            persona="A concise test persona.",
            topic="Test proposition",
            stance_score=0.0,
            retrieved_arguments=[],
            conversation_history=[],
        )

        self.assertIn("Neutral NEUTRAL / UNCERTAIN", system_prompt)
        self.assertNotIn("Neutral OPPOSE", system_prompt)

    def test_prompt_only_matched_path_does_not_reuse_static_seed_context(self):
        backend = RecordingBackend()
        self_updater = SelfUpdatingSubjectAgent(
            backend=backend,
            persona="A concise test persona.",
            name="Debater",
            topic="Test proposition",
            initial_stance=0.55,
            seed_arguments=["STATIC-SEED-SHOULD-NOT-APPEAR"],
            memory=None,
            response_shell=MATCHED_RESPONSE_SHELL,
        )

        _response, retrieved = self_updater.generate_reply(six_message_history())

        self.assertEqual(retrieved, [])
        self.assertNotIn("STATIC-SEED-SHOULD-NOT-APPEAR", backend.last_call["system_prompt"])
        self.assertIn(
            "No retrieved arguments are available.",
            backend.last_call["system_prompt"],
        )

    def test_matched_self_update_call_uses_four_message_window(self):
        backend = RecordingBackend()
        self_updater = SelfUpdatingSubjectAgent(
            backend=backend,
            persona="A concise test persona.",
            name="Debater",
            topic="Test proposition",
            initial_stance=0.55,
            response_shell=MATCHED_RESPONSE_SHELL,
        )
        history = six_message_history()

        self_updater.observe_message(history[-1], history)

        update_prompt = backend.last_call["messages"][0]["content"]
        self.assertNotIn("old-self-1", update_prompt)
        self.assertNotIn("old-opponent-2", update_prompt)
        for expected in (
            "recent-self-3",
            "recent-opponent-4",
            "recent-self-5",
            "recent-opponent-6",
        ):
            self.assertIn(expected, update_prompt)

    def test_legacy_defaults_remain_selected_without_explicit_opt_in(self):
        records = [argument("pro", POLARITY_AFFIRMATIVE, 0.75)]
        subject = SubjectAgent(
            backend=RecordingBackend(),
            memory=RecordingMemory(stance=0.55, arguments=records),
            persona="A concise test persona.",
        )
        self_updater = SelfUpdatingSubjectAgent(
            backend=RecordingBackend(),
            persona="A concise test persona.",
            name="Debater",
            topic="Test proposition",
        )

        self.assertEqual(subject.response_shell, LEGACY_RESPONSE_SHELL)
        self.assertEqual(self_updater.response_shell, LEGACY_RESPONSE_SHELL)
        self.assertEqual(self_updater.update_context_messages, 6)


if __name__ == "__main__":
    unittest.main()
