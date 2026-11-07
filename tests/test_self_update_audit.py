from __future__ import annotations

import tempfile
import unittest

from src.agents.self_updating_agent import SelfUpdatingSubjectAgent
from src.simulation.simulation_runner import SimulationRunner


class _SequencedBackend:
    def __init__(self, responses):
        self.responses = list(responses)
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
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _Memory:
    def __init__(self, retrieved=None, error=None):
        self.retrieved = [] if retrieved is None else retrieved
        self.error = error
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.retrieved


def _agent(backend, name="Pro", stance=0.75, memory=None):
    return SelfUpdatingSubjectAgent(
        backend=backend,
        persona="A concise test persona.",
        name=name,
        topic="Test proposition",
        initial_stance=stance,
        memory=memory,
        response_shell="matched_v1",
    )


def _opponent_message(name="Con"):
    return {
        "role": "user",
        "agent_name": name,
        "content": "A test counterargument.",
    }


class SelfUpdateAuditTests(unittest.TestCase):
    def test_valid_and_invalid_updates_preserve_legacy_stance_behavior(self):
        backend = _SequencedBackend(
            [
                '{"stance": -0.25, "confidence": 0.8, "reason": "persuasive"}',
                "not json",
            ]
        )
        agent = _agent(backend)
        message = _opponent_message()

        agent.observe_message(message, [message])
        self.assertEqual(agent.get_stance(), -0.25)
        self.assertEqual(agent.last_update_reason, "persuasive")

        agent.observe_message(message, [message])
        self.assertEqual(agent.get_stance(), -0.25)
        self.assertEqual(agent.last_update_reason, "self-reported update")
        self.assertEqual(
            agent.self_update_audit_summary(),
            {
                "agent_type": "self_update",
                "uses_memory": False,
                "self_update_calls": 2,
                "self_update_json_parse_successes": 1,
                "self_update_json_parse_failures": 1,
                "self_update_api_errors": 0,
                "self_update_stance_changed": 1,
                "self_update_stance_unchanged": 1,
                "retrieval_calls": 0,
                "retrieval_nonempty": 0,
                "retrieval_empty": 0,
                "retrieval_errors": 0,
            },
        )

    def test_update_api_error_is_counted_and_reraised(self):
        agent = _agent(_SequencedBackend([RuntimeError("backend failed")]))
        message = _opponent_message()

        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            agent.observe_message(message, [message])

        summary = agent.self_update_audit_summary()
        self.assertEqual(agent.get_stance(), 0.75)
        self.assertEqual(summary["self_update_calls"], 1)
        self.assertEqual(summary["self_update_api_errors"], 1)
        self.assertEqual(summary["self_update_json_parse_successes"], 0)
        self.assertEqual(summary["self_update_json_parse_failures"], 0)
        self.assertEqual(summary["self_update_stance_changed"], 0)
        self.assertEqual(summary["self_update_stance_unchanged"], 0)

    def test_retrieval_results_and_errors_are_unchanged_and_counted(self):
        records = [object()]
        successful_memory = _Memory(retrieved=records)
        successful_agent = _agent(
            _SequencedBackend(["unchanged generated reply"]),
            memory=successful_memory,
        )

        response, retrieved = successful_agent.generate_reply([])

        self.assertEqual(response, "unchanged generated reply")
        self.assertIs(retrieved, records)
        self.assertEqual(successful_memory.calls[0]["k"], 10)
        successful_summary = successful_agent.self_update_audit_summary()
        self.assertEqual(successful_summary["retrieval_calls"], 1)
        self.assertEqual(successful_summary["retrieval_nonempty"], 1)
        self.assertEqual(successful_summary["retrieval_empty"], 0)
        self.assertEqual(successful_summary["retrieval_errors"], 0)

        failing_agent = _agent(
            _SequencedBackend([]),
            memory=_Memory(error=OSError("retrieval failed")),
        )
        self.assertEqual(failing_agent._retrieved_arguments("query"), [])
        failing_summary = failing_agent.self_update_audit_summary()
        self.assertEqual(failing_summary["retrieval_calls"], 1)
        self.assertEqual(failing_summary["retrieval_nonempty"], 0)
        self.assertEqual(failing_summary["retrieval_empty"], 0)
        self.assertEqual(failing_summary["retrieval_errors"], 1)

    def test_runner_exposes_per_agent_and_total_audits(self):
        pro = _agent(
            _SequencedBackend(
                [
                    "Pro debate reply.",
                    '{"stance": 0.5, "confidence": 0.8, "reason": "updated"}',
                ]
            ),
            name="Pro",
        )
        con = _agent(
            _SequencedBackend(
                [
                    '{"stance": -0.5, "confidence": 0.8, "reason": "updated"}',
                    "Con debate reply.",
                ]
            ),
            name="Con",
            stance=-0.75,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            result = SimulationRunner(
                [pro, con],
                rounds=1,
                output_dir=output_dir,
                artifact_mode="compact",
            ).run()

        audit = result["metadata"]["self_update_baseline"]
        self.assertEqual(set(audit["agents"]), {"Pro", "Con"})
        self.assertEqual(audit["agents"]["Pro"]["self_update_calls"], 1)
        self.assertEqual(audit["agents"]["Con"]["self_update_calls"], 1)
        self.assertEqual(audit["totals"]["self_update_calls"], 2)
        self.assertEqual(audit["totals"]["self_update_json_parse_successes"], 2)
        self.assertEqual(audit["totals"]["self_update_stance_changed"], 2)


if __name__ == "__main__":
    unittest.main()
