from __future__ import annotations

import tempfile
import unittest

from src.memory.argument_record import ArgumentRecord
from src.memory.memory_store import MemoryStore


def record(claim: str) -> ArgumentRecord:
    return ArgumentRecord(
        topic_id="Test proposition",
        claim=claim,
        polarity="affirmative",
        source_role="opponent",
        support_strength=0.8,
    )


class NoDedupAblationTests(unittest.TestCase):
    def test_disabled_gate_preserves_exact_repeated_occurrences(self):
        with tempfile.TemporaryDirectory() as memory_dir:
            memory = MemoryStore(
                storage_mode="episodic",
                stance_method="bayesian",
                update_sensitivity=0.2,
                anchor_sensitivity=0.4,
                allow_self_arguments=False,
                belief_engine_type="bayesian",
                belief_semantics_mode="active_explicit_prior_v1",
                deduplication_enabled=False,
                initial_stance=0.0,
                memory_dir=memory_dir,
            )
            first = record("The exact same evidence.")
            second = record("The exact same evidence.")

            first_status = memory.add_argument(first)[0]
            second_status = memory.add_argument(second)[0]

            self.assertEqual((first_status, second_status), ("added", "added"))
            self.assertEqual(len(memory.arguments), 2)
            self.assertNotEqual(first.id, second.id)
            self.assertTrue(
                all(item.credence_relevant for item in memory.arguments.values())
            )
            audit = memory.deduplication_audit_summary()
            self.assertFalse(audit["deduplication_enabled"])
            self.assertEqual(audit["status_counts"], {"added": 2})
            self.assertEqual(
                audit["dynamic_status_counts"],
                {"added": 2},
            )
            self.assertEqual(audit["seed_status_counts"], {})
            self.assertEqual(audit["no_dedup_id_collisions_preserved"], 1)
            self.assertEqual(audit["final_dynamic_records_total"], 2)
            self.assertEqual(audit["final_dynamic_active_records"], 2)
            self.assertEqual(audit["final_dynamic_archived_records"], 0)
            self.assertEqual(len(audit["final_dynamic_state_sha256"]), 64)
            self.assertEqual(
                len(audit["final_dynamic_active_state_sha256"]),
                64,
            )
            self.assertEqual(
                len(audit["final_dynamic_canonical_state_sha256"]),
                64,
            )
            self.assertEqual(
                len(
                    audit[
                        "final_dynamic_active_canonical_state_sha256"
                    ]
                ),
                64,
            )


if __name__ == "__main__":
    unittest.main()
