"""Regression tests for historical and active-record belief update modes."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.factory import create_agent
from src.constants import POLARITY_AFFIRMATIVE
from src.constants import POLARITY_NEGATIVE
from src.memory.argument_record import ArgumentRecord
from src.memory.memory_store import MemoryStore
from src.simulation.belief_engine import (
    BELIEF_SEMANTICS_ACTIVE_EXPLICIT_PRIOR_V1,
    BELIEF_SEMANTICS_ACTIVE_SEED_V1,
    BELIEF_SEMANTICS_INCREMENTAL_V1,
)


def _argument(
    *,
    argument_id: str,
    strength: float,
    polarity: str = POLARITY_AFFIRMATIVE,
    source_role: str = "opponent",
) -> ArgumentRecord:
    return ArgumentRecord(
        id=argument_id,
        topic_id="Test proposition",
        claim=f"Claim {argument_id}",
        polarity=polarity,
        source_role=source_role,
        support_strength=strength,
    )


def _stance_from_log_odds(log_odds: float) -> float:
    return (2.0 / (1.0 + math.exp(-log_odds))) - 1.0


def _active_set_log_odds(memory: MemoryStore) -> float:
    """Apply the manuscript equation to the store's currently active records."""
    total = 0.0
    for argument in memory.arguments.values():
        if not argument.credence_relevant:
            continue

        sensitivity = (
            memory.belief_engine.anchor_sensitivity
            if argument.source_role == "seed"
            else memory.belief_engine.sensitivity
        )
        sign = 1.0 if argument.polarity == POLARITY_AFFIRMATIVE else -1.0
        total += sign * math.log1p(argument.support_strength * sensitivity)
    return total


def _memory(
    directory: str,
    *,
    initial_stance: float = 0.0,
    belief_semantics_mode: str = BELIEF_SEMANTICS_INCREMENTAL_V1,
    confirmation_bias: float = 0.0,
    stance_method: str = "bayesian",
    belief_engine_type: str = "bayesian",
    storage_mode: str = "episodic",
) -> MemoryStore:
    return MemoryStore(
        storage_mode=storage_mode,
        retrieval_strategy="relevant",
        worldview_openness="balanced",
        stance_method=stance_method,
        update_sensitivity=1.0,
        anchor_sensitivity=0.5,
        confirmation_bias=confirmation_bias,
        belief_engine_type=belief_engine_type,
        belief_semantics_mode=belief_semantics_mode,
        initial_stance=initial_stance,
        memory_dir=directory,
        agent_id="semantic-test-agent",
    )


def _add_one_seed(memory, topic, side, config):
    memory.add_argument(
        _argument(
            argument_id="seed",
            strength=0.8,
            source_role="seed",
        )
    )


def _subject_factory_inputs(
    belief_semantics_mode: str = BELIEF_SEMANTICS_INCREMENTAL_V1,
):
    global_config = {
        "backend": {"type": "test", "model": "none"},
        "debate": {"topic": "Test proposition"},
        "generation": {"temperature": 0.0},
        "memory": {
            "storage_mode": "episodic",
            "retrieval_strategy": "relevant",
            "worldview_openness": "balanced",
            "stance_method": "bayesian",
            "update_sensitivity": 1.0,
            "anchor_sensitivity": 0.5,
            "confirmation_bias": 0.0,
            "belief_semantics_mode": belief_semantics_mode,
        },
    }
    agent_config = {
        "type": "subject",
        "name": "Initialized agent",
        "persona": "Test persona",
        "seed_stance": "pro",
        "target_initial_stance": 0.75,
    }
    return agent_config, global_config


class ReviewerReleaseBeliefSemanticsTests(unittest.TestCase):

    def test_memory_store_initial_stance_is_metadata_not_engine_initialization(self):
        """Direct MemoryStore construction leaves the latent engine neutral."""
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(directory, initial_stance=0.75)

            self.assertEqual(memory.initial_stance, 0.75)
            self.assertEqual(memory.belief_engine.log_odds, 0.0)
            self.assertEqual(memory.get_stance(verbose=False), 0.0)

    def test_replacement_archives_old_record_without_changing_latent_state(self):
        """Characterize the incremental replacement behavior in the release."""
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(directory)
            old = _argument(argument_id="old", strength=0.2)
            replacement = _argument(argument_id="replacement", strength=0.8)

            old_status, *_ = memory.add_argument(old)
            log_odds_after_old = memory.belief_engine.log_odds
            with patch.object(
                memory.judgement_strategy.credence_manager,
                "find_similar_arguments",
                return_value=[(old, 0.99)],
            ):
                replacement_status, *_ = memory.add_argument(replacement)

            self.assertEqual(old_status, "added")
            self.assertEqual(replacement_status, "replaced")
            self.assertFalse(old.credence_relevant)
            self.assertTrue(replacement.credence_relevant)

            # Reviewer-release semantics: no subtract-old or add-new operation
            # occurs on the replacement branch.
            self.assertAlmostEqual(log_odds_after_old, math.log1p(0.2))
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                log_odds_after_old,
            )
            self.assertAlmostEqual(
                memory.get_stance(verbose=False),
                _stance_from_log_odds(log_odds_after_old),
            )
            self.assertEqual(memory.belief_engine.history_count, 1)

            # The active record would imply a different state under Eq. 1.
            self.assertAlmostEqual(_active_set_log_odds(memory), math.log1p(0.8))
            self.assertNotAlmostEqual(
                memory.belief_engine.log_odds,
                _active_set_log_odds(memory),
            )

    def test_incremental_state_file_keeps_reviewer_release_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(directory)
            memory.add_argument(_argument(argument_id="one", strength=0.2))

            with Path(memory.belief_engine.state_file).open() as handle:
                state = json.load(handle)

            self.assertEqual(set(state), {"log_odds", "history_count"})

    def test_incremental_replacement_is_not_active_set_recomputation(self):
        """Historical replacement retains the previously accumulated contribution."""
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(directory)
            old = _argument(argument_id="old", strength=0.2)
            replacement = _argument(argument_id="replacement", strength=0.8)
            memory.add_argument(old)
            with patch.object(
                memory.judgement_strategy.credence_manager,
                "find_similar_arguments",
                return_value=[(old, 0.99)],
            ):
                memory.add_argument(replacement)

            self.assertNotAlmostEqual(
                memory.belief_engine.log_odds,
                _active_set_log_odds(memory),
            )


class ActiveSetBeliefSemanticsTests(unittest.TestCase):
    def test_active_seed_add_ignore_and_replace_recompute_current_records(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(
                directory,
                belief_semantics_mode=BELIEF_SEMANTICS_ACTIVE_SEED_V1,
            )
            old = _argument(argument_id="old", strength=0.2)

            added_status, *_ = memory.add_argument(old)
            self.assertEqual(added_status, "added")
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                math.log1p(0.2),
            )
            self.assertEqual(memory.belief_engine.history_count, 1)

            ignored = _argument(argument_id="ignored", strength=0.1)
            with patch.object(
                memory.judgement_strategy.credence_manager,
                "find_similar_arguments",
                return_value=[(old, 0.99)],
            ):
                ignored_status, *_ = memory.add_argument(ignored)

            self.assertEqual(ignored_status, "ignored")
            self.assertFalse(ignored.credence_relevant)
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                math.log1p(0.2),
            )
            self.assertEqual(memory.belief_engine.history_count, 1)

            replacement = _argument(argument_id="replacement", strength=0.8)
            with patch.object(
                memory.judgement_strategy.credence_manager,
                "find_similar_arguments",
                return_value=[(old, 0.99)],
            ):
                replacement_status, *_ = memory.add_argument(replacement)

            self.assertEqual(replacement_status, "replaced")
            self.assertFalse(old.credence_relevant)
            self.assertTrue(replacement.credence_relevant)
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                math.log1p(0.8),
            )
            self.assertEqual(memory.belief_engine.history_count, 2)

            opposing = _argument(
                argument_id="opposing",
                strength=0.5,
                polarity=POLARITY_NEGATIVE,
            )
            opposing_status, *_ = memory.add_argument(opposing)
            self.assertEqual(opposing_status, "added")
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                math.log1p(0.8) - math.log1p(0.5),
            )
            self.assertEqual(memory.belief_engine.history_count, 3)

    def test_active_seed_uses_anchor_sensitivity_for_active_seed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(
                directory,
                belief_semantics_mode=BELIEF_SEMANTICS_ACTIVE_SEED_V1,
            )
            seed = _argument(
                argument_id="seed",
                strength=0.8,
                source_role="seed",
            )

            memory.add_argument(seed)

            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                math.log1p(0.8 * 0.5),
            )
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                _active_set_log_odds(memory),
            )

    def test_explicit_prior_excludes_active_seeds_but_includes_later_evidence(self):
        prior_stance = 0.5
        prior_log_odds = math.log(
            (1.0 + prior_stance) / (1.0 - prior_stance)
        )
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(
                directory,
                initial_stance=prior_stance,
                belief_semantics_mode=(
                    BELIEF_SEMANTICS_ACTIVE_EXPLICIT_PRIOR_V1
                ),
            )
            seed = _argument(
                argument_id="seed",
                strength=0.8,
                source_role="seed",
            )

            memory.add_argument(seed)
            self.assertTrue(seed.credence_relevant)
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                prior_log_odds,
            )

            later = _argument(argument_id="later", strength=0.2)
            with patch.object(
                memory.judgement_strategy.credence_manager,
                "find_similar_arguments",
                return_value=[],
            ):
                memory.add_argument(later)

            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                prior_log_odds + math.log1p(0.2),
            )

    def test_active_mode_state_file_records_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(
                directory,
                belief_semantics_mode=BELIEF_SEMANTICS_ACTIVE_SEED_V1,
            )
            memory.add_argument(_argument(argument_id="one", strength=0.2))

            with Path(memory.belief_engine.state_file).open() as handle:
                state = json.load(handle)

            self.assertEqual(
                state["belief_semantics_mode"],
                BELIEF_SEMANTICS_ACTIVE_SEED_V1,
            )
            self.assertIn("explicit_prior_log_odds", state)

    def test_active_mode_refuses_state_from_different_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            incremental = _memory(directory)
            incremental.add_argument(
                _argument(argument_id="one", strength=0.2)
            )

            with self.assertRaisesRegex(
                ValueError,
                "semantics mismatch",
            ):
                _memory(
                    directory,
                    belief_semantics_mode=BELIEF_SEMANTICS_ACTIVE_SEED_V1,
                )

    def test_active_modes_reject_path_dependent_or_non_bayesian_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "Unknown belief_semantics_mode",
            ):
                _memory(
                    directory,
                    belief_semantics_mode="not-a-real-mode",
                )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "confirmation_bias=0"):
                _memory(
                    directory,
                    belief_semantics_mode=BELIEF_SEMANTICS_ACTIVE_SEED_V1,
                    confirmation_bias=0.1,
                )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "stance_method='bayesian'"):
                _memory(
                    directory,
                    belief_semantics_mode=BELIEF_SEMANTICS_ACTIVE_SEED_V1,
                    stance_method="linear",
                )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "belief_engine_type='bayesian'",
            ):
                _memory(
                    directory,
                    belief_semantics_mode=BELIEF_SEMANTICS_ACTIVE_SEED_V1,
                    belief_engine_type="agentic",
                )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "AgenticMemoryStore"):
                _memory(
                    directory,
                    belief_semantics_mode=BELIEF_SEMANTICS_ACTIVE_SEED_V1,
                    storage_mode="agentic",
                )

    def test_active_mode_rejects_incremental_update_api(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = _memory(
                directory,
                belief_semantics_mode=BELIEF_SEMANTICS_ACTIVE_SEED_V1,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "recompute_from_active_records",
            ):
                memory.belief_engine.update(
                    _argument(argument_id="one", strength=0.2)
                )


class SubjectFactoryInitializationTests(unittest.TestCase):
    def test_target_initial_stance_is_applied_after_seed_updates(self):
        """The factory overwrites seed-derived log odds with the target score."""
        target_stance = 0.75

        def add_one_seed(memory, topic, side, config):
            memory.add_argument(
                _argument(
                    argument_id="seed",
                    strength=0.8,
                    source_role="seed",
                )
            )

        global_config = {
            "backend": {"type": "test", "model": "none"},
            "debate": {"topic": "Test proposition"},
            "generation": {"temperature": 0.0},
            "memory": {
                "storage_mode": "episodic",
                "retrieval_strategy": "relevant",
                "worldview_openness": "balanced",
                "stance_method": "bayesian",
                "update_sensitivity": 1.0,
                "anchor_sensitivity": 0.5,
                "confirmation_bias": 0.0,
            },
        }
        agent_config = {
            "type": "subject",
            "name": "Initialized agent",
            "persona": "Test persona",
            "seed_stance": "pro",
            "target_initial_stance": target_stance,
        }

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "src.simulation.seeding_helper.seed_review_artifactory",
                side_effect=add_one_seed,
            ):
                agent = create_agent(
                    agent_config,
                    global_config,
                    backend=None,
                    log_dir=directory,
                )

            memory = agent.memory
            self.assertEqual(memory.belief_engine.history_count, 1)
            self.assertAlmostEqual(memory.get_stance(verbose=False), target_stance)
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                math.log((1.0 + target_stance) / (1.0 - target_stance)),
            )
            self.assertTrue(Path(memory.belief_engine.state_file).exists())

            # The seed remains active, but its active-set contribution is not
            # the latent score after the factory's target override.
            self.assertEqual(
                [argument.id for argument in memory.arguments.values()],
                ["seed"],
            )
            self.assertAlmostEqual(_active_set_log_odds(memory), math.log1p(0.4))
            self.assertNotAlmostEqual(
                memory.belief_engine.log_odds,
                _active_set_log_odds(memory),
            )

    def test_active_seed_factory_initialization_uses_seed_set_not_target_override(self):
        agent_config, global_config = _subject_factory_inputs(
            BELIEF_SEMANTICS_ACTIVE_SEED_V1
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "src.simulation.seeding_helper.seed_review_artifactory",
                side_effect=_add_one_seed,
            ):
                agent = create_agent(
                    agent_config,
                    global_config,
                    backend=None,
                    log_dir=directory,
                )

            memory = agent.memory
            self.assertEqual(memory.initial_stance, 0.75)
            self.assertEqual(
                memory.belief_engine.semantics_mode,
                BELIEF_SEMANTICS_ACTIVE_SEED_V1,
            )
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                math.log1p(0.8 * 0.5),
            )
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                _active_set_log_odds(memory),
            )
            self.assertNotAlmostEqual(
                memory.get_stance(verbose=False),
                0.75,
            )

    def test_explicit_prior_factory_initialization_does_not_double_count_seed(self):
        agent_config, global_config = _subject_factory_inputs(
            BELIEF_SEMANTICS_ACTIVE_EXPLICIT_PRIOR_V1
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "src.simulation.seeding_helper.seed_review_artifactory",
                side_effect=_add_one_seed,
            ):
                agent = create_agent(
                    agent_config,
                    global_config,
                    backend=None,
                    log_dir=directory,
                )

            memory = agent.memory
            prior_log_odds = math.log((1.0 + 0.75) / (1.0 - 0.75))
            self.assertEqual(
                memory.belief_engine.semantics_mode,
                BELIEF_SEMANTICS_ACTIVE_EXPLICIT_PRIOR_V1,
            )
            self.assertAlmostEqual(memory.get_stance(verbose=False), 0.75)
            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                prior_log_odds,
            )
            self.assertTrue(memory.arguments["seed"].credence_relevant)

            with patch.object(
                memory.judgement_strategy.credence_manager,
                "find_similar_arguments",
                return_value=[],
            ):
                memory.add_argument(
                    _argument(argument_id="later", strength=0.2)
                )

            self.assertAlmostEqual(
                memory.belief_engine.log_odds,
                prior_log_odds + math.log1p(0.2),
            )

    def test_incremental_target_override_is_not_seed_recomputation(self):
        """A target override also diverges from a seed-only active-set state."""
        target_stance = 0.75

        def add_one_seed(memory, topic, side, config):
            memory.add_argument(
                _argument(
                    argument_id="seed",
                    strength=0.8,
                    source_role="seed",
                )
            )

        global_config = {
            "backend": {"type": "test", "model": "none"},
            "debate": {"topic": "Test proposition"},
            "memory": {
                "storage_mode": "episodic",
                "retrieval_strategy": "relevant",
                "stance_method": "bayesian",
                "update_sensitivity": 1.0,
                "anchor_sensitivity": 0.5,
            },
        }
        agent_config = {
            "type": "subject",
            "name": "Initialized agent",
            "persona": "Test persona",
            "seed_stance": "pro",
            "target_initial_stance": target_stance,
        }

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "src.simulation.seeding_helper.seed_review_artifactory",
                side_effect=add_one_seed,
            ):
                agent = create_agent(
                    agent_config,
                    global_config,
                    backend=None,
                    log_dir=directory,
                )

            self.assertNotAlmostEqual(
                agent.memory.belief_engine.log_odds,
                _active_set_log_odds(agent.memory),
            )


if __name__ == "__main__":
    unittest.main()
