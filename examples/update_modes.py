"""Compare replacement behavior using synthetic records and no model calls."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.argument_record import ArgumentRecord
from src.simulation.belief_engine import BayesianBeliefEngine


def main():
    for mode in ("incremental_v1", "active_seed_v1", "active_explicit_prior_v1"):
        with tempfile.TemporaryDirectory() as directory:
            engine = BayesianBeliefEngine("example", directory, {
                "belief_semantics_mode": mode,
                "update_sensitivity": .4, "anchor_sensitivity": .8,
                "initial_stance": .5 if mode == "active_explicit_prior_v1" else 0,
            })
            old = ArgumentRecord("Example proposition", "First claim", "affirmative",
                                 "opponent", .2, id="old", credence_relevant=True)
            new = ArgumentRecord("Example proposition", "Stronger replacement", "affirmative",
                                 "opponent", .8, id="new", credence_relevant=True)
            if mode == "incremental_v1":
                engine.update(old)
            else:
                engine.recompute_from_active_records([old])
            before = engine.get_belief_score()
            old.credence_relevant = False
            # Historical MemoryStore replacement leaves accumulated state alone.
            if mode != "incremental_v1":
                engine.recompute_from_active_records([old, new])
            print(f"{mode}: {before:.6f} -> {engine.get_belief_score():.6f}")


if __name__ == "__main__":
    main()
