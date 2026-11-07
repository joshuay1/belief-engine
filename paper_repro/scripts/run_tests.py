"""Run behavioral regression tests with outbound network connections blocked."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def deny_network(*args, **kwargs):
    raise RuntimeError("Behavioral tests must not open network connections")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", action="store_true",
                        help="Run classifier integration tests (requires the classifier extra).")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # A small real BPE encoding exercises truncation/counting without fetching
    # cl100k_base. This fixture is only used by unit tests, never experiments.
    test_encoding = tiktoken.Encoding(
        name="offline_test_bytes", pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={},
    )
    # Unit tests use scripted backends and synthetic tensors, never hosted LLMs
    # or downloaded checkpoints. Block accidental requests as well as HF fetches.
    with patch.object(socket.socket, "connect", deny_network), \
         patch.object(socket, "create_connection", deny_network), \
         patch.object(tiktoken, "get_encoding", return_value=test_encoding):
        suite = unittest.TestSuite()
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            is_classifier = path.name == "test_classifier_integration.py"
            if is_classifier == args.classifier:
                suite.addTests(unittest.defaultTestLoader.discover(
                    str(ROOT / "tests"), pattern=path.name,
                ))
        result = unittest.TextTestRunner(verbosity=2, buffer=True).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
