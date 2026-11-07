from __future__ import annotations

import unittest

from src.simulation.simulation_runner import SimulationRunner


class _RecordingAgent:
    type = "subject"
    memory = None

    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.stance = 0.0

    def get_current_stance(self):
        return self.stance

    def generate_reply(self, transcript):
        del transcript
        self.events.append(("speak", self.name, self.stance))
        return f"{self.name} stance={self.stance}", {}

    def observe_message(self, message, transcript):
        del transcript
        sender = message["agent_name"]
        self.events.append(("observe", self.name, sender))
        if sender != self.name:
            self.stance += 1.0


class SimulationSpeakerOrderTests(unittest.TestCase):
    def run_pair(self, speaker_order=None):
        events = []
        agents = [
            _RecordingAgent("Pro", events),
            _RecordingAgent("Con", events),
        ]
        kwargs = {}
        if speaker_order is not None:
            kwargs["speaker_order"] = speaker_order
        result = SimulationRunner(
            agents,
            rounds=1,
            live_json=False,
            **kwargs,
        ).run()
        return events, result

    def test_default_remains_pro_first(self):
        events, result = self.run_pair()

        self.assertEqual(
            [message["agent_name"] for message in result["transcript"]],
            ["Pro", "Con"],
        )
        self.assertEqual(events[0], ("speak", "Pro", 0.0))
        self.assertEqual(events[3], ("speak", "Con", 1.0))
        self.assertEqual(
            result["metadata"]["simulation"],
            {
                "speaker_order": "pro-first",
                "turn_order": ["Pro", "Con"],
            },
        )

    def test_con_first_speaks_and_updates_pro_before_pro_reply(self):
        events, result = self.run_pair("con-first")

        self.assertEqual(
            [message["agent_name"] for message in result["transcript"]],
            ["Con", "Pro"],
        )
        self.assertEqual(events[0], ("speak", "Con", 0.0))
        self.assertEqual(events[1], ("observe", "Pro", "Con"))
        self.assertEqual(events[3], ("speak", "Pro", 1.0))
        self.assertEqual(
            [message["role"] for message in result["transcript"]],
            ["user", "assistant"],
        )
        self.assertEqual(
            result["metadata"]["simulation"],
            {
                "speaker_order": "con-first",
                "turn_order": ["Con", "Pro"],
            },
        )

    def test_con_first_rejects_non_two_agent_simulation(self):
        with self.assertRaisesRegex(ValueError, "two-agent"):
            SimulationRunner(
                [_RecordingAgent("Pro", [])],
                rounds=1,
                speaker_order="con-first",
            )


if __name__ == "__main__":
    unittest.main()
