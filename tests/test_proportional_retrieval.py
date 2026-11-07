from __future__ import annotations

import unittest

from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE
from src.memory.argument_record import ArgumentRecord
from src.memory.retrieval import retrieve_arguments


def argument(argument_id, polarity, strength, *, active):
    record = ArgumentRecord(
        id=argument_id,
        topic_id="Test proposition",
        claim=f"Evidence {argument_id}",
        polarity=polarity,
        source_role="opponent",
        support_strength=strength,
    )
    record.credence_relevant = active
    return record


class ProportionalRetrievalTests(unittest.TestCase):
    def test_archived_records_never_reenter_through_quota_or_backfill(self):
        active_pro = argument("active-pro", POLARITY_AFFIRMATIVE, 0.2, active=True)
        active_con = argument("active-con", POLARITY_NEGATIVE, 0.3, active=True)
        archived = [
            argument(
                f"archived-{index}",
                POLARITY_AFFIRMATIVE if index % 2 == 0 else POLARITY_NEGATIVE,
                1.0,
                active=False,
            )
            for index in range(8)
        ]

        retrieved = retrieve_arguments(
            [active_pro, active_con, *archived],
            strategy="proportional",
            k=10,
        )

        self.assertEqual(
            {record.id for record in retrieved},
            {"active-pro", "active-con"},
        )
        self.assertTrue(all(record.credence_relevant for record in retrieved))

    def test_quota_denominator_uses_only_active_records(self):
        records = [
            argument("pro-strong", POLARITY_AFFIRMATIVE, 0.9, active=True),
            argument("pro-mid", POLARITY_AFFIRMATIVE, 0.7, active=True),
            argument("pro-weak", POLARITY_AFFIRMATIVE, 0.2, active=True),
            argument("con", POLARITY_NEGATIVE, 0.8, active=True),
            argument("archived-con", POLARITY_NEGATIVE, 1.0, active=False),
        ]

        retrieved = retrieve_arguments(records, strategy="proportional", k=3)

        self.assertEqual(
            [record.id for record in retrieved],
            ["pro-strong", "pro-mid", "con"],
        )

    def test_all_archived_memory_returns_empty_context(self):
        records = [
            argument("archived-pro", POLARITY_AFFIRMATIVE, 1.0, active=False),
            argument("archived-con", POLARITY_NEGATIVE, 1.0, active=False),
        ]

        self.assertEqual(
            retrieve_arguments(records, strategy="proportional", k=5),
            [],
        )


if __name__ == "__main__":
    unittest.main()
