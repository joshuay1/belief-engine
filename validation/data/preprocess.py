"""
Build per-participant timelines from raw DEBATE event log.

Each ParticipantTimeline contains the full experience of one participant:
initial opinion, 3 rounds of messages (sent + received), and final opinion.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import pandas as pd

from .scale_mapping import likert_to_stance

logger = logging.getLogger(__name__)


@dataclass
class RoundData:
    round_number: int
    partner_id: str
    tweets_received: List[str] = field(default_factory=list)
    tweet_sent: str = ""
    messages_received: List[str] = field(default_factory=list)
    messages_sent: List[str] = field(default_factory=list)


@dataclass
class ParticipantTimeline:
    worker_id: str
    group_file: str
    topic: str
    topic_category: str  # "depth" or "breadth"
    initial_opinion: float  # Likert 1-6
    initial_stance: float   # Mapped to [-1, 1]
    initial_text: str       # Free-text justification
    rounds: List[RoundData] = field(default_factory=list)
    final_opinion: Optional[float] = None  # Likert 1-6
    final_stance: Optional[float] = None   # Mapped to [-1, 1]
    final_text: str = ""
    demographics: Dict = field(default_factory=dict)

    @property
    def opinion_change(self) -> Optional[float]:
        """Actual opinion change in stance space [-1, 1]."""
        if self.final_stance is not None:
            return self.final_stance - self.initial_stance
        return None

    @property
    def is_complete(self) -> bool:
        """Has initial + final opinion and at least 1 round of conversation."""
        return (
            self.final_opinion is not None
            and len(self.rounds) > 0
            and any(
                r.messages_received or r.messages_sent or r.tweets_received
                for r in self.rounds
            )
        )


def _is_valid_text(text) -> bool:
    """Check if text is a non-empty, non-nan string."""
    if pd.isna(text):
        return False
    s = str(text).strip()
    return s != "" and s.lower() != "nan"


def build_participant_timelines(
    df: pd.DataFrame,
    max_participants: Optional[int] = None,
) -> List[ParticipantTimeline]:
    """Reconstruct per-participant timelines from raw DEBATE event log.

    Args:
        df: DataFrame from load_debate_dataset() with topic, group_file columns.
        max_participants: If set, stop after collecting this many complete timelines.

    Returns:
        List of ParticipantTimeline objects (only complete timelines included).
    """
    timelines = []
    skipped = {"no_initial": 0, "no_final": 0, "no_messages": 0}

    for group_file, group_df in df.groupby("group_file"):
        group_df = group_df.sort_values("event_order")
        topic = group_df["topic"].iloc[0]
        topic_category = group_df["topic_category"].iloc[0]

        # Get initial opinions
        init_events = group_df[group_df["event_type"] == "Initial Opinion"]
        # Get post opinions
        post_events = group_df[group_df["event_type"] == "Post Opinion"]
        # Get exit surveys
        exit_events = group_df[group_df["event_type"] == "exit_survey"]

        # Index post opinions by worker
        post_by_worker = {}
        for _, row in post_events.iterrows():
            post_by_worker[row["worker_id"]] = row

        # Index demographics by worker
        demo_by_worker = {}
        for _, row in exit_events.iterrows():
            wid = row["worker_id"]
            if wid not in demo_by_worker:
                demo_by_worker[wid] = {}
            if _is_valid_text(row["field"]) and _is_valid_text(row["text"]):
                demo_by_worker[wid][row["field"]] = row["text"]

        # Build timeline per worker
        for _, init_row in init_events.iterrows():
            worker_id = init_row["worker_id"]

            if pd.isna(init_row["sliderValue"]):
                skipped["no_initial"] += 1
                continue

            initial_opinion = float(init_row["sliderValue"])
            initial_text = str(init_row["text"]) if _is_valid_text(init_row["text"]) else ""

            timeline = ParticipantTimeline(
                worker_id=worker_id,
                group_file=group_file,
                topic=topic,
                topic_category=topic_category,
                initial_opinion=initial_opinion,
                initial_stance=likert_to_stance(initial_opinion),
                initial_text=initial_text,
                demographics=demo_by_worker.get(worker_id, {}),
            )

            # Post opinion
            if worker_id in post_by_worker:
                post_row = post_by_worker[worker_id]
                if not pd.isna(post_row["sliderValue"]):
                    timeline.final_opinion = float(post_row["sliderValue"])
                    timeline.final_stance = likert_to_stance(timeline.final_opinion)
                    timeline.final_text = (
                        str(post_row["text"]) if _is_valid_text(post_row["text"]) else ""
                    )

            # Build rounds
            worker_events = group_df[group_df["worker_id"] == worker_id].sort_values(
                "event_order"
            )
            for round_num in [1.0, 2.0, 3.0]:
                round_events = worker_events[
                    worker_events["chat_round_order"] == round_num
                ]
                if round_events.empty:
                    continue

                # Determine partner from sent/received messages or tweet recipient
                partner_id = _infer_partner(round_events)

                rd = RoundData(round_number=int(round_num), partner_id=partner_id or "")

                # Own tweet this round
                own_tweets = round_events[round_events["event_type"] == "tweet"]
                for _, t in own_tweets.iterrows():
                    if _is_valid_text(t["text"]):
                        rd.tweet_sent = str(t["text"])

                # Messages sent
                sent = round_events[round_events["event_type"] == "message_sent"]
                for _, m in sent.iterrows():
                    if _is_valid_text(m["text"]):
                        rd.messages_sent.append(str(m["text"]))

                # Messages received
                recv = round_events[round_events["event_type"] == "message_received"]
                for _, m in recv.iterrows():
                    if _is_valid_text(m["text"]):
                        rd.messages_received.append(str(m["text"]))

                # Tweets are directed in the raw event log; only include tweets
                # addressed to this participant, not every tweet in the session.
                directed_tweets = group_df[
                    (group_df["chat_round_order"] == round_num)
                    & (group_df["event_type"] == "tweet")
                    & (group_df["recipient_id"] == worker_id)
                ]
                for _, t in directed_tweets.iterrows():
                    if _is_valid_text(t["text"]):
                        rd.tweets_received.append(str(t["text"]))

                timeline.rounds.append(rd)

            # Only keep complete timelines
            if timeline.is_complete:
                timelines.append(timeline)
                if max_participants and len(timelines) >= max_participants:
                    break
            else:
                if timeline.final_opinion is None:
                    skipped["no_final"] += 1
                else:
                    skipped["no_messages"] += 1

        if max_participants and len(timelines) >= max_participants:
            break

    logger.info(
        f"Built {len(timelines)} complete timelines, "
        f"skipped: {skipped}"
    )
    print(
        f"Built {len(timelines)} complete timelines, "
        f"skipped: {skipped}"
    )
    return timelines


def _infer_partner(round_events: pd.DataFrame) -> Optional[str]:
    """Infer the dyadic partner from message_sent/received events or tweet recipient."""
    # From messages received: sender_id is the partner
    recv = round_events[round_events["event_type"] == "message_received"]
    for _, r in recv.iterrows():
        if _is_valid_text(r["sender_id"]):
            return str(r["sender_id"])

    # From messages sent: recipient_id is the partner
    sent = round_events[round_events["event_type"] == "message_sent"]
    for _, r in sent.iterrows():
        if _is_valid_text(r["recipient_id"]):
            return str(r["recipient_id"])

    # From tweet: recipient_id is the partner for that round
    tweets = round_events[round_events["event_type"] == "tweet"]
    for _, t in tweets.iterrows():
        if _is_valid_text(t["recipient_id"]):
            return str(t["recipient_id"])

    return None
