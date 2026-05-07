from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE, POLARITY_MAPPING

@dataclass
class ArgumentRecord:
    """
    Represents a single extracted argument.
    """
    topic_id: str
    claim: str  # Canonical claim sentence
    polarity: str  # "affirmative" or "negative"
    source_role: str  # "self" or "opponent"
    support_strength: float  # 0.0 to 1.0 (active score, follows config)
    id: str = field(default="") # Unique ID
    support_strength_llm: float = 0.0  # LLM-assigned score
    support_strength_classifier: float = 0.0  # Classifier-assigned score
    evidence_snippets: List[str] = field(default_factory=list)
    category_tags: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    last_accessed: datetime = field(default_factory=datetime.now)
    context: str = "General"
    category: str = "General"
    links: List[Dict] = field(default_factory=list)
    retrieval_count: int = 0
    evolution_history: List = field(default_factory=list)
    credence_relevant: bool = False  # Boolean tag for credence relevance

    def __post_init__(self):
        if not self.id:
            self.id = str(hash(self.claim))
        
        # Normalize polarity
        if self.polarity in POLARITY_MAPPING:
            self.polarity = POLARITY_MAPPING[self.polarity]

    def to_dict(self):
        return {
            "topic_id": self.topic_id,
            "claim": self.claim,
            "polarity": self.polarity,
            "source_role": self.source_role,
            "support_strength": self.support_strength,
            "support_strength_llm": self.support_strength_llm,
            "support_strength_classifier": self.support_strength_classifier,
            "evidence_snippets": self.evidence_snippets,
            "category_tags": self.category_tags,
            "timestamp": self.timestamp.isoformat(),
            "credence_relevant": self.credence_relevant
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            topic_id=data["topic_id"],
            claim=data["claim"],
            polarity=data["polarity"],
            source_role=data["source_role"],
            support_strength=data["support_strength"],
            support_strength_llm=data.get("support_strength_llm", 0.0),
            support_strength_classifier=data.get("support_strength_classifier", 0.0),
            evidence_snippets=data.get("evidence_snippets", []),
            category_tags=data.get("category_tags", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            credence_relevant=data.get("credence_relevant", False)
        )
