from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod
import json


@dataclass
class AspectSentiment:
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    text_span: Tuple[int, int]
    model_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        self.aspect = self.aspect.strip()
        if not self.aspect:
            raise ValueError("Aspect cannot be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be [0, 1], got {self.confidence}")

        start, end = self.text_span
        if start < 0 or end < start:
            raise ValueError(f"Invalid text_span: {self.text_span}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aspect": self.aspect,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "text_span": list(self.text_span),
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AspectSentiment":
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if "text_span" in data and isinstance(data["text_span"], list):
            data["text_span"] = tuple(data["text_span"])
        return cls(**data)


class ABSAAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> List[AspectSentiment]:
        raise NotImplementedError

    def analyze_batch(self, texts: List[str]) -> List[List[AspectSentiment]]:
        return [self.analyze(text) for text in texts]

    def validate_input(self, text: str, max_length: int = 100_000) -> None:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        text = text.replace('\x00', '').strip()
        if not text:
            raise ValueError("Input text cannot be empty")

        if len(text) > max_length:
            raise ValueError(f"Text length ({len(text)}) exceeds maximum ({max_length})")