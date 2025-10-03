# Base classes and interfaces
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AspectSentiment:
    """
    Data class for storing aspect-sentiment pairs.
    """
    aspect: str  # The aspect/feature mentioned
    sentiment: str  # Sentiment label: 'positive', 'negative', 'neutral'
    confidence: float  # Confidence score (0.0 to 1.0)
    text_span: Tuple[int, int]  # Optional: (start, end) indices in the original text


class ABSAAnalyzer:
    """
    Base class/interface for all ABSA implementations.
    Every implementation (LexiconABSA, ML_ABSA, LLMABSA) must inherit this class.
    """

    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze text and extract aspect-sentiment pairs.

        Args:
            text (str): Input text to analyze.

        Returns:
            List[AspectSentiment]: List of extracted aspect-sentiment objects.
        """
        raise NotImplementedError("Subclasses must implement this method")
