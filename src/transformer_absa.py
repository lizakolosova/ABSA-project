# transformer_absa.py - Implementation 2 corrected (head noun only)

from typing import List
import re
import spacy
from transformers import pipeline
from src.base import ABSAAnalyzer, AspectSentiment
from src.utils import _split_candidates, _is_valid_candidate

PRONOUNS = {"i", "you", "he", "she", "it", "they", "we", "me", "him", "her", "them", "us"}
TEMPORAL_WORDS = {
    "while", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "morning", "evening", "afternoon",
    "yesterday", "today", "tomorrow", "holiday", "weekend"
}

DETERMINERS = {"this", "that", "these", "those", "a", "an", "the"}


def _merge_adjacent_aspects(aspects: List[AspectSentiment]) -> List[AspectSentiment]:
    if not aspects:
        return []
    aspects.sort(key=lambda a: a.text_span[0])
    merged = [aspects[0]]
    for cur in aspects[1:]:
        last = merged[-1]
        if cur.sentiment == last.sentiment and cur.text_span[0] <= last.text_span[1] + 1:
            merged[-1] = AspectSentiment(
                aspect=f"{last.aspect} {cur.aspect}",
                sentiment=last.sentiment,
                confidence=max(last.confidence, cur.confidence),
                text_span=(last.text_span[0], cur.text_span[1])
            )
        else:
            merged.append(cur)
    return merged


def _normalize_candidate(text: str) -> str:
    """Remove leading determiners like 'the', 'a', 'an', 'this', 'that'"""
    words = text.strip().split()
    while words and words[0].lower() in DETERMINERS:
        words.pop(0)
    return " ".join(words)


class TransformerABSA(ABSAAnalyzer):
    """
    Transformer-based ABSA:
    - Extracts candidate aspects using spaCy noun chunks (head noun only).
    - Removes leading determiners from candidates.
    - Splits candidates on slashes, &, commas, "and".
    - Runs ABSA model per candidate.
    - Deduplicates and merges only truly adjacent aspects of the same sentiment.
    """

    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1", confidence_threshold: float = 0.55):
        self.nlp = spacy.load("en_core_web_sm")
        self.absa_pipeline = pipeline("text-classification", model=model_name)
        self.conf_threshold = confidence_threshold

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)
        candidates = []

        # Extract noun chunks (head noun only)
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == "PRON":
                continue
            chunk_text = chunk.root.text  # only the head noun
            for c in _split_candidates(chunk_text):
                c = _normalize_candidate(c)
                if _is_valid_candidate(c, self.nlp):
                    candidates.append(c)

        # Deduplicate candidates
        seen = set()
        deduped = []
        for c in candidates:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        # Run ABSA pipeline
        aspects = []
        for cand in deduped:
            input_text = f"{text} [ASP] {cand}"
            try:
                pred = self.absa_pipeline(input_text)[0]
            except Exception:
                continue

            score = float(pred.get("score", 0.0))
            label = str(pred.get("label", "")).lower()
            if score < self.conf_threshold:
                continue

            sentiment = "neutral"
            if "pos" in label:
                sentiment = "positive"
            elif "neg" in label:
                sentiment = "negative"

            m = re.search(re.escape(cand), text, flags=re.IGNORECASE)
            if m:
                start, end = m.start(), m.end()
            else:
                start = text.lower().find(cand.lower())
                end = start + len(cand)

            aspects.append(
                AspectSentiment(
                    aspect=cand,
                    sentiment=sentiment,
                    confidence=score,
                    text_span=(start, end)
                )
            )

        return _merge_adjacent_aspects(aspects)
