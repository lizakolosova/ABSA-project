# Helper functions
from typing import Dict, Tuple, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string

PRONOUNS = {"i", "you", "he", "she", "it", "they", "we", "me", "him", "her", "them", "us"}
TEMPORAL_WORDS = {
    "while", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "morning", "evening", "afternoon",
    "yesterday", "today", "tomorrow", "holiday", "weekend"
}

def build_prompt(text: str) -> str:
    base_prompt = """
You are an Aspect-Based Sentiment Analysis (ABSA) assistant.

Your task:
- Extract aspects (features, entities, or attributes) mentioned in the text.
- Determine sentiment toward each aspect: "positive", "negative", or "neutral".
- Provide confidence as a number between 0.0 and 1.0.
- Include the character span [start_index, end_index] of each aspect in the original text.

Output format:
Respond ONLY with valid JSON in this format:
[
  {{
    "aspect": "<aspect term>",
    "sentiment": "<positive|negative|neutral>",
    "confidence": <0.0-1.0>,
    "text_span": [<start_index>, <end_index>]
  }}
]

Example:
Input: "The pizza was delicious but the service was terrible."
Output:
[
  {{"aspect": "pizza", "sentiment": "positive", "confidence": 0.95, "text_span": [4, 9]}},
  {{"aspect": "service", "sentiment": "negative", "confidence": 0.92, "text_span": [29, 36]}}
]

Now analyze this text:
"{text}"
"""
    return base_prompt.format(text=text)

def convert_score_to_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def has_phrase_negation(token):
    negations = {"not", "never", "no", "none", "n't"}
    check_tokens = list(token.subtree) + list(token.head.subtree)
    return any(t.text.lower() in negations for t in check_tokens)

def find_aspect_root(chunk) -> str:
    aspect_tokens = [t.text.lower() for t in chunk if t.dep_ == "compound" or t == chunk.root]
    return " ".join(aspect_tokens) if aspect_tokens else chunk.root.text.lower()

def get_or_create_aspect(aspect_dict: dict, aspect: str, span: tuple):
    if aspect not in aspect_dict:
        aspect_dict[aspect] = {"scores": [], "text_span": span}

def compute_intensity_modifier(token, intensifiers: Dict[str, float], diminishers: Dict[str, float]) -> float:
    intensity_mod = 1.0
    for adv in token.lefts:
        if adv.dep_ == "advmod":
            adv_lower = adv.text.lower()
            if adv_lower in intensifiers:
                intensity_mod += intensifiers[adv_lower]
            elif adv_lower in diminishers:
                intensity_mod += diminishers[adv_lower]
    return max(0.1, intensity_mod)


def average_conjunct_sentiments(token, analyzer: SentimentIntensityAnalyzer) -> float:
    score = analyzer.polarity_scores(token.text)["compound"]
    for conj in [t for t in token.conjuncts if t.pos_ == "ADJ"]:
        conj_phrase_tokens = [t.text for t in conj.lefts if t.dep_ == "advmod"] + [conj.text]
        conj_phrase = " ".join(conj_phrase_tokens)
        conj_score = analyzer.polarity_scores(conj_phrase)["compound"]
        score = (score + conj_score) / 2
    return score


def aggregate_sentiment_scores(scores: List[float]) -> Tuple[str, float]:
    weights = [abs(s) for s in scores]
    total_weight = sum(weights) if sum(weights) > 0 else 1
    avg_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

    sentiment_label = convert_score_to_label(avg_score)
    confidence = abs(avg_score)

    if len(scores) > 1:
        consistency = 1 - (sum(abs(s - avg_score) for s in scores) / len(scores))
        confidence = min(1.0, confidence * (1 + consistency * 0.3))

    return sentiment_label, confidence


def _is_valid_candidate(candidate: str, nlp) -> bool:
    text = candidate.strip()
    if not text or len(re.sub(r"[^A-Za-z0-9]", "", text)) < 2:
        return False

    low = text.lower()
    if low in PRONOUNS or any(t in low for t in TEMPORAL_WORDS):
        return False

    doc = nlp(text)
    return any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc)


def _split_candidates(raw: str) -> List[str]:
        raw = raw.strip().strip('''"'“”‘’()[]{}.,:;!?-''')
        if not raw:
            return []

        parts = re.split(r'[/&,]', raw)
        final_parts = []
        for p in parts:
            p = p.strip()
            if not p:
                continue

            if " and " in p.lower():
                halves = [h.strip() for h in re.split(r'\band\b', p, flags=re.IGNORECASE) if h.strip()]
                if all(len(h.split()) <= 4 for h in halves) and len(halves) > 1:
                    final_parts.extend(halves)
                    continue
            final_parts.append(p)

        return final_parts

def normalize_aspect(aspect: str) -> str:
    return aspect.lower().strip()

def is_valid_aspect_token(token: str) -> bool:
    return token.strip() and token not in string.punctuation

