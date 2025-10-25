# Helper functions
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string

def build_prompt(text: str) -> str:
    base_prompt = """
You are an Aspect-Based Sentiment Analysis (ABSA) assistant.

Your task:
- Extract aspects (features, entities, or attributes) mentioned in the text.
- Determine sentiment toward each aspect: "positive", "negative", or "neutral".
- Provide confidence as a number between 0.0 and 1.0.
- Include the character span [start_index, end_index] of each aspect in the original text.

Respond ONLY with valid JSON in this format:
[
  {{
    "aspect": "<aspect term>",
    "sentiment": "<positive|negative|neutral>",
    "confidence": <0.0-1.0>,
    "text_span": [<start_index>, <end_index>]
  }}
]

Below are several examples to guide you.

Example 1:
Input: "The pizza was delicious but the service was terrible."
Output:
[
  {{"aspect": "pizza", "sentiment": "positive", "confidence": 0.95, "text_span": [4, 9]}},
  {{"aspect": "service", "sentiment": "negative", "confidence": 0.92, "text_span": [29, 36]}}
]

Example 2:
Input: "The laptop screen is bright, but the battery life is disappointing."
Output:
[
  {{"aspect": "screen", "sentiment": "positive", "confidence": 0.93, "text_span": [11, 17]}},
  {{"aspect": "battery life", "sentiment": "negative", "confidence": 0.90, "text_span": [32, 45]}}
]

Example 3:
Input: "The hotel room was okay and the location was fine."
Output:
[
  {{"aspect": "room", "sentiment": "neutral", "confidence": 0.76, "text_span": [10, 14]}},
  {{"aspect": "location", "sentiment": "neutral", "confidence": 0.78, "text_span": [31, 39]}}
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


def has_phrase_negation(token) -> bool:
    for t in token.subtree:
        if t.dep_ == "neg" or t.lower_ in {"not", "never", "no"}:
            return True
    return False

def find_aspect_root(chunk) -> str:
    return chunk.text.lower().strip()

def get_or_create_aspect(aspect_dict: dict, aspect: str, span: tuple):
    if aspect not in aspect_dict:
        aspect_dict[aspect] = {"scores": [], "text_span": span}

def compute_intensity_modifier(token, intensifiers, diminishers) -> float:
    intensity_mod = 1.0
    for adv in (t for t in token.lefts if t.dep_ == "advmod"):
        mod = adv.text.lower()
        intensity_mod += intensifiers.get(mod, 0) + diminishers.get(mod, 0)
    return max(0.1, intensity_mod)


def average_conjunct_sentiments(token, analyzer: SentimentIntensityAnalyzer) -> float:
    scores = [analyzer.polarity_scores(token.text)["compound"]]
    for conj in token.conjuncts:
        if conj.pos_ == "ADJ":
            phrase = " ".join([t.text for t in conj.lefts if t.dep_ == "advmod"] + [conj.text])
            scores.append(analyzer.polarity_scores(phrase)["compound"])
    return sum(scores) / len(scores)

def aggregate_sentiment_scores(scores):
    weights = [abs(s) for s in scores]
    total_weight = sum(weights) or 1
    avg = sum(s * w for s, w in zip(scores, weights)) / total_weight
    sentiment = convert_score_to_label(avg)
    confidence = abs(avg)
    if len(scores) > 1:
        consistency = 1 - (sum(abs(s - avg) for s in scores) / len(scores))
        confidence = min(1.0, confidence * (1 + consistency * 0.3))
    return sentiment, confidence



def _is_valid_candidate(candidate: str, nlp) -> bool:
    text = candidate.strip()
    if not text or len(re.sub(r"[^\w]", "", text)) < 2:
        return False
    doc = nlp(text)
    return any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc)


def _split_candidates(raw: str) -> List[str]:
    raw = raw.strip().strip('''"'“”‘’()[]{}.,:;!?-''')
    if not raw:
        return []
    parts = re.split(r'[/&,]', raw)
    final = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if " and " in p.lower():
            halves = [h.strip() for h in re.split(r'\band\b', p, flags=re.I) if h.strip()]
            if all(len(h.split()) <= 4 for h in halves) and len(halves) > 1:
                final.extend(halves)
                continue
        final.append(p)
    return final

def normalize_aspect(aspect: str) -> str:
    return aspect.lower().strip()