from typing import Tuple, List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def score_to_label(score: float) -> str:
    match score:
        case s if s >= 0.05:
            return "positive"
        case s if s <= -0.05:
            return "negative"
        case _:
            return "neutral"


def compute_intensity_modifier(
    token,
    intensifiers: Dict[str, float],
    diminishers: Dict[str, float]
) -> float:
    intensity = 1.0
    
    for adv in token.lefts:
        if adv.dep_ == "advmod":
            adv_lower = adv.text.lower()
            intensity += intensifiers.get(adv_lower, diminishers.get(adv_lower, 0.0))
    
    return max(0.1, intensity)


def average_conjunct_sentiments(token, analyzer: SentimentIntensityAnalyzer) -> float:
    score = analyzer.polarity_scores(token.text)["compound"]

    for conj in (t for t in token.conjuncts if t.pos_ == "ADJ"):
        phrase_tokens = [t.text for t in conj.lefts if t.dep_ == "advmod"] + [conj.text]
        phrase = " ".join(phrase_tokens)
        conj_score = analyzer.polarity_scores(phrase)["compound"]
        score = (score + conj_score) / 2
    
    return score


def aggregate_scores(scores: List[float]) -> Tuple[str, float]:
    if not scores:
        return "neutral", 0.0

    weights = [abs(s) for s in scores]
    total_weight = sum(weights) or 1.0
    avg_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

    sentiment_label = score_to_label(avg_score)
    confidence = abs(avg_score)

    if len(scores) > 1:
        consistency = 1 - (sum(abs(s - avg_score) for s in scores) / len(scores))
        confidence = min(1.0, confidence * (1 + consistency * 0.3))
    
    return sentiment_label, confidence


def create_aspect_entry(aspect: str, span: Tuple[int, int], aspect_dict: dict) -> None:
    if aspect not in aspect_dict:
        aspect_dict[aspect] = {"scores": [], "text_span": span}
