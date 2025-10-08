import json
import random
import pytest
from src.transformer_absa import TransformerABSA
from src.lexicon_absa import LexiconABSA

# Load datasets
with open("../data/test_samples.json", "r", encoding="utf-8") as f:
    REVIEWS = json.load(f)

with open("../data/evaluation_data.json", "r", encoding="utf-8") as f:
    EVAL_DATA = json.load(f)

# Initialize both models
transformer_absa = TransformerABSA(confidence_threshold=0.0)
lexicon_absa = LexiconABSA()

ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}

def to_set(aspects):
    """Convert list of AspectSentiment objects, dicts, or tuples into a set of (aspect, sentiment) tuples."""
    if not aspects:
        return set()
    sample = aspects[0]
    if isinstance(sample, dict):
        return {(a["aspect"].lower(), a["sentiment"].lower()) for a in aspects}
    elif hasattr(sample, "aspect"):
        return {(a.aspect.lower(), a.sentiment.lower()) for a in aspects}
    elif isinstance(sample, (list, tuple)):
        return {tuple(a) for a in aspects}
    return set()


def assert_aspects_valid(text, aspects):
    """Ensure each aspect is structurally correct (valid span, label, sentiment)."""
    for a in aspects:
        if isinstance(a, dict):
            aspect, sentiment, span = a["aspect"], a["sentiment"], a["text_span"]
        else:
            aspect, sentiment, span = a.aspect, a.sentiment, a.text_span
        start, end = span
        assert 0 <= start < end <= len(text), f"Invalid span {span} for '{aspect}'"
        assert isinstance(aspect, str), f"Aspect not a string: {aspect}"
        assert sentiment.lower() in ALLOWED_SENTIMENTS, f"Invalid sentiment: {sentiment}"

SAMPLE_TEXTS = []
if "sample" in REVIEWS:
    SAMPLE_TEXTS.extend(REVIEWS["sample"])
if "simple" in REVIEWS:
    SAMPLE_TEXTS.extend(REVIEWS["simple"])
SAMPLE_TEXTS = random.sample(SAMPLE_TEXTS, k=min(10, len(SAMPLE_TEXTS)))


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
@pytest.mark.parametrize("model", ["transformer", "lexicon"])
def test_structure_accuracy(text, model):
    """
    Verify both models produce well-structured outputs:
    - list of AspectSentiment objects (or dicts)
    - valid text spans and sentiment labels
    """
    absa = transformer_absa if model == "transformer" else lexicon_absa
    aspects = absa.analyze(text)
    assert isinstance(aspects, list), f"{model}: output not a list"
    assert_aspects_valid(text, aspects)

@pytest.mark.parametrize("entry", EVAL_DATA.get(["simple", "medium"], []))
@pytest.mark.parametrize("model", ["transformer", "lexicon"])
def test_exact_match_(entry, model):
    """
    Compare model predictions on simple sentences.
    """
    absa = transformer_absa if model == "transformer" else lexicon_absa
    text = entry["text"]
    expected = to_set(entry["expected_aspects"])
    predicted = to_set(absa.analyze(text))
    assert predicted == expected, (
        f"{model.upper()} | Text: {text}\n"
        f"Expected: {expected}\nPredicted: {predicted}"
    )