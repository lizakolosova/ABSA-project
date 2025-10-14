import json
import random
import pytest
import shutil
from src.llm_absa import LLMABSA

# Skip all tests if Ollama is not installed locally
pytestmark = pytest.mark.skipif(
    shutil.which("ollama") is None,
    reason="Ollama not installed locally"
)

# --- Load test data ---
with open("../data/test_samples.json", "r", encoding="utf-8") as f:
    REVIEWS = json.load(f)

with open("../data/evaluation_data.json", "r", encoding="utf-8") as f:
    EVAL_DATA = json.load(f)

llm_absa = LLMABSA(model="llama3")

ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}


def to_set(aspects):
    """Convert list of AspectSentiment objects or dicts into a set of (aspect, sentiment) pairs."""
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
    """Ensure each aspect has valid structure (span, label, sentiment)."""
    for a in aspects:
        if isinstance(a, dict):
            aspect, sentiment, span = a["aspect"], a["sentiment"], a["text_span"]
        else:
            aspect, sentiment, span = a.aspect, a.sentiment, a.text_span
        start, end = span
        assert isinstance(aspect, str), f"Aspect not a string: {aspect}"
        assert sentiment.lower() in ALLOWED_SENTIMENTS, f"Invalid sentiment: {sentiment}"
        assert 0 <= start < end <= len(text), f"Invalid span {span} for '{aspect}'"


SAMPLE_TEXTS = []
if "sample" in REVIEWS:
    SAMPLE_TEXTS.extend(REVIEWS["sample"])
if "simple" in REVIEWS:
    SAMPLE_TEXTS.extend(REVIEWS["simple"])
SAMPLE_TEXTS = random.sample(SAMPLE_TEXTS, k=min(10, len(SAMPLE_TEXTS)))


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_llm_output_structure(text):
    """
    Test 1: Verify that LLMABSA outputs a valid structure.
    """
    aspects = llm_absa.analyze(text)
    assert isinstance(aspects, list), "LLMABSA output is not a list"
    assert_aspects_valid(text, aspects)


@pytest.mark.parametrize("entry", EVAL_DATA.get("simple", []))
def test_llm_expected_aspects(entry):
    """
    Test 2: Compare predicted vs expected aspects on simple sentences.
    """
    text = entry["text"]
    expected = to_set(entry["expected_aspects"])
    predicted = to_set(llm_absa.analyze(text))

    assert predicted == expected, (
        f"LLMABSA | Text: {text}\n"
        f"Expected: {expected}\nPredicted: {predicted}"
    )
