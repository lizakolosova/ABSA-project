import pytest
from unittest.mock import MagicMock, patch
import torch

from src.transformer_absa import TransformerABSA
from src.lexicon_absa import LexiconABSA, _normalize_candidate

# Tests for TransformerABSA

@pytest.fixture
def transformer_absa():
    with patch("src.transformer_absa.pipeline") as mock_pipeline, \
         patch("src.transformer_absa.AutoTokenizer"), \
         patch("src.transformer_absa.AutoModelForSequenceClassification"):

        # Mock aspect extractor
        mock_extractor = MagicMock()
        mock_extractor.return_value = [
            {"word": "Food", "entity_group": "B-ASP"},
            {"word": "Service", "entity_group": "B-ASP"},
            {"word": "Omitted", "entity_group": "O"}
        ]
        mock_pipeline.return_value = mock_extractor

        analyzer = TransformerABSA(device=-1)
        yield analyzer

def test_extract_aspects(transformer_absa, monkeypatch):
    monkeypatch.setattr("src.transformer_absa.normalize_aspect", lambda w: w.lower())
    monkeypatch.setattr("src.transformer_absa.is_valid_aspect_token", lambda w: True)

    text = "The Food was great and the Service was slow."
    aspects = transformer_absa.extract_aspects(text)
    assert "food" in aspects
    assert "service" in aspects
    assert len(aspects) == 2

def test_classify_sentiment(transformer_absa, monkeypatch):
    class DummyModelOutput:
        def __init__(self):
            self.logits = torch.tensor([[0.1, 0.2, 0.7]])  # predicts index 2 -> positive

    dummy_tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])})
    monkeypatch.setattr(transformer_absa, "tokenizer", dummy_tokenizer)
    monkeypatch.setattr(transformer_absa, "sentiment_model", MagicMock(return_value=DummyModelOutput()))

    sentiment = transformer_absa.classify_sentiment("Great food!", "Food")
    assert sentiment == "positive"

def test_analyze(transformer_absa, monkeypatch):
    monkeypatch.setattr("src.transformer_absa.normalize_aspect", lambda w: w.lower())
    monkeypatch.setattr("src.transformer_absa.is_valid_aspect_token", lambda w: True)
    class DummyModelOutput:
        def __init__(self):
            self.logits = torch.tensor([[0.1, 0.2, 0.7]])
    transformer_absa.sentiment_model = MagicMock(return_value=DummyModelOutput())
    transformer_absa.tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])})

    results = transformer_absa.analyze("The Food was great and the Service was slow.")
    assert ("food", "positive") in results
    assert ("service", "positive") in results

# Tests for LexiconABSA
@pytest.fixture
def lexicon_absa(monkeypatch):
    # Mock SpaCy NLP pipeline and SentimentIntensityAnalyzer
    monkeypatch.setattr("src.lexicon_absa.spacy.load", lambda model: MagicMock())
    monkeypatch.setattr("src.lexicon_absa.SentimentIntensityAnalyzer", lambda: MagicMock(polarity_scores=lambda t: {"compound": 0.5}))
    analyzer = LexiconABSA()
    yield analyzer

def test_normalize_candidate():
    assert _normalize_candidate("The food") == "food"
    assert _normalize_candidate("a service") == "service"
    assert _normalize_candidate("these things") == "things"

def test_get_sentiment_score(monkeypatch, lexicon_absa):
    dummy_token = MagicMock()
    dummy_token.text = "great"
    dummy_token.lefts = []
    dummy_token.children = []
    dummy_token.dep_ = "amod"
    dummy_token.pos_ = "ADJ"
    monkeypatch.setattr("src.lexicon_absa.compute_intensity_modifier", lambda token, ints, dims: 1.0)
    monkeypatch.setattr("src.lexicon_absa.average_conjunct_sentiments", lambda token, analyzer: 0.5)
    monkeypatch.setattr("src.lexicon_absa.has_phrase_negation", lambda token: False)

    score = lexicon_absa.get_sentiment_score(dummy_token)
    assert score == 0.5

# Note: Full analyze method heavily depends on SpaCy NLP output; we test basic functionality
def test_analyze_returns_list(monkeypatch, lexicon_absa):
    monkeypatch.setattr("src.lexicon_absa.get_or_create_aspect", lambda adict, aspect, span: adict.setdefault(aspect, {"scores": [], "text_span": span}))
    monkeypatch.setattr("src.lexicon_absa.find_aspect_root", lambda chunk: "food")
    monkeypatch.setattr("src.lexicon_absa.aggregate_sentiment_scores", lambda scores: ("positive", 0.9))

    # Mock doc with minimal noun_chunks
    dummy_doc = MagicMock()
    dummy_doc.noun_chunks = [MagicMock(start_char=0, end_char=4, root=MagicMock(pos_="NOUN", children=[]))]
    dummy_doc.__iter__.return_value = []
    monkeypatch.setattr(lexicon_absa, "nlp", lambda text: dummy_doc)

    results = lexicon_absa.analyze("The food is great.")
    assert isinstance(results, list)
