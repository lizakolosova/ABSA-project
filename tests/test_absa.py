import pytest
from unittest.mock import MagicMock, patch
import torch
from src.transformer_absa import TransformerABSA
from src.lexicon_absa import LexiconABSA


# ----------------------------
# TransformerABSA
# ----------------------------
@pytest.fixture
def transformer_absa():
    with patch("src.transformer_absa.pipeline") as mock_pipeline, \
         patch("src.transformer_absa.AutoTokenizer"), \
         patch("src.transformer_absa.AutoModelForSequenceClassification"):

        # Make the mock extractor callable
        mock_extractor = MagicMock()
        mock_extractor.side_effect = lambda text: [
            {"word": "Food", "entity_group": "B-ASP", "score": 0.9, "start": 4, "end": 8},
            {"word": "Service", "entity_group": "B-ASP", "score": 0.8, "start": 21, "end": 28},
            {"word": "Omitted", "entity_group": "O", "score": 0.5}
        ]
        mock_pipeline.return_value = mock_extractor

        analyzer = TransformerABSA(device=-1)
        yield analyzer

def test_extract_aspects(transformer_absa, monkeypatch):
    # Patch normalization to lowercase
    monkeypatch.setattr("src.utils.normalize_aspect", lambda w: w.lower())

    aspects = transformer_absa.extract_aspects("The Food was great and the Service was slow.")
    assert aspects == ["food", "service"]
    assert len(aspects) == 2


def test_classify_sentiment(transformer_absa, monkeypatch):
    class DummyModelOutput:
        def __init__(self):
            self.logits = torch.tensor([[0.1, 0.2, 0.7]])  # → index 2 → positive

    transformer_absa.tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])})
    transformer_absa.sentiment_model = MagicMock(return_value=DummyModelOutput())

    sentiment = transformer_absa.classify_sentiment("Great food!", "Food")
    assert sentiment == "positive"


def test_analyze(transformer_absa, monkeypatch):
    monkeypatch.setattr("src.utils.normalize_aspect", lambda w: w.lower())

    class DummyModelOutput:
        def __init__(self):
            self.logits = torch.tensor([[0.1, 0.2, 0.7]])  # → index 2 → positive

    transformer_absa.sentiment_model = MagicMock(return_value=DummyModelOutput())
    transformer_absa.tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])})

    results = transformer_absa.analyze("The Food was great and the Service was slow.")
    assert isinstance(results, list)
    assert all(hasattr(r, "aspect") and hasattr(r, "sentiment") for r in results)
    aspects = [(r.aspect, r.sentiment) for r in results]
    assert ("food", "positive") in aspects
    assert ("service", "positive") in aspects


# ----------------------------
# LexiconABSA
# ----------------------------
@pytest.fixture
def lexicon_absa(monkeypatch):
    dummy_doc = MagicMock()
    dummy_doc.noun_chunks = []
    dummy_doc.__iter__.return_value = []
    dummy_doc.text = "dummy"

    mock_nlp = MagicMock(return_value=dummy_doc)
    mock_nlp.Defaults.stop_words = set()

    monkeypatch.setattr("src.lexicon_absa.spacy.load", lambda model: mock_nlp)
    monkeypatch.setattr("src.lexicon_absa.SentimentIntensityAnalyzer",
                        lambda: MagicMock(polarity_scores=lambda t: {"compound": 0.5}))
    return LexiconABSA()


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


def test_analyze_returns_list(monkeypatch, lexicon_absa):
    monkeypatch.setattr("src.lexicon_absa.get_or_create_aspect",
                        lambda adict, aspect, span: adict.setdefault(aspect, {"scores": [], "text_span": span}))
    monkeypatch.setattr("src.lexicon_absa.aggregate_sentiment_scores",
                        lambda scores: ("positive", 0.9))
    monkeypatch.setattr("src.lexicon_absa._strip_leading_dets",
                        lambda chunk: chunk)
    monkeypatch.setattr("src.lexicon_absa._split_candidates",
                        lambda candidate: [candidate])
    monkeypatch.setattr("src.lexicon_absa._is_valid_candidate",
                        lambda cand, nlp: True)

    dummy_chunk = MagicMock(start_char=0, end_char=4, text="food", root=MagicMock(pos_="NOUN", children=[]))
    dummy_doc = MagicMock()
    dummy_doc.noun_chunks = [dummy_chunk]
    dummy_doc.__iter__.return_value = []
    lexicon_absa.nlp = MagicMock(return_value=dummy_doc)

    results = lexicon_absa.analyze("The food is great.")
    assert isinstance(results, list)