import pytest
from unittest.mock import MagicMock, patch, Mock
import torch

from src.transformer_absa import TransformerABSA
from src.lexicon_absa import LexiconABSA
from src.llm_absa import LLMABSA
from src.base import AspectSentiment
from src.base import ABSAAnalyzer


class TestAspectSentiment:
    def test_valid_creation(self):
        aspect = AspectSentiment(
            aspect="food",
            sentiment="positive",
            confidence=0.9,
            text_span=(0, 4)
        )
        assert aspect.aspect == "food"
        assert aspect.sentiment == "positive"
        assert aspect.confidence == 0.9

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            AspectSentiment("food", "positive", 1.5, (0, 4))

    def test_invalid_span_raises(self):
        with pytest.raises(ValueError):
            AspectSentiment("food", "positive", 0.9, (10, 5))

    def test_to_dict(self):
        aspect = AspectSentiment("food", "positive", 0.9, (0, 4))
        data = aspect.to_dict()
        assert data["aspect"] == "food"
        assert data["sentiment"] == "positive"
        assert data["text_span"] == [0, 4]

    def test_from_dict(self):
        data = {
            "aspect": "food",
            "sentiment": "positive",
            "confidence": 0.9,
            "text_span": [0, 4]
        }
        aspect = AspectSentiment.from_dict(data)
        assert aspect.aspect == "food"
        assert aspect.text_span == (0, 4)


class TestTransformerABSA:
    @pytest.fixture
    def mock_transformer(self):
        with patch("src.transformer_absa.pipeline"), \
                patch("src.transformer_absa.AutoTokenizer"), \
                patch("src.transformer_absa.AutoModelForSequenceClassification"):
            analyzer = TransformerABSA.__new__(TransformerABSA)

            mock_extractor = MagicMock()
            mock_extractor.return_value = [
                {"word": "food", "entity_group": "B-ASP", "score": 0.9, "start": 0, "end": 4}
            ]
            analyzer.aspect_extractor = mock_extractor

            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]])
            }
            analyzer.tokenizer = mock_tokenizer

            mock_model = MagicMock()
            mock_output = Mock()
            mock_output.logits = torch.tensor([[0.1, 0.2, 0.7]])
            mock_model.return_value = mock_output
            analyzer.sentiment_model = mock_model

            analyzer.label_map = ["negative", "neutral", "positive"]
            analyzer.device = "cpu"
            analyzer.max_length = 512

            yield analyzer

    def test_extract_aspects(self, mock_transformer):
        aspects = mock_transformer.extract_aspects("The food was great")
        assert "food" in aspects

    def test_analyze_returns_list(self, mock_transformer):
        results = mock_transformer.analyze("The food was great")
        assert isinstance(results, list)
        assert all(isinstance(r, AspectSentiment) for r in results)


class TestLexiconABSA:
    @pytest.fixture
    def mock_lexicon(self, monkeypatch):
        mock_doc = MagicMock()
        mock_doc.noun_chunks = []
        mock_doc.__iter__ = lambda self: iter([])

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("src.model_cache.ModelCache.get", return_value=mock_nlp):
            analyzer = LexiconABSA()
            yield analyzer

    def test_initialization(self, mock_lexicon):
        assert mock_lexicon.model_name == "en_core_web_trf"
        assert hasattr(mock_lexicon, 'analyzer')

    def test_analyze_empty_raises(self, mock_lexicon):
        with pytest.raises(ValueError):
            mock_lexicon.analyze("")

    def test_analyze_returns_list(self, mock_lexicon):
        result = mock_lexicon.analyze("Test text")
        assert isinstance(result, list)


class TestLLMABSA:
    @pytest.fixture
    def mock_llm(self):
        with patch("src.llm_absa.ollama.list") as mock_list:
            mock_list.return_value = {"models": [{"name": "llama3"}]}

            analyzer = LLMABSA(model="llama3")

            mock_chat = MagicMock()
            mock_chat.return_value = {
                "message": {
                    "content": '[{"aspect": "food", "sentiment": "positive", "confidence": 0.9, "text_span": [0, 4]}]'
                }
            }

            with patch("src.llm_absa.ollama.chat", mock_chat):
                yield analyzer

    def test_analyze_with_cache(self, mock_llm):
        with patch("src.llm_absa.ollama.chat") as mock_chat:
            mock_chat.return_value = {
                "message": {
                    "content": '[{"aspect": "food", "sentiment": "positive", "confidence": 0.9, "text_span": [0, 4]}]'
                }
            }

            result1 = mock_llm.analyze("Test text")
            result2 = mock_llm.analyze("Test text")

            assert len(result1) > 0
            assert mock_chat.call_count == 1  # Should hit cache on 2nd call

    def test_parse_response_valid_json(self, mock_llm):
        raw = '[{"aspect": "food", "sentiment": "positive", "confidence": 0.9, "text_span": [0, 4]}]'
        parsed = mock_llm._parse_response(raw)
        assert len(parsed) == 1
        assert parsed[0]["aspect"] == "food"

    def test_parse_response_invalid_json_returns_empty(self, mock_llm):
        raw = "not json"
        parsed = mock_llm._parse_response(raw)
        assert parsed == []


class TestBatchProcessing:
    @pytest.fixture
    def mock_analyzer(self):
        analyzer = MagicMock(spec=LexiconABSA)
        analyzer.analyze.return_value = [
            AspectSentiment("food", "positive", 0.9, (0, 4))
        ]
        return analyzer

    def test_batch_processing(self, mock_analyzer):
        texts = ["Text 1", "Text 2", "Text 3"]
        results = ABSAAnalyzer.analyze_batch(mock_analyzer, texts)

        assert len(results) == 3
        assert mock_analyzer.analyze.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])