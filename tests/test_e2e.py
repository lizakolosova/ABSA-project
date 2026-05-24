"""
Smoke tests for all three ABSA approaches.

Goal: each analyzer can be imported, initialized, and called on a canonical
input without raising an exception, and the output has the correct structure.
No real models are loaded — all heavy dependencies are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import torch

from src.base import AspectSentiment

CANONICAL = "The food was excellent but the service was slow."


# ─── LexiconABSA ─────────────────────────────────────────────────────────────

class TestLexiconABSASmoke:
    @pytest.fixture
    def analyzer(self):
        mock_doc = MagicMock()
        mock_doc.noun_chunks = []
        mock_doc.__iter__ = lambda self: iter([])
        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("src.model_cache.ModelCache.get", return_value=mock_nlp):
            from src.lexicon_absa import LexiconABSA
            return LexiconABSA()

    def test_imports_and_initializes(self, analyzer):
        from src.lexicon_absa import LexiconABSA
        assert isinstance(analyzer, LexiconABSA)

    def test_analyze_returns_list(self, analyzer):
        result = analyzer.analyze(CANONICAL)
        assert isinstance(result, list)

    def test_output_structure(self, analyzer):
        for item in analyzer.analyze(CANONICAL):
            assert isinstance(item, AspectSentiment)
            assert item.sentiment in {"positive", "negative", "neutral"}
            assert 0.0 <= item.confidence <= 1.0
            assert isinstance(item.text_span, tuple)
            assert len(item.text_span) == 2

    def test_empty_input_raises(self, analyzer):
        with pytest.raises((ValueError, RuntimeError)):
            analyzer.analyze("")


# ─── TransformerABSA ──────────────────────────────────────────────────────────

class TestTransformerABSASmoke:
    @pytest.fixture
    def analyzer(self):
        with patch("src.transformer_absa.pipeline"), \
             patch("src.transformer_absa.AutoTokenizer"), \
             patch("src.transformer_absa.AutoModelForSequenceClassification"):
            from src.transformer_absa import TransformerABSA
            inst = TransformerABSA.__new__(TransformerABSA)

        inst.aspect_extractor = MagicMock(return_value=[
            {"word": "food",    "entity_group": "ASP", "score": 0.95,
             "start": 4,  "end": 8},
            {"word": "service", "entity_group": "ASP", "score": 0.88,
             "start": 29, "end": 36},
        ])

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids":      torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        inst.tokenizer = mock_tokenizer

        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 0.7]])   # → positive
        mock_model = MagicMock(return_value=mock_output)
        inst.sentiment_model = mock_model

        inst.label_map = ["negative", "neutral", "positive"]
        inst.device = "cpu"
        inst.max_length = 512
        return inst

    def test_analyze_returns_list(self, analyzer):
        assert isinstance(analyzer.analyze(CANONICAL), list)

    def test_finds_expected_aspects(self, analyzer):
        result = analyzer.analyze(CANONICAL)
        assert len(result) == 2

    def test_output_structure(self, analyzer):
        for item in analyzer.analyze(CANONICAL):
            assert isinstance(item, AspectSentiment)
            assert item.sentiment in {"positive", "negative", "neutral"}
            assert 0.0 <= item.confidence <= 1.0
            assert isinstance(item.text_span, tuple)
            assert len(item.text_span) == 2
            assert item.model_name == "TransformerABSA"

    def test_empty_input_raises(self, analyzer):
        with pytest.raises((ValueError, RuntimeError)):
            analyzer.analyze("")


# ─── LLMABSA ─────────────────────────────────────────────────────────────────

_LLM_JSON = (
    '[{"aspect": "food",    "sentiment": "positive", "confidence": 0.95, "text_span": [4,  8]},'
    ' {"aspect": "service", "sentiment": "negative", "confidence": 0.88, "text_span": [29, 36]}]'
)


class TestLLMABSASmoke:
    @pytest.fixture
    def analyzer(self):
        with patch("src.llm_absa.ollama.list") as mock_list:
            mock_list.return_value = {"models": [{"name": "llama3"}]}
            from src.llm_absa import LLMABSA
            return LLMABSA(model="llama3")

    def test_analyze_returns_list(self, analyzer):
        with patch("src.llm_absa.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": _LLM_JSON}}
            assert isinstance(analyzer.analyze(CANONICAL), list)

    def test_finds_expected_aspects(self, analyzer):
        with patch("src.llm_absa.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": _LLM_JSON}}
            result = analyzer.analyze(CANONICAL)
        assert len(result) == 2

    def test_output_structure(self, analyzer):
        with patch("src.llm_absa.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": _LLM_JSON}}
            result = analyzer.analyze(CANONICAL)
        for item in result:
            assert isinstance(item, AspectSentiment)
            assert item.sentiment in {"positive", "negative", "neutral"}
            assert 0.0 <= item.confidence <= 1.0
            assert isinstance(item.text_span, tuple)
            assert len(item.text_span) == 2

    def test_empty_input_returns_empty(self, analyzer):
        # LLMABSA catches validation errors internally and returns [] rather than re-raising
        result = analyzer.analyze("")
        assert result == []

    def test_bad_llm_json_returns_empty(self, analyzer):
        analyzer.cache = None
        with patch("src.llm_absa.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "not json at all"}}
            result = analyzer.analyze(CANONICAL)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
