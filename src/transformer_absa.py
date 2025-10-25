from typing import List, Tuple
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from .utils import normalize_aspect, is_valid_aspect_token


class TransformerABSA:
    """
    Aspect-Based Sentiment Analyzer using transformer models.

    This class performs:
        1. Aspect extraction using a token-classification transformer.
        2. Sentiment classification for each extracted aspect using a sequence classification transformer.

    Attributes:
        aspect_extractor: Hugging Face pipeline for aspect extraction.
        tokenizer: Tokenizer for sentiment classification model.
        sentiment_model: Transformer model for sentiment classification.
        device: Torch device ("cpu" or GPU index).
        label_map: Mapping of model outputs to sentiment labels.
    """

    def __init__(
        self,
        aspect_model_name: str = "gauneg/roberta-base-absa-ate-sentiment",
        sentiment_model_name: str = "yangheng/deberta-v3-base-absa-v1.1",
        device: int = -1,
    ):
        """
        Initialize TransformerABSA with specified models and device.

        Args:
            aspect_model_name (str): Pretrained model for aspect extraction.
            sentiment_model_name (str): Pretrained model for sentiment classification.
            device (int): Torch device index. -1 for CPU, >=0 for GPU.
        """
        self.device = device if device >= 0 else "cpu"

        # Initialize aspect extraction pipeline
        self.aspect_extractor = pipeline(
            task="token-classification",
            model=aspect_model_name,
            aggregation_strategy="simple",
            device=device
        )

        # Load sentiment classification model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        self.sentiment_model.eval()

        if device >= 0 and torch.cuda.is_available():
            self.sentiment_model.to(device)

        # Map model logits to sentiment labels
        self.label_map: List[str] = ["negative", "neutral", "positive"]

    def extract_aspects(self, text: str) -> List[str]:
        """
        Extract aspect terms from text using the transformer aspect extractor.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of normalized aspect terms.
        """
        results = self.aspect_extractor(text)
        aspects: List[str] = []

        for r in results:
            word = r["word"].strip()
            if r["entity_group"] != "O" and is_valid_aspect_token(word):
                aspects.append(normalize_aspect(word))

        return aspects

    def classify_sentiment(self, text: str, aspect: str) -> str:
        """
        Classify sentiment for a specific aspect in the text.

        Args:
            text (str): Input text.
            aspect (str): Aspect term to classify sentiment for.

        Returns:
            str: Sentiment label ("negative", "neutral", or "positive").
        """
        # Encode text and aspect for the sentiment model
        inputs = self.tokenizer(f"[CLS] {text} [SEP] {aspect} [SEP]", return_tensors="pt")

        # Move tensors to device if using GPU
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.sentiment_model(**inputs).logits
            pred_index = torch.argmax(logits, dim=1).item()

        return self.label_map[pred_index]

    def analyze(self, text: str) -> List[Tuple[str, str]]:
        """
        Perform end-to-end ABSA analysis: extract aspects and classify sentiment.

        Args:
            text (str): Input text.

        Returns:
            List[Tuple[str, str]]: List of tuples (aspect, sentiment).
        """
        aspects = self.extract_aspects(text)
        results: List[Tuple[str, str]] = []

        for aspect in aspects:
            sentiment = self.classify_sentiment(text, aspect)
            results.append((aspect, sentiment))

        return results