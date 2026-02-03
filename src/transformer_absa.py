import string
from typing import List, Optional
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from .base import AspectSentiment, ABSAAnalyzer
from .text_utils import normalize_aspect
from .logging_config import get_logger

logger = get_logger(__name__)


class TransformerABSA(ABSAAnalyzer):
    _model_cache = {}
    
    def __init__(
        self,
        aspect_model_name: str = "gauneg/roberta-base-absa-ate-sentiment",
        sentiment_model_name: str = "yangheng/deberta-v3-base-absa-v1.1",
        device: int = -1,
        max_length: int = 512
    ):
        self.aspect_model_name = aspect_model_name
        self.sentiment_model_name = sentiment_model_name
        self.max_length = max_length
        self.device = self._configure_device(device)
        
        try:
            self._load_models()
            logger.info(f"TransformerABSA initialized on {self.device}")
        except ImportError as e:
            if "keras" in str(e).lower():
                logger.error("TensorFlow/Keras compatibility issue detected")
                logger.info("Install: pip install transformers torch --no-deps")
                logger.info("Then: pip install tokenizers safetensors huggingface-hub")
            raise RuntimeError(f"Failed to load models. Try: pip install torch transformers --upgrade") from e
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    def _configure_device(self, device: int) -> str:
        if device >= 0:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU")
                return "cpu"
            return f"cuda:{device}"
        return "cpu"
    
    def _load_models(self):
        cache_key = (self.aspect_model_name, self.sentiment_model_name, self.device)
        
        if cache_key in self._model_cache:
            logger.info("Loading models from cache")
            cached = self._model_cache[cache_key]
            self.aspect_extractor = cached["aspect_extractor"]
            self.tokenizer = cached["tokenizer"]
            self.sentiment_model = cached["sentiment_model"]
            self.label_map = cached["label_map"]
            return
        
        logger.info("Loading models...")
        
        self.aspect_extractor = pipeline(
            task="token-classification",
            model=self.aspect_model_name,
            aggregation_strategy="simple",
            device=self.device if self.device != "cpu" else -1
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            self.sentiment_model_name
        )
        self.sentiment_model.eval()
        
        if self.device != "cpu":
            self.sentiment_model.to(self.device)
        
        self.label_map = ["negative", "neutral", "positive"]
        
        self._model_cache[cache_key] = {
            "aspect_extractor": self.aspect_extractor,
            "tokenizer": self.tokenizer,
            "sentiment_model": self.sentiment_model,
            "label_map": self.label_map
        }
    
    def extract_aspects(self, text: str) -> List[str]:
        try:
            results = self.aspect_extractor(text)
            aspects = []
            
            for r in results:
                word = r["word"].strip()
                if r["entity_group"] != "O" and word and word not in string.punctuation:
                    aspects.append(normalize_aspect(word))
            
            return aspects
        except Exception as e:
            logger.error(f"Aspect extraction failed: {e}")
            return []
    
    def classify_sentiment(self, text: str, aspect: str) -> str:
        try:
            inputs = self.tokenizer(
                text, aspect,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.sentiment_model(**inputs).logits
                pred_index = torch.argmax(logits, dim=1).item()
            
            return self.label_map[pred_index]
        except Exception as e:
            logger.error(f"Sentiment classification failed: {e}")
            return "neutral"
    
    def analyze(self, text: str) -> List[AspectSentiment]:
        start_time = time.time()
        
        try:
            self.validate_input(text)
            
            aspects_data = self.aspect_extractor(text)
            results = []
            
            for item in aspects_data:
                word = item["word"].strip()
                if item["entity_group"] == "O" or not word or word in string.punctuation:
                    continue
                
                aspect = normalize_aspect(word)
                confidence = float(item.get("score", 0.0))
                text_span = (item.get("start", 0), item.get("end", 0))
                
                inputs = self.tokenizer(
                    text, aspect,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = self.sentiment_model(**inputs).logits
                    pred_index = torch.argmax(logits, dim=1).item()
                    sentiment = self.label_map[pred_index]
                
                results.append(
                    AspectSentiment(
                        aspect=aspect,
                        sentiment=sentiment,
                        confidence=confidence,
                        text_span=text_span,
                        model_name="TransformerABSA"
                    )
                )
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Found {len(results)} aspects in {elapsed:.0f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to analyze: {e}") from e
    
    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[List[AspectSentiment]]:
        logger.info(f"Batch analyzing {len(texts)} texts")
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                try:
                    result = self.analyze(text)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to analyze text: {e}")
                    results.append([])
        
        return results