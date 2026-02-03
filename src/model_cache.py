from typing import Dict
import spacy
from .logging_config import get_logger

logger = get_logger(__name__)


class ModelCache:
    _cache: Dict[str, spacy.Language] = {}

    @classmethod
    def get(cls, model_name: str) -> spacy.Language:
        if model_name not in cls._cache:
            logger.info(f"Loading spaCy model: {model_name}")
            try:
                cls._cache[model_name] = spacy.load(model_name)
            except OSError as e:
                raise RuntimeError(
                    f"Could not load spaCy model '{model_name}'. "
                    f"Install with: python -m spacy download {model_name}"
                ) from e
        return cls._cache[model_name]
    
    @classmethod
    def clear(cls):
        cls._cache.clear()
        logger.info("Model cache cleared")
