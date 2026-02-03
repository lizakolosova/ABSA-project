from typing import List, Optional, Dict
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .base import ABSAAnalyzer, AspectSentiment
from .model_cache import ModelCache
from .text_utils import (
    is_valid_aspect, split_aspect_candidates,
    find_aspect_root, has_negation
)
from .sentiment_utils import (
    compute_intensity_modifier, average_conjunct_sentiments,
    aggregate_scores, create_aspect_entry
)
from .logging_config import get_logger

logger = get_logger(__name__)


class LexiconABSA(ABSAAnalyzer):
    SKIP_PRONOUNS = {"which", "that", "this", "it", "they", "he", "she", "we", "you"}

    INTENSIFIERS = {
        "very": 0.3, "extremely": 0.5, "incredibly": 0.5, "absolutely": 0.4,
        "really": 0.3, "totally": 0.4, "completely": 0.5, "utterly": 0.5,
        "quite": 0.2, "pretty": 0.2, "rather": 0.2, "fairly": 0.15,
        "too": 0.3, "so": 0.3, "highly": 0.4, "exceptionally": 0.5
    }
    
    DIMINISHERS = {
        "slightly": -0.3, "somewhat": -0.2, "barely": -0.4, "hardly": -0.5,
        "scarcely": -0.4, "marginally": -0.3, "a bit": -0.2, "a little": -0.2,
        "kind of": -0.3, "sort of": -0.3
    }
    
    def __init__(
        self,
        model_name: str = "en_core_web_trf",
        max_text_length: int = 100_000,
        intensifiers: Optional[Dict[str, float]] = None,
        diminishers: Optional[Dict[str, float]] = None
    ):
        self.model_name = model_name
        self.max_text_length = max_text_length
        self.intensifiers = intensifiers or self.INTENSIFIERS
        self.diminishers = diminishers or self.DIMINISHERS

        self.nlp = ModelCache.get(model_name)
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("LexiconABSA initialized")
    
    def _get_sentiment_score(self, token, check_negation: bool = True) -> float:
        try:
            phrase_tokens = [t.text for t in token.lefts if t.dep_ == "advmod"] + [token.text]
            phrase = " ".join(phrase_tokens)

            score = self.analyzer.polarity_scores(phrase)["compound"]

            score *= compute_intensity_modifier(token, self.intensifiers, self.diminishers)

            score = average_conjunct_sentiments(token, self.analyzer)

            if check_negation and has_negation(token):
                score = -score
            
            return score
        except Exception as e:
            logger.warning(f"Error computing sentiment: {e}")
            return 0.0
    
    def analyze(self, text: str) -> List[AspectSentiment]:
        start_time = time.time()
        
        try:
            self.validate_input(text, self.max_text_length)
            
            doc = self.nlp(text)
            aspect_dict = {}
            seen_aspects = set()

            for chunk in doc.noun_chunks:
                if chunk.root.pos_ == "PRON":
                    continue
                
                aspect_candidate = chunk.root.text
                for candidate in split_aspect_candidates(aspect_candidate):
                    if is_valid_aspect(candidate, self.nlp) and candidate.lower() not in self.SKIP_PRONOUNS:
                        if candidate.lower() not in seen_aspects:
                            seen_aspects.add(candidate.lower())
                            span = (chunk.start_char, chunk.end_char)
                            create_aspect_entry(candidate, span, aspect_dict)

            for chunk in doc.noun_chunks:
                aspect = find_aspect_root(chunk)
                if aspect in self.SKIP_PRONOUNS:
                    continue
                
                for child in chunk.root.children:
                    if child.dep_ == "amod" and child.pos_ == "ADJ":
                        score = self._get_sentiment_score(child)
                        span = (chunk.start_char, chunk.end_char)
                        create_aspect_entry(aspect, span, aspect_dict)
                        aspect_dict[aspect]["scores"].append(score)

            for token in doc:
                if token.pos_ == "ADJ" and token.dep_ in {"acomp", "attr"}:
                    head = token.head
                    
                    for subj in (t for t in head.children if t.dep_ in {"nsubj", "nsubjpass"}):
                        for chunk in doc.noun_chunks:
                            if subj in chunk:
                                aspect = find_aspect_root(chunk)
                                if aspect in self.SKIP_PRONOUNS:
                                    continue
                                
                                score = self._get_sentiment_score(token, check_negation=False)
                                if has_negation(head):
                                    score = -score * 0.8
                                
                                span = (chunk.start_char, chunk.end_char)
                                create_aspect_entry(aspect, span, aspect_dict)
                                aspect_dict[aspect]["scores"].append(score)

            for token in doc:
                if token.pos_ == "VERB" and token.lemma_ not in {"be", "have", "do"}:
                    verb_group = [token] + [t for t in token.conjuncts if t.pos_ == "VERB"]
                    
                    for verb in verb_group:
                        verb_score = self.analyzer.polarity_scores(verb.text)["compound"]
                        
                        if abs(verb_score) > 0.1:
                            for child in verb.children:
                                if child.dep_ in {"dobj", "pobj", "nsubj"}:
                                    for chunk in doc.noun_chunks:
                                        if child in chunk:
                                            aspect = find_aspect_root(chunk)
                                            if aspect in self.SKIP_PRONOUNS:
                                                continue
                                            
                                            score = verb_score
                                            if has_negation(verb):
                                                score = -score * 0.8
                                            
                                            span = (chunk.start_char, chunk.end_char)
                                            create_aspect_entry(aspect, span, aspect_dict)
                                            aspect_dict[aspect]["scores"].append(score)

            results = []
            for aspect_text, data in aspect_dict.items():
                if not data["scores"]:
                    continue
                
                sentiment_label, confidence = aggregate_scores(data["scores"])
                results.append(
                    AspectSentiment(
                        aspect=aspect_text,
                        sentiment=sentiment_label,
                        confidence=confidence,
                        text_span=data["text_span"],
                        model_name="LexiconABSA"
                    )
                )
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Found {len(results)} aspects in {elapsed:.0f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to analyze: {e}") from e