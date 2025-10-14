# Implementation 1

from typing import List, Tuple
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .base import ABSAAnalyzer, AspectSentiment
from .utils import (
    find_aspect_root,
    get_or_create_aspect,
    compute_intensity_modifier,
    average_conjunct_sentiments,
    aggregate_sentiment_scores,
    has_phrase_negation,
    _split_candidates,
    _is_valid_candidate
)

DETERMINERS = {"this", "that", "these", "those", "a", "an", "the"}


def _normalize_candidate(text: str) -> str:
    """
    Normalize a candidate aspect by removing leading determiners.

    Args:
        text (str): The candidate aspect text.

    Returns:
        str: Normalized candidate aspect.
    """
    words = text.strip().split()
    while words and words[0].lower() in DETERMINERS:
        words.pop(0)
    return " ".join(words)


class LexiconABSA(ABSAAnalyzer):
    """
    Rule-based Aspect-Based Sentiment Analyzer using lexicon methods
    and VADER sentiment scoring.
    """

    PRONOUNS_TO_SKIP = {"which", "that", "this", "it", "they", "he", "she", "we", "you"}

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

    def __init__(self):
        """
        Initialize the LexiconABSA analyzer by loading SpaCy NLP pipeline
        and the VADER sentiment analyzer.
        """
        self.nlp = spacy.load("en_core_web_trf")
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_score(self, token, check_negation: bool = True) -> float:
        """
        Compute the sentiment score of a token considering modifiers and negation.

        Args:
            token: spaCy token to evaluate.
            check_negation (bool): Whether to check for negation.

        Returns:
            float: Sentiment score in range [-1, 1].
        """
        phrase_tokens = [t.text for t in token.lefts if t.dep_ == "advmod"] + [token.text]
        phrase = " ".join(phrase_tokens)
        score = self.analyzer.polarity_scores(phrase)["compound"]
        score *= compute_intensity_modifier(token, self.INTENSIFIERS, self.DIMINISHERS)
        score = average_conjunct_sentiments(token, self.analyzer)

        if check_negation and has_phrase_negation(token):
            score = -score
        return score

    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Perform aspect extraction and sentiment scoring on input text.

        Args:
            text (str): Input text to analyze.

        Returns:
            List[AspectSentiment]: List of aspects with sentiment and confidence.
        """
        doc = self.nlp(text)
        aspect_dict = {}
        seen_aspects = set()

        # Step 1: Extract candidate aspects from noun chunks
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == "PRON":
                continue
            aspect_candidate = chunk.root.text
            for c in _split_candidates(_normalize_candidate(aspect_candidate)):
                if _is_valid_candidate(c, self.nlp) and c.lower() not in self.PRONOUNS_TO_SKIP:
                    if c.lower() not in seen_aspects:
                        seen_aspects.add(c.lower())
                        span = (chunk.start_char, chunk.end_char)
                        get_or_create_aspect(aspect_dict, c, span)

        # Step 2: Score adjectives modifying aspects
        for chunk in doc.noun_chunks:
            aspect = find_aspect_root(chunk)
            if aspect in self.PRONOUNS_TO_SKIP:
                continue
            for child in chunk.root.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    score = self.get_sentiment_score(child)
                    span = (chunk.start_char, chunk.end_char)
                    get_or_create_aspect(aspect_dict, aspect, span)
                    aspect_dict[aspect]["scores"].append(score)

        # Step 3: Score standalone adjectives
        for token in doc:
            if token.pos_ == "ADJ" and token.dep_ in {"acomp", "attr"}:
                head = token.head
                for subj in [t for t in head.children if t.dep_ in {"nsubj", "nsubjpass"}]:
                    for chunk in doc.noun_chunks:
                        if subj in chunk:
                            aspect = find_aspect_root(chunk)
                            if aspect in self.PRONOUNS_TO_SKIP:
                                continue
                            score = self.get_sentiment_score(token, check_negation=False)
                            if has_phrase_negation(head):
                                score = -score * 0.8
                            span = (chunk.start_char, chunk.end_char)
                            get_or_create_aspect(aspect_dict, aspect, span)
                            aspect_dict[aspect]["scores"].append(score)

        # Step 4: Score verbs related to aspects
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
                                        if aspect in self.PRONOUNS_TO_SKIP:
                                            continue
                                        score = verb_score
                                        if has_phrase_negation(verb):
                                            score = -score * 0.8
                                        span = (chunk.start_char, chunk.end_char)
                                        get_or_create_aspect(aspect_dict, aspect, span)
                                        aspect_dict[aspect]["scores"].append(score)

        # Step 5: Aggregate sentiment scores per aspect
        aspect_sentiments: List[AspectSentiment] = []
        for aspect_text, data in aspect_dict.items():
            if not data["scores"]:
                continue
            sentiment_label, confidence = aggregate_sentiment_scores(data["scores"])
            aspect_sentiments.append(
                AspectSentiment(
                    aspect=aspect_text,
                    sentiment=sentiment_label,
                    confidence=confidence,
                    text_span=data["text_span"]
                )
            )

        return aspect_sentiments
