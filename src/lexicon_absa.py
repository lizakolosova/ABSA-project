# Implementation 1
from typing import List

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.base import ABSAAnalyzer, AspectSentiment

class LexiconABSA(ABSAAnalyzer):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.analyzer = SentimentIntensityAnalyzer()

    @staticmethod
    def convert_score_to_label(score: float) -> str:
        if score >= 0.05:
            return "POSITIVE"
        elif score <= -0.05:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

#the method to check if the token(part of the text) has "negation" (for example not good means negative, not positive etc)
    @staticmethod
    def has_negation(token):
        for child in token.children:
            if child.dep_ == 'neg':
                return True
        for child in token.head.children:
            if child.dep_ == 'neg':
                return True
        return False

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)
        aspect_sentiments = []

        for chunk in doc.noun_chunks:
            aspect = chunk.text
            sentiments = []

            #chunk.root is the main noun of the noun phrase and .childen are the related to this noun.
            for child in chunk.root.children:
                # it check if the related words is adjectival modifier (amod) and it's adjective (adj)
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    # so here we check if we have aby adverbial modifiers("advmod", like "very")
                    phrase_tokens = [t.text for t in child.lefts if t.dep_ == "advmod"] + [child.text]
                    phrase = " ".join(phrase_tokens)
                    # here we use VADER to get a score (from -1 to 1)
                    score = self.analyzer.polarity_scores(phrase)["compound"]
                    # here if we have negation then it will make the score opposite
                    if self.has_negation(child):
                        score = -score
                    sentiments.append(score)

            for token in doc:
                # check if the token is adjective or adjectival complement(acomp) example - food is good, good is acomp here. "is" is head here
                if token.pos_ == "ADJ" and token.dep_ == "acomp":
                    head = token.head # head is a linking verb
                    for child in head.children:
                        # nsubj is subject
                        if child.dep_ == "nsubj" and child.lemma_.lower() in chunk.lemma_.lower():
                            phrase_tokens = [t.text for t in token.lefts if t.dep_ == "advmod"] + [token.text]
                            phrase = " ".join(phrase_tokens)
                            score = self.analyzer.polarity_scores(phrase)["compound"]
                            if self.has_negation(token) or self.has_negation(head):
                                score = -score
                            sentiments.append(score)

            avg_score = sum(sentiments) / len(sentiments) if sentiments else 0
            sentiment_label = self.convert_score_to_label(avg_score)
            confidence = abs(avg_score)

            aspect_sentiments.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment_label,
                    confidence=confidence,
                    text_span=(chunk.start_char, chunk.end_char)
                )
            )

        return aspect_sentiments

if __name__ == "__main__":
    analyzer = LexiconABSA()
    text = "The pasta was perfectly cooked, but the service was not good."
    results = analyzer.analyze(text)
    for res in results:
        print(f"Aspect: '{res.aspect}', Sentiment: {res.sentiment}, Confidence: {res.confidence:.2f}, Span: {res.text_span}") # the confidence is okay for this one
