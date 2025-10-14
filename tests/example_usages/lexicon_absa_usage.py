from src.lexicon_absa import LexiconABSA

# Initialize analyzer
analyzer = LexiconABSA()

# Example text
text = "The hotel room was nice but the breakfast was disappointing."

# Full analysis
aspect_sentiments = analyzer.analyze(text)

for aspect in aspect_sentiments:
    print(f"Aspect: {aspect.aspect}, Sentiment: {aspect.sentiment}, Confidence: {aspect.confidence}")

# Expected output:
# Aspect: hotel room, Sentiment: positive, Confidence: 0.5
# Aspect: breakfast, Sentiment: negative, Confidence: 0.5
