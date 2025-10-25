from src.transformer_absa import TransformerABSA

analyzer = TransformerABSA()

text = "The hotel room was nice but the breakfast was disappointing."

# Extract aspects only
aspects = analyzer.extract_aspects(text)
print("Aspects:", aspects)

# Classify sentiment for a single aspect
sentiment = analyzer.classify_sentiment(text, "hotel room")
print("Hotel room sentiment:", sentiment)

# Full analysis
results = analyzer.analyze(text)
print("Full ABSA results:", results)

# Expected output:
# Aspects: ['hotel room', 'breakfast']
# Pizza sentiment: positive
# Full ABSA results: [AspectSentiment(aspect='hotel room', sentiment='positive', confidence=0.5279850363731384,
# text_span=(4, 14)), AspectSentiment(aspect='breakfast', sentiment='negative', confidence=0.8696303963661194, text_span=(32, 41))]
