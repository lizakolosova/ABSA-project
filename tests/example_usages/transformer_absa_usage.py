from src.transformer_absa import TransformerABSA

analyzer = TransformerABSA()

text = "The hotel room was nice but the breakfast was disappointing."

# Extract aspects only
aspects = analyzer.extract_aspects(text)
print("Aspects:", aspects)

# Classify sentiment for a single aspect
sentiment = analyzer.classify_sentiment(text, "pizza")
print("Pizza sentiment:", sentiment)

# Full analysis
results = analyzer.analyze(text)
print("Full ABSA results:", results)

# Expected output:
# Aspects: ['pizza', 'service']
# Pizza sentiment: positive
# Full ABSA results: [('pizza', 'positive'), ('service', 'negative')]
