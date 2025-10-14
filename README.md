# data_6_llm_project
## Project overview

The project implements Aspect-Based Sentiment Analysis (ABSA) in three different ways:

_1. Transformer-based ABSA (TransformerABSA)_

Uses Hugging Face transformers for aspect extraction and sentiment classification.

Offers decent accuracy leveraging pretrained models for both tasks.

_2. Lexicon-based ABSA (LexiconABSA)_

Uses rule-based extraction with SpaCy noun chunks and modifiers.

Uses VADER lexicon for sentiment scoring.

Good for low-resource setups and easy-to-understand sentiment.

_3. LLM-based ABSA (LLMABSA)_
Uses a large language model to extract aspects and determine their sentiment in one step.
Works well with complex text and can provide context-aware, explainable sentiment results.


## Installation setup
##### git clone https://gitlab.com/lzkolosova/data_6_llm_project.git
##### cd data_6_llm_project
##### pip install -r requirements.txt
##### python -m spacy download en_core_web_trf
##### pip install ollama
##### ollama pull llama3

## Usage Examples
**You can find and run it in tests/example_usages**
### 1. LexiconABSA

```python
from src.lexicon_absa import LexiconABSA

# Initialize analyzer
analyzer = LexiconABSA()

# Example text
text = "The hotel room was spacious but the breakfast was disappointing."

# Full analysis
aspect_sentiments = analyzer.analyze(text)

for aspect in aspect_sentiments:
    print(f"Aspect: {aspect.aspect}, Sentiment: {aspect.sentiment}, Confidence: {aspect.confidence}")
```

_Aspect: room, Sentiment: positive, Confidence: 0.5
Aspect: breakfast, Sentiment: negative, Confidence: 0.5_


### 2. TransformerABSA

```python
from src.transformer_absa import TransformerABSA

# Initialize analyzer (CPU by default)
analyzer = TransformerABSA()

# Example text
text = "The pizza was delicious but the service was slow."

# Extract aspects only
aspects = analyzer.extract_aspects(text)
print("Aspects:", aspects)

# Classify sentiment for a single aspect
sentiment = analyzer.classify_sentiment(text, "pizza")
print("Pizza sentiment:", sentiment)

# Full analysis
results = analyzer.analyze(text)
print("Full ABSA results:", results)
```

_Aspects: ['pizza', 'service']
Pizza sentiment: positive
Full ABSA results: [('pizza', 'positive'), ('service', 'negative')]_

### 3. LLMABSA
```python
from src.llm_absa import LLMABSA

if __name__ == "__main__":
    analyzer = LLMABSA(model="llama3")  # or "mistral" depends on what I pull
    text = "The laptop has a great screen but terrible battery life."
    results = analyzer.analyze(text)
    for r in results:
        print(r)
```
_AspectSentiment(aspect='screen', sentiment='positive', confidence=0.95, text_span=(8, 12))
AspectSentiment(aspect='battery life', sentiment='negative', confidence=0.93, text_span=(17, 29))_

## Design decisions and rationale
### Lexicon-based:
Uses VaderSentiment like it was asked in the assignment.
A lot of checks because the implementation itself is based on spacy - as we need to make sure that the meaningful part is extracted properly
Normalization to remove determiners, to lowercase the words etc
Handles negation both for adjectives and verbs
Added intensifiers/diminishers to modify sentiment scores.
Can be extended easily
### Transformer-based:
Uses pretrained models
Aspect extraction and sentiment classification are handled by two specialized models, allowing independent improvements and better fine-tuning
Extracted aspect tokens are also normalized and validated
Supports both CPU and GPU inference transparently(I have CPU but maybe someone has GPU who knows)
Can easily switch pretrained models or update the label mapping to support new languages, domains, or sentiment categories.
### LLM-based:
Uses Local llama3 large language model accessed via Ollama to perform aspect extraction and sentiment classification in one step.
It can handle complex and nuanced text better than rule-based or transformer-only approaches, providing context-aware and explainable results. 
However, it requires significant computational resources and longer inference times, which can limit its practicality for real-time applications. 
This approach benefits from easy adaptability to new domains or languages through prompt adjustments rather than retraining.
### API design:
ABSAAnalyzer: Abstract base class that specifies the analyze(text: str) method, returning a list of AspectSentiment.
AspectSentiment: Standardized data class containing the aspect, sentiment, confidence score, and text span.
This design allows multiple ABSA implementations (rule-based, transformer-based, or LLM-based) to be changed with each other, ensuring consistent input/output formats and easy integration into some pipelines.