# Aspect-Based Sentiment Analysis (ABSA) Comparison Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![spaCy](https://img.shields.io/badge/spaCy-3.0%2B-09a3d5.svg)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-orange.svg)](https://ollama.ai/)

> A comprehensive comparison of three different approaches to Aspect-Based Sentiment Analysis: Rule-based (Lexicon), Transformer-based, and Large Language Model approaches.

## Overview

This project implements and compares **three distinct approaches** to Aspect-Based Sentiment Analysis on restaurant review data. Each method has unique strengths and trade-offs in terms of accuracy, speed, and interpretability.

The goal is to extract specific aspects (e.g., "food", "service", "ambiance") from text and determine the sentiment (positive, negative, neutral) expressed toward each aspect.

## What is ABSA?

**Aspect-Based Sentiment Analysis (ABSA)** goes beyond traditional sentiment analysis by identifying:
1. **Aspects**: Specific features or attributes mentioned in text
2. **Sentiment**: The opinion expressed toward each aspect

### Example

**Input Text:**
> "The pizza was delicious but the service was terrible."

**ABSA Output:**
- Aspect: "pizza" → Sentiment: **positive** ✅
- Aspect: "service" → Sentiment: **negative** ❌

## Approaches Implemented

### 1. LexiconABSA (Rule-Based)

**Method**: Uses linguistic rules, dependency parsing, and the VADER sentiment lexicon.

**How it works**:
- Extracts aspects from noun chunks using spaCy's dependency parser
- Identifies sentiment-bearing adjectives and verbs related to aspects
- Applies rule-based sentiment scoring with VADER
- Handles negations, intensifiers, and diminishers

**Strengths**:
- **Fastest** inference time (~0.02s per review)
- Highly interpretable and explainable
- No training required
- Minimal resource requirements

**Limitations**:
- Limited vocabulary coverage
- Struggles with implicit aspects and complex sentence structures
- Misses nuanced or sarcastic language

---

### 2. TransformerABSA (Deep Learning)

**Method**: Uses pre-trained transformer models for both aspect extraction and sentiment classification.

**Models Used**:
- **Aspect Extraction**: `roberta-base-absa-ate-sentiment` (Token Classification)
- **Sentiment Analysis**: `deberta-v3-base-absa-v1.1` (Sequence Classification)

**How it works**:
- Fine-tuned RoBERTa identifies aspect terms using token classification
- DeBERTa classifies sentiment for each aspect-text pair
- Leverages contextual embeddings for nuanced understanding

**Strengths**:
- Balanced accuracy and speed (~0.5s per review)
- Better context understanding than rule-based methods
- Handles complex syntax and implicit sentiment
- Pre-trained on ABSA-specific data

**Limitations**:
- Requires GPU for optimal performance
- Black-box model with limited interpretability
- Needs significant computational resources

---

### 3. LLMABSA (Large Language Model)

**Method**: Uses Ollama with Llama3 for zero-shot ABSA via prompt engineering.

**How it works**:
- Sends structured prompts with few-shot examples to Llama3
- LLM identifies aspects and sentiments using natural language understanding
- Parses JSON-formatted responses

**Strengths**:
- **Highest accuracy** (F1-score: 0.89)
- Excellent contextual understanding
- Handles sarcasm, implicit aspects, and complex language
- No fine-tuning required

**Limitations**:
- **Slowest** inference (~10s per review)
- Requires powerful hardware or API access
- Non-deterministic outputs
- Depends on external LLM service (Ollama)

### Key Findings

1. **LLM achieves the best overall performance** with highest F1-score and accuracy
2. **TransformerABSA offers the best speed-accuracy tradeoff** for production use
3. **LexiconABSA is ideal for real-time applications** where speed is critical
4. **All models achieve high precision** (~97-98%), meaning predictions are reliable when made
5. **Recall varies significantly**: LexiconABSA misses ~40% of aspects due to limited vocabulary

### Confusion Matrix Insights

- **LexiconABSA**: Often misclassifies sentiment as "neutral" when uncertain
- **TransformerABSA**: Balanced confusion matrix with occasional neutral bias
- **LLMABSA**: Rarely predicts "neutral" (~2% of predictions), showing confidence in sentiment classification

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/lizakolosova/ABSA-project.git
cd ABSA-project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy Model

```bash
python -m spacy download en_core_web_trf
```

### Step 5: Install Ollama (for LLM approach)

Follow instructions at [ollama.ai](https://ollama.ai/) to install Ollama, then:

```bash
ollama pull llama3
```

## Usage

### Basic Usage

```python
from src.lexicon_absa import LexiconABSA
from src.transformer_absa import TransformerABSA
from src.llm_absa import LLMABSA


analyzer = LexiconABSA() 
# analyzer = TransformerABSA(device=0)
# analyzer = LLMABSA(model="llama3")

text = "The food was amazing but the service was slow."
results = analyzer.analyze(text)

for result in results:
    print(f"Aspect: {result.aspect}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Text span: {result.text_span}")
    print("---")
```

### Output Example

```
Aspect: food
Sentiment: positive
Confidence: 0.92
Text span: (4, 8)
---
Aspect: service
Sentiment: negative
Confidence: 0.88
Text span: (29, 36)
---
```

### Advanced Usage

#### Use TransformerABSA with GPU

```python
# Use GPU device 0
analyzer = TransformerABSA(
    aspect_model_name="gauneg/roberta-base-absa-ate-sentiment",
    sentiment_model_name="yangheng/deberta-v3-base-absa-v1.1",
    device=0  # Use -1 for CPU
)
```

#### Customize LLM Model

```python
analyzer = LLMABSA(model="llama3")  # or "mistral", "mixtral", etc.
```

#### Extract Only Aspects

```python
from src.transformer_absa import TransformerABSA

analyzer = TransformerABSA()
aspects = analyzer.extract_aspects("The battery life is great!")
print(aspects)  # ['battery life']
```

## Performance Analysis

### Aspect Detection Analysis

**Insights**:
- LexiconABSA has **highest precision** but misses many aspects
- Transformer and LLM models detect more aspects but occasionally hallucinate
- LLM provides best balance of recall and precision

### Sentiment Classification Distribution

- **LexiconABSA**: Predicts neutral frequently when uncertain
- **TransformerABSA**: Balanced distribution across all sentiments
- **LLMABSA**: Rarely predicts neutral (~2%), showing strong sentiment confidence

### Speed vs Accuracy Tradeoff

```
Speed:     LexiconABSA >>> TransformerABSA >> LLMABSA
Accuracy:  LLMABSA > TransformerABSA > LexiconABSA
```

## Examples

### Example 1: Restaurant Review

```python
text = "The pasta was bland but the atmosphere was cozy."

# Using LexiconABSA
results = lexicon_analyzer.analyze(text)
# Output: [('pasta', 'negative', 0.85), ('atmosphere', 'positive', 0.79)]
```

### Example 2: Complex Sentiment

```python
text = "I love the food, but the wait times are ridiculous!"

# Using LLMABSA (best for nuanced language)
results = llm_analyzer.analyze(text)
# Output: [('food', 'positive', 0.95), ('wait times', 'negative', 0.92)]
```

### Example 3: Implicit Aspects

```python
text = "The cocktails were creative and the menu was diverse."

# TransformerABSA handles implicit aspects well
results = transformer_analyzer.analyze(text)
# Output: [('cocktails', 'positive', 0.91), ('menu', 'positive', 0.88)]
```

## Testing

Run the test suite:

```bash
pytest tests/test_absa.py -v
```

Test specific implementation:

```bash
pytest tests/test_absa.py::test_lexicon_analyze -v
```

Coverage report:

```bash
pytest --cov=src tests/
```
