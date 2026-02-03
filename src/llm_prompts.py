"""
LLM prompt templates with few-shot examples for ABSA.
Modern, clean prompt engineering patterns.
"""


def build_detailed_prompt(text: str) -> str:
    """
    Advanced prompt with few-shot examples.
    Optimized for highest accuracy.
    """
    return f"""You are an expert at Aspect-Based Sentiment Analysis for restaurant reviews.
Your task is to identify ALL aspects mentioned and their sentiments with high accuracy.

EXAMPLES:

Review: "The pasta was delicious but the service was terrible."
Output: [{{"aspect": "pasta", "sentiment": "positive", "confidence": 0.95, "text_span": [4, 9]}}, {{"aspect": "service", "sentiment": "negative", "confidence": 0.95, "text_span": [31, 38]}}]

Review: "Great atmosphere and excellent wine selection!"
Output: [{{"aspect": "atmosphere", "sentiment": "positive", "confidence": 0.9, "text_span": [6, 16]}}, {{"aspect": "wine selection", "sentiment": "positive", "confidence": 0.9, "text_span": [31, 45]}}]

Review: "The pizza was okay, nothing special."
Output: [{{"aspect": "pizza", "sentiment": "neutral", "confidence": 0.85, "text_span": [4, 9]}}]

NOW ANALYZE THIS REVIEW:

Review: "{text}"

IMPORTANT:
- Find ALL aspects mentioned (food items, service, atmosphere, price, etc.)
- Use "positive" for good/great/excellent/amazing/delicious
- Use "negative" for bad/terrible/awful/disappointing/slow
- Use "neutral" for okay/average/nothing special
- Set confidence 0.9-1.0 for clear sentiments, 0.7-0.8 for subtle ones
- Be thorough - don't miss any aspects!

Return ONLY the JSON array:"""


def build_simple_prompt(text: str) -> str:
    """
    Simpler prompt for retry attempt.
    One clear example.
    """
    return f"""Extract aspects and sentiments from restaurant review.

Example:
Text: "Great food but slow service"
JSON: [{{"aspect":"food","sentiment":"positive","confidence":0.9,"text_span":[6,10]}},{{"aspect":"service","sentiment":"negative","confidence":0.9,"text_span":[21,28]}}]

Now analyze:
Text: "{text}"
JSON:"""


def build_strict_prompt(text: str) -> str:
    """
    Minimal prompt for final retry.
    Maximum simplicity.
    """
    return f"""Review: "{text}"

Find aspects (food/service/atmosphere) and sentiment (positive/negative/neutral).

JSON array only:"""
