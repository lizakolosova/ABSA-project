# Helper functions

def build_prompt(text: str) -> str:
    base_prompt = """
You are an Aspect-Based Sentiment Analysis (ABSA) assistant.

Your task:
- Extract aspects (features, entities, or attributes) mentioned in the text.
- Determine sentiment toward each aspect: "positive", "negative", or "neutral".
- Provide confidence as a number between 0.0 and 1.0.
- Include the character span [start_index, end_index] of each aspect in the original text.

Output format:
Respond ONLY with valid JSON in this format:
[
  {{
    "aspect": "<aspect term>",
    "sentiment": "<positive|negative|neutral>",
    "confidence": <0.0-1.0>,
    "text_span": [<start_index>, <end_index>]
  }}
]

Example:
Input: "The pizza was delicious but the service was terrible."
Output:
[
  {{"aspect": "pizza", "sentiment": "positive", "confidence": 0.95, "text_span": [4, 9]}},
  {{"aspect": "service", "sentiment": "negative", "confidence": 0.92, "text_span": [29, 36]}}
]

Now analyze this text:
"{text}"
"""
    return base_prompt.format(text=text)
