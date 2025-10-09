# Helper functions

def build_prompt(text: str) -> str:
    base_prompt = """
You are an Aspect-Based Sentiment Analysis (ABSA) assistant.

Your task:
- Extract aspects (features, entities, or attributes) mentioned in the text.
- Determine sentiment toward each aspect: "positive", "negative", or "neutral".
- Provide confidence as a number between 0.0 and 1.0.
- Include the character span [start_index, end_index] of each aspect in the original text.

Respond ONLY with valid JSON in this format:
[
  {{
    "aspect": "<aspect term>",
    "sentiment": "<positive|negative|neutral>",
    "confidence": <0.0-1.0>,
    "text_span": [<start_index>, <end_index>]
  }}
]

Below are several examples to guide you.

Example 1:
Input: "The pizza was delicious but the service was terrible."
Output:
[
  {{"aspect": "pizza", "sentiment": "positive", "confidence": 0.95, "text_span": [4, 9]}},
  {{"aspect": "service", "sentiment": "negative", "confidence": 0.92, "text_span": [29, 36]}}
]

Example 2:
Input: "The laptop screen is bright, but the battery life is disappointing."
Output:
[
  {{"aspect": "screen", "sentiment": "positive", "confidence": 0.93, "text_span": [11, 17]}},
  {{"aspect": "battery life", "sentiment": "negative", "confidence": 0.90, "text_span": [32, 45]}}
]

Example 3:
Input: "The hotel room was okay and the location was fine."
Output:
[
  {{"aspect": "room", "sentiment": "neutral", "confidence": 0.76, "text_span": [10, 14]}},
  {{"aspect": "location", "sentiment": "neutral", "confidence": 0.78, "text_span": [31, 39]}}
]

Now analyze this text:
"{text}"
"""
    return base_prompt.format(text=text)
