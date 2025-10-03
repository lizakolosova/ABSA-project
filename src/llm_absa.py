# Implementation 3
# src/llm_absa.py
import ollama
import json
from typing import List
from src.base import ABSAAnalyzer, AspectSentiment
from src.utils import build_prompt

class LLMABSA(ABSAAnalyzer):
    def __init__(self, model: str = "llama3"):
        self.model = model

    def analyze(self, text: str) -> List[AspectSentiment]:
        prompt = build_prompt(text)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_output = response["message"]["content"]

            # Try parsing JSON output
            data = json.loads(raw_output)
            results = []
            for item in data:
                results.append(
                    AspectSentiment(
                        aspect=item.get("aspect", ""),
                        sentiment=item.get("sentiment", "neutral"),
                        confidence=float(item.get("confidence", 0.5)),
                        text_span=tuple(item.get("text_span", (0, 0)))
                    )
                )
            return results

        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            return []
