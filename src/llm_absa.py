import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import ollama

from .base import ABSAAnalyzer, AspectSentiment
from .llm_prompts import build_detailed_prompt, build_simple_prompt, build_strict_prompt
from .logging_config import get_logger
from .text_utils import _extract_json

logger = get_logger(__name__)


class ResponseCache:
    
    def __init__(self, ttl: timedelta = timedelta(hours=1)):
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        self.cache[key] = (value, datetime.now())
    
    def clear(self) -> None:
        self.cache.clear()


class LLMABSA(ABSAAnalyzer):
    
    def __init__(
        self,
        model: str = "llama3",
        timeout: int = 30,
        max_retries: int = 3,
        use_cache: bool = True
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = ResponseCache() if use_cache else None
        
        self._verify_ollama()
        logger.info(f"LLMABSA initialized with {model}")
    
    def _verify_ollama(self) -> None:
        try:
            response = ollama.list()
            models = [
                m.get("name", m.get("model", ""))
                for m in response.get("models", [])
                if isinstance(m, dict)
            ]
            
            if models and self.model not in models:
                logger.warning(f"Model '{self.model}' not in available models: {models}")
        except Exception as e:
            logger.warning(f"Could not verify Ollama: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    def _call_ollama(self, prompt: str, temperature: float = 0.3) -> dict:
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "timeout": self.timeout,
                        "temperature": temperature,
                        "top_p": 0.9
                    }
                )
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Network attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _parse_response(raw_output: str) -> List[Dict[str, Any]]:
        if not raw_output:
            logger.warning("Empty LLM response")
            return []

        try:
            if not (json_str := _extract_json(raw_output)):
                logger.warning("Could not extract JSON")
                logger.debug(f"Raw output: {raw_output[:200]}")
                return []

            data = json.loads(json_str)
            if not isinstance(data, list):
                logger.warning(f"Expected list, got {type(data).__name__}")
                return []

            validated = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                if not all(k in item for k in ["aspect", "sentiment"]):
                    logger.debug(f"Skipping incomplete item: {item}")
                    continue

                sentiment = item.get("sentiment", "neutral").lower()
                if sentiment not in ["positive", "negative", "neutral"]:
                    logger.debug(f"Invalid sentiment '{sentiment}', using neutral")
                    sentiment = "neutral"

                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))

                text_span = item.get("text_span", [0, 0])
                if not isinstance(text_span, list) or len(text_span) != 2:
                    text_span = [0, 0]

                validated.append({
                    "aspect": str(item.get("aspect", "")).strip(),
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "text_span": text_span
                })

            return validated

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []
        except Exception as e:
            logger.warning(f"Unexpected parsing error: {e}")
            return []

    def analyze(self, text: str) -> List[AspectSentiment]:
        start_time = time.time()
        
        try:
            self.validate_input(text, max_length=50_000)

            if self.cache and (cached := self.cache.get(self._get_cache_key(text))):
                logger.info("Cache hit")
                return cached

            prompts = [
                ("detailed", build_detailed_prompt(text)),
                ("simple", build_simple_prompt(text)),
                ("strict", build_strict_prompt(text))
            ]
            
            results = []
            for attempt, (name, prompt) in enumerate(prompts, 1):
                if attempt > 1:
                    logger.info(f"Retry {attempt} with {name} prompt")
                
                try:
                    response = self._call_ollama(prompt)
                    raw_output = response.get("message", {}).get("content", "")
                    
                    if not raw_output:
                        logger.warning(f"Empty response on attempt {attempt}")
                        if attempt < len(prompts):
                            time.sleep(2)
                            continue
                        return []
                    
                    logger.debug(f"Response length: {len(raw_output)} chars")

                    if not (data := self._parse_response(raw_output)):
                        logger.warning(f"Parse failed on attempt {attempt}")
                        if attempt < len(prompts):
                            time.sleep(2)
                            continue
                        return []

                    for item in data:
                        try:
                            results.append(
                                AspectSentiment(
                                    aspect=item["aspect"],
                                    sentiment=item["sentiment"],
                                    confidence=item["confidence"],
                                    text_span=tuple(item["text_span"]),
                                    model_name=f"LLMABSA-{self.model}"
                                )
                            )
                        except Exception as e:
                            logger.debug(f"Skipping invalid aspect: {e}")
                    
                    if results:
                        logger.info(f"Success on attempt {attempt}: {len(results)} aspects")
                        break
                    
                    if attempt < len(prompts):
                        time.sleep(2)
                
                except Exception as e:
                    logger.error(f"Attempt {attempt} error: {e}")
                    if attempt < len(prompts):
                        time.sleep(2)
                        continue
                    return []

            if self.cache and results:
                self.cache.set(self._get_cache_key(text), results)
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Found {len(results)} aspects in {elapsed:.0f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []
    
    def clear_cache(self) -> None:
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")