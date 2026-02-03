from dataclasses import dataclass
from pathlib import Path
import os
import json


@dataclass
class Config:
    aspect_model: str = "gauneg/roberta-base-absa-ate-sentiment"
    sentiment_model: str = "yangheng/deberta-v3-base-absa-v1.1"
    llm_model: str = "llama3"
    spacy_model: str = "en_core_web_trf"
    
    device: int = -1
    batch_size: int = 16
    max_text_length: int = 100_000
    timeout_seconds: int = 30
    
    cache_dir: Path = Path.home() / ".cache" / "absa"
    log_level: str = "INFO"
    
    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, path: Path) -> "Config":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "Config":
        kwargs = {}
        if val := os.getenv("ABSA_DEVICE"):
            kwargs["device"] = int(val)
        if val := os.getenv("ABSA_BATCH_SIZE"):
            kwargs["batch_size"] = int(val)
        if val := os.getenv("ABSA_LOG_LEVEL"):
            kwargs["log_level"] = val
        return cls(**kwargs)


config = Config()
