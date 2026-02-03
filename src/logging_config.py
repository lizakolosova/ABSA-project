import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True
    )
    
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
