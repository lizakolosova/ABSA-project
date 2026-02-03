import re
from typing import List, Optional

PRONOUNS = {"i", "you", "he", "she", "it", "they", "we", "me", "him", "her", "them", "us"}
TEMPORAL_WORDS = {
    "while", "minute", "hour", "day", "week", "month", "year",
    "morning", "evening", "afternoon", "yesterday", "today", "tomorrow"
}


def is_valid_aspect(candidate: str, nlp) -> bool:
    text = candidate.strip()

    if not text or len(re.sub(r"[^A-Za-z0-9]", "", text)) < 2:
        return False

    low = text.lower()
    if low in PRONOUNS or any(t in low for t in TEMPORAL_WORDS):
        return False

    doc = nlp(text)
    return any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc)


def split_aspect_candidates(raw: str) -> List[str]:
    raw = raw.strip().strip('''"'""''()[]{}.,:;!?-''')
    if not raw:
        return []

    parts = re.split(r'[/&,]', raw)
    result = []
    
    for part in parts:
        if not (part := part.strip()):
            continue

        if " and " in part.lower():
            halves = [h.strip() for h in re.split(r'\band\b', part, flags=re.IGNORECASE)]
            if all(len(h.split()) <= 4 for h in halves) and len(halves) > 1:
                result.extend(h for h in halves if h)
                continue
        
        result.append(part)
    
    return result


def find_aspect_root(chunk) -> str:
    tokens = [
        t.text.lower() 
        for t in chunk 
        if t.dep_ == "compound" or t == chunk.root
    ]
    return " ".join(tokens) if tokens else chunk.root.text.lower()


def has_negation(token) -> bool:
    negations = {"not", "never", "no", "none", "n't"}
    check_tokens = list(token.subtree) + list(token.head.subtree)
    return any(t.text.lower() in negations for t in check_tokens)


def normalize_aspect(aspect: str) -> str:
    return aspect.lower().strip()

def _extract_json(text: str) -> Optional[str]:
    if not (text := text.strip()):
        return None

    if text.startswith('[') and text.endswith(']'):
        return text

    if match := re.search(r'\[[\s\S]*]', text):
        return match.group(0)

    if '```json' in text:
        parts = text.split('```json')
        if len(parts) > 1 and '```' in (after := parts[1]):
            return after.split('```')[0].strip()

    if '```' in text:
        parts = text.split('```')
        if len(parts) >= 3 and (middle := parts[1].strip()).startswith('['):
            return middle

    return text