import re


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = text.replace(";", " ").replace(",", " ")
    # Parentheses are intentionally preserved: HS descriptions use them for critical
    # exclusion clauses (e.g. "excluding centrifuges used in laboratories") that affect matching.
    return re.sub(r"\s+", " ", text).strip()


def prepare_for_embedding(text: str) -> str:
    # Stop word filtering is intentionally omitted: sentence-transformer models are trained on
    # full natural language and perform better with syntactic context intact. Filtering also
    # inverts meaning in HS text (e.g. "not elsewhere specified" → "elsewhere specified").
    return normalize_text(text)
