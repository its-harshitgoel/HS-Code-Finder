"""
Text preprocessing utilities for the HSCodeFinder application.

Purpose: Normalizes user input and HS descriptions for better embedding quality.
Inputs: Raw text strings.
Outputs: Cleaned, normalized text strings.
"""

import re
import string


# Common English stop words to remove for better semantic matching
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "not", "no", "nor",
    "as", "if", "than", "too", "very", "just", "about", "above", "after",
    "again", "all", "also", "am", "any", "because", "before", "between",
    "both", "each", "few", "further", "here", "how", "into", "more",
    "most", "other", "our", "out", "over", "own", "same", "so", "some",
    "such", "then", "there", "through", "under", "until", "up", "what",
    "when", "where", "which", "while", "who", "whom", "why",
})


def normalize_text(text: str) -> str:
    """Normalize text for embedding generation.

    Applies: lowercase, strip extra whitespace, normalize punctuation.
    Preserves meaningful content words.

    Args:
        text: Raw input text.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower().strip()

    # Replace semicolons and commas with spaces (common in HS descriptions)
    text = text.replace(";", " ").replace(",", " ")

    # Remove parenthetical content like "(Gadus morhua)" for cleaner matching
    text = re.sub(r"\([^)]*\)", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stop_words(text: str) -> str:
    """Remove common stop words from text.

    Args:
        text: Normalized text.

    Returns:
        Text with stop words removed.
    """
    words = text.split()
    filtered = [w for w in words if w not in STOP_WORDS]
    return " ".join(filtered) if filtered else text


def prepare_for_embedding(text: str) -> str:
    """Full preprocessing pipeline for embedding generation.

    Combines normalization and stop word removal.

    Args:
        text: Raw input text.

    Returns:
        Clean text ready for embedding model.
    """
    normalized = normalize_text(text)
    return remove_stop_words(normalized)


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text for question generation.

    Args:
        text: HS description or user input.

    Returns:
        List of unique keywords.
    """
    normalized = normalize_text(text)
    words = normalized.split()
    # Remove stop words and very short words
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique
