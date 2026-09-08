"""Token-local stopping conditions for finite holds."""

from __future__ import annotations


SENTENCE_TERMINATORS = ".!?"


def token_boundaries(text: str) -> frozenset[str]:
    """Classify decoded token text, including delimiters inside compound tokens.

    The entire matching token is retained. Separate closing quotes, whitespace,
    and subsequent tokens are left for the teacher; no lookahead is required.
    """
    boundaries = set()
    if any(character in text for character in SENTENCE_TERMINATORS):
        boundaries.add("sentence")
    if "\n" in text:
        boundaries.add("newline")
    return frozenset(boundaries)
