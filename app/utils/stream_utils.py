"""
app/utils/stream_utils.py — Sentence-level stream buffering for real-time Voice TTS

Buffers incremental token chunks from LLM generation and emits complete sentences/clauses
as soon as punctuation boundaries are reached, enabling streaming TTS while subsequent
tokens are still being generated.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator, List

# Common abbreviations to avoid false positive sentence splitting
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "vs.", "etc.", "inc.", "ltd.", "co.",
    "ceo.", "cto.", "cfo.", "ai.",
}

# Regex to detect sentence ending punctuation (.!? or newline, or Urdu full stop \u06D4, or Urdu question mark \u061F)
# Followed by a space, newline, or quote/end of text
_SENTENCE_END_PATTERN = re.compile(r"([.!?\n\u06D4\u061F]+)(?:\s+|$)")

# Regex to detect clause punctuation (,;: or Urdu comma \u060C or Urdu semicolon \u061B) after sufficient word count
_CLAUSE_END_PATTERN = re.compile(r"([,;:\u060C\u061B]+)\s+")


def _ends_with_abbreviation(text: str) -> bool:
    """Checks if the trailing word in text matches a known abbreviation."""
    words = text.strip().split()
    if not words:
        return False
    last_word = words[-1].lower()
    return last_word in _ABBREVIATIONS


def _is_decimal_number(text: str, match_pos: int) -> bool:
    """Checks if a period at match_pos is part of a decimal number (e.g. 3.5)."""
    if match_pos > 0 and match_pos < len(text) - 1:
        before = text[match_pos - 1]
        after = text[match_pos + 1]
        if before.isdigit() and after.isdigit():
            return True
    return False


async def stream_sentence_chunks(
    token_stream: AsyncGenerator[str, None],
    min_clause_words: int = 15,
) -> AsyncGenerator[str, None]:
    """
    Consumes tokens from an async generator, buffers them, and yields complete
    sentences or long clauses as soon as they are ready for TTS synthesis.

    Parameters:
        token_stream: Async generator yielding string token deltas from LLM.
        min_clause_words: Minimum word count required before splitting on clause boundaries (,;:)
    """
    buffer = ""

    async for token in token_stream:
        if not token:
            continue
        buffer += token

        # Check for sentence endings (. ! ? \n)
        while True:
            match = _SENTENCE_END_PATTERN.search(buffer)
            if not match:
                break

            split_pos = match.end()
            candidate = buffer[:split_pos].strip()
            punct_pos = match.start()

            # Verify it's not a decimal number (e.g. 3.14) or abbreviation (e.g. Dr.)
            if _is_decimal_number(buffer, punct_pos) or _ends_with_abbreviation(candidate):
                # Search further in the remaining buffer
                next_search_start = match.end()
                next_match = _SENTENCE_END_PATTERN.search(buffer, next_search_start)
                if not next_match:
                    break
                split_pos = next_match.end()
                candidate = buffer[:split_pos].strip()

            if candidate:
                yield candidate
            buffer = buffer[split_pos:].lstrip()

        # Check for clause boundaries if buffer has grown long
        word_count = len(buffer.split())
        if word_count >= min_clause_words:
            clause_match = _CLAUSE_END_PATTERN.search(buffer)
            if clause_match:
                split_pos = clause_match.end()
                clause_candidate = buffer[:split_pos].strip()
                if clause_candidate and not _ends_with_abbreviation(clause_candidate):
                    yield clause_candidate
                    buffer = buffer[split_pos:].lstrip()

    # Flush any remaining text in the buffer upon stream completion
    final_chunk = buffer.strip()
    if final_chunk:
        yield final_chunk
