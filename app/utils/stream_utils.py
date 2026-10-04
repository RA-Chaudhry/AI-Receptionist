"""
app/utils/stream_utils.py — Sentence-level stream buffering for real-time Voice TTS

Buffers incremental token chunks from LLM generation and emits:
1. The FIRST chunk as early as possible (at first comma/semicolon/clause after >= 15 chars, or first sentence end)
2. Subsequent chunks at full sentence boundaries
Also performs markdown and emoji cleaning for optimal TTS synthesis latency and pronunciation.
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

# Regex to detect clause punctuation (,;: or Urdu comma \u060C or Urdu semicolon \u061B or em-dash)
_FIRST_CLAUSE_PATTERN = re.compile(r"([,;:\u060C\u061B\u2014\-]+)(?:\s+|$)")


def clean_tts_text(text: str) -> str:
    """Removes markdown formatting and emojis that delay or distort TTS."""
    if not text:
        return ""
    # Strip markdown bold, italic, backticks, hashes, bullet points
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s*[-*#]+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[#_~\[\]><|\\]", "", text)
    # Remove emoji Unicode planes
    text = re.sub(r"[\U00010000-\U0010ffff]|[\u2600-\u27BF]|[\uD800-\uDBFF][\uDC00-\uDFFF]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    min_first_clause_chars: int = 15,
    min_clause_words: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Consumes tokens from an async generator, buffers them, and yields:
    - First chunk at first clause boundary (,;: / Urdu comma) once >= 15 chars (or min_clause_words), or first sentence end.
    - Subsequent chunks at full sentence boundaries (. ! ? \n) or clause boundaries if min_clause_words is specified.

    Parameters:
        token_stream: Async generator yielding string token deltas from LLM.
        min_first_clause_chars: Minimum character length required before emitting first clause.
        min_clause_words: Optional minimum word count threshold for clause splitting.
    """
    buffer = ""
    is_first_chunk = True

    async for token in token_stream:
        if not token:
            continue
        buffer += token

        # 1. First chunk check: split early at clause boundary if length/word threshold met
        if is_first_chunk:
            meets_thresh = (len(buffer.strip()) >= min_first_clause_chars) or (
                min_clause_words is not None and len(buffer.strip().split()) >= min_clause_words
            )
            if meets_thresh:
                clause_match = _FIRST_CLAUSE_PATTERN.search(buffer)
                if clause_match:
                    split_pos = clause_match.end()
                    candidate = buffer[:split_pos].strip()
                    if candidate and not _ends_with_abbreviation(candidate):
                        cleaned = clean_tts_text(candidate)
                        if cleaned:
                            yield cleaned
                            is_first_chunk = False
                            buffer = buffer[split_pos:].lstrip()
                            continue
        elif min_clause_words is not None and len(buffer.strip().split()) >= min_clause_words:
            clause_match = _FIRST_CLAUSE_PATTERN.search(buffer)
            if clause_match:
                split_pos = clause_match.end()
                candidate = buffer[:split_pos].strip()
                if candidate and not _ends_with_abbreviation(candidate):
                    cleaned = clean_tts_text(candidate)
                    if cleaned:
                        yield cleaned
                        buffer = buffer[split_pos:].lstrip()
                        continue

        # 2. Check for sentence endings (. ! ? \n \u06D4 \u061F)
        while True:
            match = _SENTENCE_END_PATTERN.search(buffer)
            if not match:
                break

            split_pos = match.end()
            candidate = buffer[:split_pos].strip()
            punct_pos = match.start()

            # Verify it's not a decimal number (e.g. 3.14) or abbreviation (e.g. Dr.)
            if _is_decimal_number(buffer, punct_pos) or _ends_with_abbreviation(candidate):
                next_search_start = match.end()
                next_match = _SENTENCE_END_PATTERN.search(buffer, next_search_start)
                if not next_match:
                    break
                split_pos = next_match.end()
                candidate = buffer[:split_pos].strip()

            if candidate:
                cleaned = clean_tts_text(candidate)
                if cleaned:
                    yield cleaned
                    is_first_chunk = False
            buffer = buffer[split_pos:].lstrip()

    # Flush any remaining text in the buffer upon stream completion
    final_chunk = buffer.strip()
    if final_chunk:
        cleaned = clean_tts_text(final_chunk)
        if cleaned:
            yield cleaned

