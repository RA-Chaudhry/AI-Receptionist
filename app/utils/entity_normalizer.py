"""
app/utils/entity_normalizer.py — Robust Entity Normalization & Extraction for Voice Pipeline

Provides deterministic extraction and validation for:
1. Spoken digits (English, Roman Urdu, Urdu script, word numbers, multipliers: double/triple)
2. Pakistani mobile phone numbers (03XXXXXXXXX canonical format)
3. Working days and hours (9 AM–5 PM, Monday–Friday) with strict guards against phone digit contamination
4. Entity context locking to prevent cross-entity state corruption
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Number and Digit Dictionaries
# ──────────────────────────────────────────────────────────────────────────────

_WORD_TO_DIGIT = {
    # English
    "zero": "0", "oh": "0", "o": "0", "nil": "0", "null": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Roman Urdu
    "sifar": "0", "sefar": "0", "cipher": "0",
    "aik": "1", "ek": "1", "yak": "1",
    "do": "2", "doo": "2",
    "teen": "3", "tin": "3",
    "char": "4", "chaar": "4",
    "panch": "5", "paanch": "5",
    "chhe": "6", "che": "6", "chhey": "6", "chey": "6",
    "saat": "7", "sat": "7",
    "aath": "8", "ath": "8",
    "nau": "9", "nao": "9",
    # Urdu Script words
    "صفر": "0", "ایک": "1", "دو": "2", "تین": "3", "چار": "4",
    "پانچ": "5", "چھ": "6", "سات": "7", "آٹھ": "8", "نو": "9",
    # Arabic / Eastern Numerals
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

_MULTIPLIERS = {
    "double": 2,
    "triple": 3,
    "ڈبل": 2,
    "ٹرپل": 3,
}

_DAYS_MAP = {
    # English
    "monday": "Monday", "mon": "Monday",
    "tuesday": "Tuesday", "tue": "Tuesday", "tues": "Tuesday",
    "wednesday": "Wednesday", "wed": "Wednesday",
    "thursday": "Thursday", "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday",
    "friday": "Friday", "fri": "Friday",
    "saturday": "Saturday", "sat": "Saturday",
    "sunday": "Sunday", "sun": "Sunday",
    # Roman Urdu
    "peer": "Monday", "somwar": "Monday", "somwaar": "Monday",
    "mangal": "Tuesday",
    "budh": "Wednesday",
    "jumeraat": "Thursday", "jumerat": "Thursday", "jumerat": "Thursday", "kamis": "Thursday",
    "jummah": "Friday", "jumma": "Friday", "juma": "Friday",
    "hafta": "Saturday", "hafte": "Saturday",
    "itwar": "Sunday", "itwaar": "Sunday",
    # Urdu script
    "پیر": "Monday", "سوموار": "Monday",
    "منگل": "Tuesday",
    "بدھ": "Wednesday",
    "جمعرات": "Thursday",
    "جمعہ": "Friday",
    "ہفتہ": "Saturday",
    "اتوار": "Sunday",
}

WORKING_DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
WEEKEND_DAYS = {"Saturday", "Sunday"}


_EASTERN_NUMERALS_TRANS = str.maketrans({
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})


# ──────────────────────────────────────────────────────────────────────────────
# 2. Digit Normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_spoken_digits(text: str) -> str:
    """
    Normalizes spoken digit sequences into numeric characters.
    Handles multipliers like 'double 5' -> '5 5' and 'triple 8' -> '8 8 8'.
    Converts English, Roman Urdu, Urdu digit words, and Eastern numerals.
    """
    if not text:
        return ""

    # Translate Eastern numerals (۰-۹) to standard digits (0-9)
    text = text.translate(_EASTERN_NUMERALS_TRANS)

    # Tokenize words, punctuation, and digits
    raw_tokens = re.findall(r"[\w\u0600-\u06FF]+|[^\w\s\u0600-\u06FF]", text, flags=re.UNICODE)
    normalized_tokens: list[str] = []

    i = 0
    while i < len(raw_tokens):
        token = raw_tokens[i].lower()

        # Check multiplier (e.g. 'double 5', 'triple 8')
        if token in _MULTIPLIERS and i + 1 < len(raw_tokens):
            multiplier = _MULTIPLIERS[token]
            next_token = raw_tokens[i + 1].lower()
            digit_val = _WORD_TO_DIGIT.get(next_token, next_token if next_token.isdigit() else None)
            if digit_val and len(digit_val) == 1:
                normalized_tokens.extend([digit_val] * multiplier)
                i += 2
                continue

        # Check single digit word
        if token in _WORD_TO_DIGIT:
            normalized_tokens.append(_WORD_TO_DIGIT[token])
        else:
            normalized_tokens.append(raw_tokens[i])
        i += 1

    return " ".join(normalized_tokens)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Phone Number Extraction & Validation
# ──────────────────────────────────────────────────────────────────────────────

def extract_phone_number(text: str) -> Tuple[bool, Optional[str], str]:
    """
    Extracts and validates a Pakistani mobile phone number from speech or text.
    Returns: (is_valid: bool, canonical_number: Optional[str], status_reason: str)
    Canonical format: 03XXXXXXXXX (11 digits).
    """
    if not text or not text.strip():
        return False, None, "empty"

    lower_text = text.lower()

    # Guard: If text contains time/day markers and no explicit phone indicator, it is not a phone number
    has_time_context = bool(re.search(
        r"\b(am|pm|a\.m\.|p\.m\.|baje|بجے|o['']?clock|morning|afternoon|evening|night|noon)\b",
        lower_text,
    )) or bool(re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|peer|mangal|budh|jumeraat|jumerat|jummah|jumma|hafta|itwar)\b",
        lower_text,
    ))
    has_explicit_phone_marker = bool(re.search(
        r"\b(phone|number|num|mobile|contact|digits|call|cell|فون|نمبر)\b",
        lower_text,
    ))
    if has_time_context and not has_explicit_phone_marker:
        return False, None, "no_digits"

    # 1. Normalize spoken words to digits
    normalized = normalize_spoken_digits(text)

    # 2. Extract all contiguous sequences containing digits, hyphens, spaces, plus signs
    # Also check if user entered known non-number filler like 'pipe', 'double-pipe', 'abc'
    garbage_markers = ["double-pipe", "pipe", "pipe.", "slash", "null", "undefined", "n/a"]
    for marker in garbage_markers:
        if marker in text.lower():
            return False, None, "malformed_noise"

    # Extract all digits
    digits = re.sub(r"\D", "", normalized)

    if not digits:
        return False, None, "no_digits"

    # 3. Normalize country prefixes
    # e.g. +923451234567 -> 03451234567, 00923451234567 -> 03451234567, 923451234567 -> 03451234567
    if digits.startswith("0092") and len(digits) == 14:
        digits = "0" + digits[4:]
    elif digits.startswith("92") and len(digits) == 12:
        digits = "0" + digits[2:]
    elif digits.startswith("3") and len(digits) == 10:
        digits = "0" + digits

    # 4. Validate exact Pakistani mobile pattern: 03 followed by 9 digits = 11 digits
    if re.match(r"^03\d{9}$", digits):
        return True, digits, "valid"

    # If message contains other text (like time or greeting) but has explicit phone context, extract embedded 11-digit phone
    if has_explicit_phone_marker:
        embedded_clean = re.sub(r"[^\d\s\+\-]", " ", text)
        for chunk in embedded_clean.split():
            c_digits = re.sub(r"\D", "", chunk)
            if c_digits.startswith("0092") and len(c_digits) == 14:
                c_digits = "0" + c_digits[4:]
            elif c_digits.startswith("92") and len(c_digits) == 12:
                c_digits = "0" + c_digits[2:]
            elif c_digits.startswith("3") and len(c_digits) == 10:
                c_digits = "0" + c_digits
            if re.match(r"^03\d{9}$", c_digits):
                return True, c_digits, "valid"

        # Also search merged spaced digits in explicit phone context
        match = re.search(r"(?:^|[^\d])(03\d{9})(?:[^\d]|$)", re.sub(r"\s+", "", normalized))
        if match:
            return True, match.group(1), "valid"

    if digits.startswith("03") and len(digits) < 11:
        return False, digits, "incomplete"

    if len(digits) != 11:
        return False, None, "invalid_length"

    if not digits.startswith("03"):
        return False, None, "invalid_prefix"

    return False, None, "invalid"


def accumulate_phone_digits(text: str, previous_digits: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
    """
    Accumulates spoken or typed phone digits across turns.
    Guards strictly against appending time expressions (e.g. '3 PM', 'at 4', '3 baje')
    or day names to phone digits.
    Returns: (is_valid: bool, canonical_or_partial_number: Optional[str], status_reason: str)
    """
    if not text or not text.strip():
        if previous_digits:
            return False, previous_digits, "incomplete" if len(previous_digits) < 11 else "invalid"
        return False, None, "empty"

    # 1. If text by itself is a complete valid phone number, use it immediately
    is_valid, standalone_phone, reason = extract_phone_number(text)
    if is_valid and standalone_phone:
        return True, standalone_phone, "valid"
    if reason == "malformed_noise":
        return False, None, "malformed_noise"

    # 2. Strict guard against time expressions or day names without phone intent
    lower_text = text.lower()
    has_time_context = bool(re.search(
        r"\b(am|pm|a\.m\.|p\.m\.|baje|بجے|o['']?clock|morning|afternoon|evening|night|noon)\b",
        lower_text,
    )) or bool(re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|peer|mangal|budh|jumeraat|jumerat|jummah|jumma|hafta|itwar)\b",
        lower_text,
    ))
    has_explicit_phone_marker = bool(re.search(
        r"\b(phone|number|num|mobile|contact|digits|call|cell|فون|نمبر)\b",
        lower_text,
    ))

    # If message is purely or primarily time/day context, do NOT extract its numbers as phone digits
    if has_time_context and not has_explicit_phone_marker:
        if previous_digits:
            return False, previous_digits, "incomplete"
        return False, None, "no_digits"

    # 3. If there are previous partial digits starting with 03 (or country code):
    if previous_digits:
        prev_clean = re.sub(r"\D", "", previous_digits)
        if prev_clean.startswith("0092") and len(prev_clean) >= 14:
            prev_clean = "0" + prev_clean[4:]
        elif prev_clean.startswith("92") and len(prev_clean) >= 12:
            prev_clean = "0" + prev_clean[2:]
        elif prev_clean.startswith("3") and len(prev_clean) >= 10:
            prev_clean = "0" + prev_clean

        norm_chunk = normalize_spoken_digits(text)
        new_digits = re.sub(r"\D", "", norm_chunk)

        if not new_digits:
            return False, prev_clean, "incomplete"

        if prev_clean.startswith("03"):
            combined = prev_clean + new_digits
            if len(combined) == 11 and re.match(r"^03\d{9}$", combined):
                return True, combined, "valid"
            elif len(combined) < 11:
                return False, combined, "incomplete"
            else:
                return False, combined, "invalid_length"

    # Fallback to standalone extraction
    return is_valid, standalone_phone, reason


def format_phone_display(phone_digits: str) -> str:
    """Formats 03XXXXXXXXX as '0345 678 8888' for human readability and speech clarity."""
    if not phone_digits:
        return ""
    if len(phone_digits) == 11 and phone_digits.startswith("03"):
        return f"{phone_digits[:4]} {phone_digits[4:7]} {phone_digits[7:]}"
    if len(phone_digits) > 4:
        return f"{phone_digits[:4]} {phone_digits[4:]}"
    return phone_digits


# ──────────────────────────────────────────────────────────────────────────────
# 4. Day Extraction & Validation
# ──────────────────────────────────────────────────────────────────────────────

import datetime

_RELATIVE_DAYS = {
    "today": 0, "aaj": 0, "آج": 0,
    "tomorrow": 1, "kal": 1, "کل": 1,
    "parson": 2, "parso": 2, "پرسوں": 2,
}


def extract_day(text: str, ref_date: Optional[datetime.date] = None) -> Optional[str]:
    """
    Extracts the canonical day name (e.g. 'Monday', 'Friday') from text.
    Handles explicit weekday names and relative words ('today', 'tomorrow', 'aaj', 'kal', 'آج', 'کل').
    Returns None if no recognizable day name is present.
    """
    if not text:
        return None

    words = re.findall(r"[\w\u0600-\u06FF]+", text.lower(), flags=re.UNICODE)
    # Check explicit weekday names first
    for word in words:
        if word in _DAYS_MAP:
            return _DAYS_MAP[word]

    # Substring check for compound phrases (e.g. 'peer ko', 'jummah ke din')
    lower_text = text.lower()
    for raw_key, canonical in _DAYS_MAP.items():
        if re.search(rf"\b{raw_key}\b", lower_text):
            return canonical

    # Check relative day words
    for word in words:
        if word in _RELATIVE_DAYS:
            offset = _RELATIVE_DAYS[word]
            base_date = ref_date or datetime.date.today()
            return (base_date + datetime.timedelta(days=offset)).strftime("%A")

    return None


def validate_day(day_name: str) -> Tuple[bool, str]:
    """Validates whether a day is a working day (Monday-Friday) or weekend (Saturday-Sunday)."""
    if day_name in WORKING_DAYS:
        return True, "working_day"
    if day_name in WEEKEND_DAYS:
        return False, "weekend"
    return False, "unknown_day"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Time & Hour Extraction (Guarded against phone number digits)
# ──────────────────────────────────────────────────────────────────────────────

# Strict regex matching: requires explicit AM/PM or time keyword context
_TIME_EXPLICIT_PATTERN = re.compile(
    r"""
    \b
    (?P<hour>1[0-2]|0?[1-9])            # 1-12
    (?::(?P<minute>[0-5]\d))?           # optional :00-59
    \s*
    (?P<meridiem>a\.m\.|p\.m\.|am|pm|a|p)\b  # mandatory meridiem
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TIME_KEYWORD_PATTERN = re.compile(
    r"""
    (?:at|around|time|timing|baje|بجے|o['']?clock)\s*
    (?P<hour>1[0-9]|2[0-3]|0?[1-9])
    (?::(?P<minute>[0-5]\d))?
    (?:\s*(?P<meridiem>am|pm|a\.m\.|p\.m\.))?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TIME_POST_KEYWORD_PATTERN = re.compile(
    r"""
    \b
    (?P<hour>1[0-9]|2[0-3]|0?[1-9])
    (?::(?P<minute>[0-5]\d))?
    \s*
    (?:baje|بجے|o['']?clock)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WORD_HOURS = {
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "one": 13, "two": 14, "three": 15, "four": 16, "five": 17,
    "nau": 9, "das": 10, "gyara": 11, "bara": 12,
    "aik": 13, "do": 14, "teen": 15, "char": 16, "panch": 17,
    "نو": 9, "دس": 10, "گیارہ": 11, "بارہ": 12,
    "ایک": 13, "دو": 14, "تین": 15, "چار": 16, "پانچ": 17,
}


def extract_time_hour(text: str, is_awaiting_time_entity: bool = False) -> Optional[int]:
    """
    Extracts the 24-hour integer (0-23) from text.
    STRICT GUARD: Plain digits from phone numbers (e.g. '0, 3, 4, 5...') will NOT match.
    """
    if not text:
        return None

    # Guard: If text is a phone number or long digit sequence (>= 8 digits), do not parse as time!
    digits_only = re.sub(r"\D", "", text)
    if len(digits_only) >= 7:
        return None

    # 1. Explicit meridiem pattern: '3 PM', '9 AM', '3:30 pm', '4p'
    match = _TIME_EXPLICIT_PATTERN.search(text)
    if match:
        hour = int(match.group("hour"))
        meridiem = match.group("meridiem").lower().replace(".", "")
        if meridiem in ("am", "a"):
            return 0 if hour == 12 else hour
        elif meridiem in ("pm", "p"):
            return 12 if hour == 12 else hour + 12

    # 2. Time keyword patterns: 'at 3', 'at 15:00', '3 baje', '3 o clock'
    match_kw = _TIME_KEYWORD_PATTERN.search(text) or _TIME_POST_KEYWORD_PATTERN.search(text)
    if match_kw:
        hour = int(match_kw.group("hour"))
        meridiem_str = match_kw.groupdict().get("meridiem")
        if meridiem_str:
            m = meridiem_str.lower().replace(".", "")
            if m in ("am", "a"):
                return 0 if hour == 12 else hour
            elif m in ("pm", "p"):
                return 12 if hour == 12 else hour + 12

        # Context heuristics for numbers like '3 baje' / 'at 3':
        # Business hours are 9 to 17 (9 AM to 5 PM). 1-5 mean 1 PM to 5 PM (13-17).
        if 1 <= hour <= 5:
            return hour + 12
        if 9 <= hour <= 17:
            return hour
        return hour

    # 3. Word-based times: 'teen baje', 'three o clock', 'das baje'
    for word, hr in _WORD_HOURS.items():
        if re.search(rf"\b{word}\s*(?:pm|am|p\.m\.|a\.m\.|baje|بجے|o['']?clock)\b", text, re.IGNORECASE):
            return hr

    # 4. If conversation state is explicitly awaiting time and user responds with just a single number (9-17 or 1-5):
    if is_awaiting_time_entity:
        single_num_match = re.match(r"^\s*(\d{1,2})\s*$", text.strip())
        if single_num_match:
            num = int(single_num_match.group(1))
            if 1 <= num <= 5:
                return num + 12
            if 9 <= num <= 17:
                return num

    return None


def validate_working_hour(hour: int) -> Tuple[bool, str]:
    """Validates whether an hour is within office working hours (9 AM to 5 PM inclusive)."""
    if hour == 0:
        return False, "midnight"
    if 9 <= hour <= 17:
        return True, "working_hour"
    return False, "out_of_hours"


def format_hour_display(hour: int) -> str:
    """Converts 24-h integer (9-17) into a display string ('9 AM', '3 PM', '5 PM')."""
    if hour == 0:
        return "12 AM"
    if hour < 12:
        return f"{hour} AM"
    if hour == 12:
        return "12 PM"
    return f"{hour - 12} PM"


def correct_voice_vocabulary(text: str) -> str:
    """
    Replaces misheard STT variations of company/brand names (e.g. 'logicals',
    'ligit pulse', 'logic pals', 'lodge pulse') with the canonical name 'Logic Pulse'.
    Uses regex word boundaries (\\b) and sorts by length descending so multi-word
    phrases match before single words, and normal words (e.g. 'chronological', 'illogical')
    are never mangled.
    """
    if not text:
        return text

    from app.core.config import settings

    corrections = getattr(settings, "vocabulary_corrections", [
        "logicals",
        "logical",
        "ligit pulse",
        "logic pals",
        "logic pulls",
        "lodge pulse",
    ])
    target = getattr(settings, "vocabulary_target", "Logic Pulse")

    # Sort descending by length so multi-word variants match before single words
    sorted_corrections = sorted(corrections, key=len, reverse=True)

    result = text
    for phrase in sorted_corrections:
        phrase_clean = phrase.strip()
        if not phrase_clean:
            continue
        pattern = rf"\b{re.escape(phrase_clean)}\b"
        result = re.sub(pattern, target, result, flags=re.IGNORECASE)

    return result

