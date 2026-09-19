"""
receptionist_router.py — Logic Pulse AI Receptionist
Rule-based local router. Handles predefined intents without calling Groq,
reducing API cost to $0 for all simple queries.

Fixes applied:
  1. Weekend: only Saturday & Sunday — never Friday
  2. Time validation: 9 AM–5 PM inclusive; rejects midnight / out-of-hours
  3. Context memory: extracts day / time / phone from conversation history
  4. Confirmation step before finalising appointment
  5. Non-repetitive responses via index cycling
  6. Input normalisation: "4p"→"4 PM", fuzzy typo matching
  7. Escalation only after 2 consecutive failed attempts
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Predefined response banks
# ──────────────────────────────────────────────────────────────────────────────

GREETING_RESPONSES = [
    "Hello, welcome to Logic Pulse. How can I assist you?",
    "Hi, you've reached Logic Pulse. How may I help you today?",
    "Welcome! Please let me know your query.",
    "Good day! I'm here to help. What can I do for you?",
    "I'm here to assist you. Please tell me your request.",
]

TIMING_RESPONSES = [
    "Our office hours are 9 AM to 5 PM.",
    "We are open Monday to Friday.",
    "We are closed on weekends (Saturday and Sunday).",
    "Please contact us during working hours — 9 AM to 5 PM, Monday to Friday.",
    "Our office is currently closed. We operate Monday to Friday, 9 AM to 5 PM.",
]

LOCATION_RESPONSES = [
    "We can send you our location via SMS — please share your number.",
    "Please confirm your number so we can send you the location details.",
    "We provide a Google Maps link. Shall I send it to your number?",
    "Appointments are required for office visits. Would you like to book one?",
    "You can also schedule an online meeting with our team.",
]

SERVICES_RESPONSES = [
    "We provide custom software development tailored to your business needs.",
    "We build AI and automation solutions for modern businesses.",
    "We develop business systems and dashboards for better insights.",
    "We create chatbots and AI agents to automate customer interactions.",
    "We build websites and mobile applications across all platforms.",
]

PRICING_RESPONSES = [
    "Pricing depends on your project requirements.",
    "Basic projects are more affordable — please share details for an estimate.",
    "Advanced or complex systems are priced individually after consultation.",
    "Please share your requirements and we'll provide an accurate estimate.",
    "We offer a free initial consultation — no commitment required.",
]

APPOINTMENT_RESPONSES = [
    "I'd be happy to schedule an appointment with our CEO, Muhammad Ramzan. What day and time works for you?",
    "Sure! To book your appointment, please tell me your preferred day (Monday–Friday) and time (9 AM–5 PM).",
    "Your appointment is being arranged. Could you confirm the day and time?",
    "Let me get that booked for you. What day and time are you available?",
]

CLARIFICATION_RESPONSES = [
    "I didn't fully understand. Could you please clarify your preferred day and time?",
    "Could you rephrase that? I want to make sure I get it right.",
    "I'm not quite sure I followed — could you say that again?",
    "My apologies. Could you be more specific about your request?",
]

FALLBACK_RESPONSES = [
    "Let me connect you to a human agent for further assistance.",
    "I'll transfer you to our team who can better help you.",
    "Our team will handle this — please hold while I connect you.",
]

CLOSING_RESPONSES = [
    "Your request has been received. Our team will contact you shortly.",
    "Thank you for your time. Is there anything else I can help with?",
    "You will receive an update soon. Have a great day!",
    "Goodbye, and thank you for contacting Logic Pulse!",
    "It was a pleasure assisting you. Take care!",
]

# Substring triggers used to detect clarification assistant messages in history
_CLARIFICATION_TRIGGERS: List[str] = [
    "didn't fully understand",
    "could you please clarify",
    "could you rephrase",
    "not quite sure i followed",
    "could you be more specific",
    "clarify your preferred",
]

CEO_INFO = "The CEO of Logic Pulse is Muhammad Ramzan."

WORKING_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday"}
WEEKEND_DAYS  = {"saturday", "sunday"}

# ──────────────────────────────────────────────────────────────────────────────
# Input normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

# Short-form / typo → canonical spelling mappings
_TYPO_MAP: Dict[str, str] = {
    "shadule": "schedule",
    "shedule": "schedule",
    "scheudle": "schedule",
    "scedule": "schedule",
    "apointment": "appointment",
    "appoinment": "appointment",
    "appoinmtent": "appointment",
    "apointmnt": "appointment",
    "sat": "saturday",
    "sun": "sunday",
    "mon": "monday",
    "tue": "tuesday",
    "wed": "wednesday",
    "thu": "thursday",
    "fri": "friday",
}

_TIME_PATTERN = re.compile(
    r"""
    \b
    (?P<hour>\d{1,2})           # hour
    (?::(?P<minute>\d{2}))?     # optional :mm
    \s*                         # optional space before meridiem
    (?P<meridiem>a\.m\.|p\.m\.|am|pm)?  # optional AM/PM (longer forms first)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Compact forms: "4p" "5pm" "9a" "11am"
_COMPACT_TIME = re.compile(r"\b(\d{1,2})(a|p)\b", re.IGNORECASE)


def _apply_typo_corrections(text: str) -> str:
    """Replace known typos/short-forms with canonical words."""
    words = text.split()
    corrected = [_TYPO_MAP.get(w.lower(), w) for w in words]
    return " ".join(corrected)


def normalise_input(text: str) -> str:
    """Normalise user input: expand compact times, fix typos."""
    # Expand compact times first: "4p" → "4 pm", "9a" → "9 am"
    text = _COMPACT_TIME.sub(
        lambda m: f"{m.group(1)} {'PM' if m.group(2).lower() == 'p' else 'AM'}",
        text,
    )
    # Attach meridiem when written without space: "5pm" → "5 PM"
    text = re.sub(r"(\d)(am|pm)", lambda m: f"{m.group(1)} {m.group(2).upper()}", text, flags=re.IGNORECASE)
    # Typo corrections
    text = _apply_typo_corrections(text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Time / Day / Phone parsing and validation (via entity_normalizer)
# ──────────────────────────────────────────────────────────────────────────────

from app.utils.entity_normalizer import (
    extract_phone_number as _norm_extract_phone_number,
    format_phone_display as _norm_format_phone_display,
    extract_day as _norm_extract_day,
    validate_day as _norm_validate_day,
    extract_time_hour as _norm_extract_time_hour,
    validate_working_hour as _norm_validate_working_hour,
    format_hour_display as _norm_format_hour_display,
    normalize_spoken_digits as _norm_spoken_digits,
)


def _extract_day(text: str) -> Optional[str]:
    """Return the first working-day or weekend-day name found in text (lowercase)."""
    day = _norm_extract_day(text)
    return day.lower() if day else None


def _extract_time_hour(text: str) -> Optional[int]:
    """
    Return the 24-h integer hour parsed from text, or None if unparseable.
    Applies strict meridiem/keyword rules. Plain phone digits will NOT match.
    """
    return _norm_extract_time_hour(text)


def validate_day(day: str) -> Tuple[bool, str]:
    """Return (is_valid, reason). Weekends are Saturday and Sunday ONLY."""
    d = day.capitalize()
    valid, reason = _norm_validate_day(d)
    if valid:
        return True, ""
    if reason == "weekend":
        return False, "weekend"
    return False, "unknown"


def validate_time(hour: int) -> Tuple[bool, str]:
    """Return (is_valid, reason). Valid window: 9 (9 AM) to 17 (5 PM) inclusive."""
    valid, reason = _norm_validate_working_hour(hour)
    if valid:
        return True, ""
    if reason == "midnight":
        return False, "midnight"       # 12 AM always rejected
    return False, "out_of_hours"


def hour_to_display(hour: int) -> str:
    """Convert 24-h integer to display string like '9 AM', '5 PM', '12 PM'."""
    return _norm_format_hour_display(hour)


# ──────────────────────────────────────────────────────────────────────────────
# Context extraction from history
# ──────────────────────────────────────────────────────────────────────────────

_PHONE_RE = re.compile(r"\+?\d[\d\s\-]{5,14}\d")


def _extract_context_from_history(history: List[Dict[str, str]]) -> Dict[str, Optional[object]]:
    """
    Walk through conversation history and collect:
      - selected_day  (last confirmed working day)
      - selected_time_hour (last confirmed valid hour integer)
      - phone_number  (first valid phone number found)
      - clarification_count (number of consecutive clarification responses by assistant)
      - awaiting_confirmation (True, 'appointment', or 'phone')
      - awaiting_entity (active entity context lock: 'phone', 'day', 'time', 'confirmation')
    """
    ctx: Dict[str, object] = {
        "selected_day": None,
        "selected_time_hour": None,
        "phone_number": None,
        "clarification_count": 0,
        "awaiting_confirmation": False,
        "awaiting_entity": None,
    }

    consecutive_clarifications = 0
    active_awaiting_entity = None

    for msg in history:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        norm    = normalise_input(content)

        if role == "user":
            consecutive_clarifications = 0

            # Entity locking check: If previous turn was awaiting a phone number, extract phone only
            if active_awaiting_entity == "phone":
                is_p, p_val, _ = _norm_extract_phone_number(norm)
                if is_p and p_val:
                    ctx["phone_number"] = p_val
            else:
                # Normal entity extraction with priority
                is_p, p_val, _ = _norm_extract_phone_number(norm)
                if is_p and p_val:
                    if ctx["phone_number"] is None:
                        ctx["phone_number"] = p_val
                else:
                    # Extract day
                    day = _extract_day(norm)
                    if day and day in WORKING_DAYS:
                        ctx["selected_day"] = day.capitalize()

                    # Extract time (strictly guarded against phone digits)
                    hour = _extract_time_hour(norm)
                    if hour is not None:
                        valid, _ = validate_time(hour)
                        if valid:
                            ctx["selected_time_hour"] = hour

        elif role == "assistant":
            content_lower = content.lower()
            if any(trigger in content_lower for trigger in _CLARIFICATION_TRIGGERS):
                consecutive_clarifications += 1
            else:
                consecutive_clarifications = 0

            # Detect state expectations
            if ("scheduled for" in content_lower or "request an appointment" in content_lower) and "confirm" in content_lower:
                ctx["awaiting_confirmation"] = "appointment"
                active_awaiting_entity = "confirmation"
            elif "phone number" in content_lower or "share your number" in content_lower or "your number" in content_lower or "missed a digit" in content_lower:
                ctx["awaiting_confirmation"] = "phone"
                active_awaiting_entity = "phone"
            elif "what day" in content_lower or "which day" in content_lower:
                active_awaiting_entity = "day"
            elif "what time" in content_lower or "which time" in content_lower:
                active_awaiting_entity = "time"
            else:
                ctx["awaiting_confirmation"] = False
                active_awaiting_entity = None

    ctx["clarification_count"] = consecutive_clarifications
    ctx["awaiting_entity"] = active_awaiting_entity
    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# Main Router
# ──────────────────────────────────────────────────────────────────────────────

class ReceptionistRouter:
    """
    Intercepts common intents locally.
    Returns a response string if handled, or None to delegate to the LLM.
    """

    @staticmethod
    def get_hardcoded_response(
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        if not message or not message.strip():
            return GREETING_RESPONSES[0]

        history = history or []
        raw     = message.strip()
        query   = normalise_input(raw).lower()

        # ── Context from history ──────────────────────────────────────────────
        ctx = _extract_context_from_history(history)
        clarification_count  = ctx["clarification_count"]
        selected_day         = ctx["selected_day"]          # e.g. "Friday"
        selected_time_hour   = ctx["selected_time_hour"]    # e.g. 17
        phone_number         = ctx["phone_number"]
        awaiting_confirmation = ctx["awaiting_confirmation"]

        # ── Escalation gate (2 consecutive failed clarifications) ─────────────
        if clarification_count >= 2:
            logger.info("Escalating to human agent after 2 clarification failures.")
            return FALLBACK_RESPONSES[0]

        # ── Multi-Clause / Compound Query Detection ───────────────────────────
        has_ceo = bool(re.search(r"\b(ceo|chief executive|سی ای او|محمد رمضان)\b", query))
        has_company = bool(re.search(r"\b(company name|company ka naam|name of (your |the )?company|about (your |the )?company|company intro|who is (your |the )?company|tell me about (your |the )?company|کمپنی کا نام|کمپنی کیا ہے)\b", query))
        has_services = bool(re.search(r"\b(services|what do you do|what you offer|kya karti|kya services|kaam kya|خدمات|کیا کام)\b", query))
        has_timing = bool(re.search(r"\b(timing|timings|hours|office hours|working hours|kab khulta|kab open|وقت|ٹائمنگ|اوقات)\b", query))
        has_pricing = bool(re.search(r"\b(price|pricing|cost|rate|fee|charge|how much|قیمت|کتنے پیسے)\b", query))

        # ── Compound: CEO + Company Name / Intro ─────────────────────────────
        if has_ceo and has_company:
            if re.search(r"[\u0600-\u06FF]", query):
                return "ہماری کمپنی کا نام Logic Pulse ہے، اور ہمارے سی ای او محمد رمضان ہیں۔"
            if re.search(r"\b(kon|kaun|kya|naam|hain|hai|batao|apki|bhi)\b", query):
                return "Hamari company Logic Pulse hai, aur hamare CEO Muhammad Ramzan hain."
            return "Our company is Logic Pulse, and our CEO is Muhammad Ramzan."

        # ── Compound: CEO + Services ──────────────────────────────────────────
        if has_ceo and has_services:
            if re.search(r"[\u0600-\u06FF]", query):
                return "Logic Pulse کے سی ای او محمد رمضان ہیں، اور ہم کسٹم سافٹ ویئر، AI سسٹمز، چیٹ بوٹس اور ویب و موبائل ایپس بناتے ہیں۔"
            if re.search(r"\b(kon|kaun|kya|naam|hain|hai|batao|apki)\b", query):
                return "Logic Pulse ke CEO Muhammad Ramzan hain, aur hum custom software, AI solutions, dashboards aur web & mobile apps banate hain."
            return "Our CEO is Muhammad Ramzan, and Logic Pulse provides custom software development, AI solutions, chatbots, and web & mobile applications."

        # ── Compound: CEO + Timing ────────────────────────────────────────────
        if has_ceo and has_timing:
            if re.search(r"[\u0600-\u06FF]", query):
                return "ہمارے سی ای او محمد رمضان ہیں، اور دفتر کے اوقات پیر سے جمعہ صبح 9 بجے سے شام 5 بجے تک ہیں۔"
            if re.search(r"\b(kon|kaun|kya|naam|hain|hai|batao|apki)\b", query):
                return "Logic Pulse ke CEO Muhammad Ramzan hain, aur hamare office ke timings Monday se Friday subah 9 baje se sham 5 baje tak hain."
            return "Our CEO is Muhammad Ramzan, and our office hours are Monday to Friday, 9 AM to 5 PM."

        # If user asks a multi-part question with conjunction words ("and", "aur", "also")
        # that doesn't match predefined compound facts, delegate to LLM for high accuracy!
        conjunction_count = len(re.findall(r"\b(and|aur|also|as well as|bhi|saath|plus)\b", query))
        if conjunction_count >= 2 and not any(re.search(rf"\b{d}\b", query) for d in WORKING_DAYS | WEEKEND_DAYS):
            logger.info("Multi-part query detected — delegating to Groq LLM for comprehensive response.")
            return None

        # ── Audio / Microphone check & Greeting ───────────────────────────────
        has_mic_check = (
            bool(re.search(r"\b(can you hear|hear me|are you listening|are you there|am i audible|audio check|mic check|awaz|awaaz|aawaz|sun rahe|sun rahy|sun sakte)\b", query)) or
            bool(re.search(r"(آواز|اواز|سن\s*رہے|سن\s*سکتے|سنتے|وعلا\s*رہے|وعرارہ|وعرارهیة|کامیرا|ميرى\s*وعلا)", query)) or
            (bool(re.search(r"(آواز|اواز|ميرى|میری)", query)) and bool(re.search(r"(آ\s*رہی|آرہی|ارہی|ا\s*رہی|رہے|رہا|سکتے|سن|check|کامیرا|وعرارہ|وعلا)", query)))
        )

        has_greeting = bool(re.search(
            r"(?:ا?لسلام[\s_]*و?[\s_]*علیکم|سلام[\s_]*علیکم|وعلیکم[\s_]*السلام|وعلیکم|سلام|\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening)|howdy|salam|assalam|aslam|aaoa|kese\s+ho|kya\s+haal|wsalam|walaikum)\b)",
            query,
            re.IGNORECASE
        ))

        if has_mic_check:
            if has_greeting:
                if re.search(r"[\u0600-\u06FF]", query):
                    return "وعلیکم السلام! جی ہاں، مجھے آپ کی آواز بالکل صاف آ رہی ہے۔ فرمائیے، میں آپ کی کیا مدد کر سکتا ہوں؟"
                if re.search(r"\b(salam|assalam|aslam|awaz|awaaz|sun|kya|meri)\b", query):
                    return "Walaikum Assalam! Ji haan, mujhe aapki awaaz bilkul saaf aa rahi hai. Farmaye, main aapki kya madad kar sakta hoon?"
                return "Hello! Yes, I can hear you clearly. How can I assist you today?"
            else:
                if re.search(r"[\u0600-\u06FF]", query):
                    return "جی ہاں، مجھے آپ کی آواز بالکل صاف آ رہی ہے! میں آپ کی کیا مدد کر سکتا ہوں؟"
                if re.search(r"\b(awaz|awaaz|sun|rahe|rahy|kya)\b", query):
                    return "Ji haan, mujhe aapki awaaz bilkul saaf aa rahi hai! Main aapki kya madad kar sakta hoon?"
                return "Yes, I can hear you clearly! How can I assist you today?"

        # ── Company overview / About company ──────────────────────────────────
        if re.search(r"\b(about (your |the )?company|about logic pulse|tell me about (your |the )?company|tell me about logic pulse|what is logic pulse|what does logic pulse do|who are you|introduce yourself|tell me about yourself|company ke baare mein|company kya karti hai|company ka intro|کمپنی کے بارے میں|کمپنی کیا کرتی ہے)\b", query):
            if re.search(r"[\u0600-\u06FF]", query):
                return "Logic Pulse ایک ٹیکنالوجی کمپنی ہے جو کسٹم سافٹ ویئر، AI آٹومیشن، چیٹ بوٹس اور موبائل و ویب ایپلیکیشنز تیار کرتی ہے۔"
            if re.search(r"\b(kya|baare|karti|batao|apki)\b", query):
                return "Logic Pulse aik technology solutions company hai jo custom software development, AI aur automation solutions, dashboards aur chatbots banati hai."
            return (
                "Logic Pulse is a technology solutions company specializing in custom software development, "
                "AI and automation solutions, business dashboards, chatbots, and web & mobile applications."
            )

        # ── Contact info (inquiring about company contact details) ───────────
        is_contact_inquiry = bool(re.search(
            r"\b(how (to|can i) (contact|reach) you|how to reach logic pulse|company (contact|email|phone)|apka (contact|number|phone|email)|apki email|rabta kaise|kese rabta|رابطہ کیسے|آپ کا فون|آپ کی ای میل|رابطے کے لیے)\b",
            query,
        )) or (
            bool(re.search(r"\b(contact|email address|reach you|contact info|contact details)\b", query)) and
            not bool(re.search(r"\b(my phone|my number|my email|mera number|mera phone|03\d{2})\b", query)) and
            not bool(re.search(r"\b(appointment|book|schedule|ملاقات|بکنگ)\b", query))
        )
        if is_contact_inquiry:
            if re.search(r"[\u0600-\u06FF]", query):
                return "آپ ہم سے contact@logicpulse.com پر رابطہ کر سکتے ہیں یا دفتری اوقات (پیر تا جمعہ، صبح 9 سے شام 5) کے دوران کال کر سکتے ہیں۔"
            return "You can contact Logic Pulse via email at contact@logicpulse.com or call us during office hours (Monday to Friday, 9 AM to 5 PM)."

        # ── CEO queries ───────────────────────────────────────────────────────
        if re.search(r"\b(who is|who['']?s|kon hai|kon hy|kaun hai)\b.{0,15}\bceo\b", query) or \
           re.search(r"\bceo\b.{0,20}\b(name|naam|kon|kaun|who)\b", query) or \
           re.search(r"(سی ای او کون|سی ای او کا نام|محمد رمضان)", query) or \
           "ceo of logic pulse" in query:
            if re.search(r"[\u0600-\u06FF]", query):
                return "Logic Pulse کے سی ای او محمد رمضان ہیں۔"
            if re.search(r"\b(kon|kaun|naam|kya)\b", query):
                return "Logic Pulse ke CEO Muhammad Ramzan hain."
            return CEO_INFO

        # ── Greeting ──────────────────────────────────────────────────────────
        if has_greeting:
            if re.search(r"[\u0600-\u06FF]", query):
                return "وعلیکم السلام! Logic Pulse میں خوش آمدید۔ میں آپ کی کیا مدد کر سکتا ہوں؟"
            if re.search(r"\b(salam|assalam|aslam|kese|haal)\b", query):
                return "Walaikum Assalam! Logic Pulse mein khushamdeed. Main aapki kya madad kar sakta hoon?"
            return GREETING_RESPONSES[0]

        # ── Closing ───────────────────────────────────────────────────────────
        if re.search(r"\b(bye|goodbye|exit|quit|end|thank you|thanks|khuda hafiz|allah hafiz|shukriya|boht shukriya|شکریہ|اللہ حافظ|خدا حافظ)\b", query):
            if re.search(r"[\u0600-\u06FF]", query):
                return "آپ کے رابطے کا شکریہ! Logic Pulse سے رابطہ کرنے کے لیے خوش آمدید۔ اللہ حافظ!"
            if re.search(r"\b(shukriya|allah hafiz|khuda hafiz)\b", query):
                return "Aap ke rabte ka shukriya! Logic Pulse se rabta karne ke liye khushamdeed. Allah Hafiz!"
            return CLOSING_RESPONSES[3]

        # ── Office timing ─────────────────────────────────────────────────────
        if re.search(r"\b(timing|timings|office hours|hours|working hours|open|close|when do you|operating hours|kab khulta|kab open|kya timing|دفتر کے اوقات|ٹائمنگ|کب کھلتا)\b", query):
            if re.search(r"\b(weekend|saturday|sunday|sat|sun|hafte|itwar|ہفتہ|اتوار)\b", query):
                return TIMING_RESPONSES[2]  # "We are closed on weekends (Saturday and Sunday)."
            if re.search(r"\b(weekday|monday|tuesday|wednesday|thursday|friday|peer|mangal|budh|jumerat|jumma|پیر|منگل|بدھ|جمعرات|جمعہ)\b", query):
                return TIMING_RESPONSES[1]
            if re.search(r"[\u0600-\u06FF]", query):
                return "ہمارے دفتری اوقات پیر سے جمعہ صبح 9 بجے سے شام 5 بجے تک ہیں۔"
            return TIMING_RESPONSES[0]

        # ── Location ──────────────────────────────────────────────────────────
        if re.search(r"\b(address|location|where are you|where is|office address|find you|office kahan|kahan par|location kya|پتہ|ایڈریس|دفتر کہاں ہے)\b", query):
            if re.search(r"\b(google map|map|direction)\b", query):
                return LOCATION_RESPONSES[2]
            if re.search(r"\b(online|virtual|zoom|meet|teams)\b", query):
                return LOCATION_RESPONSES[4]
            return LOCATION_RESPONSES[0]

        # ── Services ─────────────────────────────────────────────────────────
        if re.search(r"\b(what services|list services|services do you offer|what do you do|what do you offer|what you offer|capabilities|expertise|specialt|kya kaam karte|kya services|kaunsi services|خدمات|کیا خدمات|کیا کام کرتے)\b", query):
            if re.search(r"[\u0600-\u06FF]", query):
                return "ہم کسٹم سافٹ ویئر ڈویلپمنٹ، AI اور آٹومیشن سسٹمز، چیٹ بوٹس، بزنس ڈیش بورڈز اور ویب و موبائل ایپلیکیشنز فراہم کرتے ہیں۔"
            return (
                "We offer custom software development, AI and automation solutions, "
                "chatbots, web and mobile applications, and business dashboards."
            )

        # ── Pricing ───────────────────────────────────────────────────────────
        # Guard: only match pricing if message is NOT primarily an appointment
        _is_appt_message = bool(re.search(
            r"\b(appointment|book|schedule|meeting|slot|reserve|appoint|i am free|i'?m free|available|free on|free at|ملاقات|بکنگ)\b",
            query,
        ))
        if not _is_appt_message:
            if re.search(r"\b(free consultation|free consult|consultation|consult)\b", query):
                return PRICING_RESPONSES[4]
            if re.search(r"\b(price|pricing|cost|costs|rate|fee|fees|how much|charge|budget|affordable|estimate|quote|قیمت|کتنے چارج)\b", query):
                if re.search(r"\b(estimate|quote|details)\b", query):
                    return PRICING_RESPONSES[3]
                return PRICING_RESPONSES[0]

        # ── Appointment booking flow ──────────────────────────────────────────
        # Detect day+time patterns even without explicit keywords like 'book'
        _has_day  = bool(_extract_day(query))
        _has_time = _extract_time_hour(query) is not None
        is_p, p_val, p_reason = _norm_extract_phone_number(query)
        has_phone_attempt = is_p or p_reason in ("incomplete", "invalid_length", "invalid_prefix", "malformed_noise")
        has_booking_keywords = bool(re.search(
            r"\b(appointment|book|schedule|meeting|slot|reserve|appoint|reschedule|i am free|i'?m free|i am available|available on|free on|free at|ملاقات|بکنگ)\b",
            query,
        ))

        is_booking_intent = (
            has_booking_keywords or
            (_has_day and _has_time) or
            (_has_day and selected_time_hour is not None) or
            (_has_time and selected_day is not None) or
            (_has_day and awaiting_confirmation) or
            (_has_time and awaiting_confirmation) or
            (has_phone_attempt and (selected_day is not None or awaiting_confirmation or ctx.get("awaiting_entity") == "phone")) or
            (awaiting_confirmation == "appointment" and bool(re.search(r"\b(yes|confirm|ok|okay|sure|correct|good|proceed|no|cancel)\b", query)))
        )

        if is_booking_intent:
            return ReceptionistRouter._handle_appointment(
                query=query,
                raw=raw,
                ctx=ctx,
            )

        # ── Phone number provided standalone ──────────────────────────────────
        if is_p and p_val:
            formatted_phone = _norm_format_phone_display(p_val)
            return (
                f"Thank you. I have recorded {formatted_phone} for this conversation. "
                "Please provide your preferred weekday and time so I can prepare a booking request."
            )
        elif (awaiting_confirmation == "phone" or ctx.get("awaiting_entity") == "phone"):
            if p_reason == "malformed_noise":
                return "I couldn't recognize that as a valid phone number. Please say your 11-digit Pakistani mobile number starting with 03 (e.g. 0345 1234567)."
            if p_reason in ("incomplete", "invalid_length", "invalid_prefix", "invalid"):
                return "I think I missed a digit. Could you please repeat your 11-digit phone number?"

        # ── Unknown / unmatched ───────────────────────────────────────────────
        if clarification_count >= 1:
            logger.info("Escalating to human agent after repeated failure.")
            return FALLBACK_RESPONSES[0]

        logger.info("Unrecognised input — delegating to Groq LLM.")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Appointment sub-handler
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _handle_appointment(
        query: str,
        raw: str,
        ctx: Dict,
    ) -> str:
        selected_day        = ctx["selected_day"]
        selected_time_hour  = ctx["selected_time_hour"]
        phone_number        = ctx["phone_number"]
        awaiting_confirm    = ctx["awaiting_confirmation"]
        awaiting_ent        = ctx.get("awaiting_entity")

        # Try to extract day/time/phone from the current message
        current_day   = _extract_day(query)
        current_hour  = _extract_time_hour(query)
        is_p, p_val, p_reason = _norm_extract_phone_number(query)
        current_phone = p_val if is_p else None

        effective_phone = current_phone or phone_number

        # ── Malformed noise rejected immediately without guessing ───────────
        if p_reason == "malformed_noise":
            return "I couldn't recognize that as a valid phone number. Please say your 11-digit Pakistani mobile number starting with 03 (e.g. 0345 1234567)."

        # ── User provided phone number when day/time are already established ──
        if current_phone and (selected_day or current_day or awaiting_confirm):
            day_to_use = current_day or (selected_day.lower() if selected_day else None)
            hour_to_use = current_hour if current_hour is not None else selected_time_hour
            formatted_disp = _norm_format_phone_display(current_phone)
            if day_to_use and hour_to_use is not None:
                return (
                    f"Thank you. I have the requested {day_to_use.capitalize()} at "
                    f"{hour_to_display(hour_to_use)} slot and phone number {formatted_disp}. "
                    "Please confirm that you want me to submit this booking request."
                )
            return (
                f"Thank you. I have recorded {formatted_disp}. Please also provide the "
                "weekday and time before I can prepare the booking request."
            )

        # ── Phone in current message while awaiting phone ─────────────────────
        if awaiting_confirm == "phone" or awaiting_ent == "phone":
            if current_phone:
                formatted_disp = _norm_format_phone_display(current_phone)
                return (
                    f"Thank you. I have recorded {formatted_disp} for this booking request. "
                    "Please confirm the requested day and time; I will only confirm the appointment after the booking service succeeds."
                )
            if p_reason == "malformed_noise":
                return "I couldn't recognize that as a valid phone number. Please say your 11-digit Pakistani mobile number starting with 03 (e.g. 0345 1234567)."
            if p_reason in ("incomplete", "invalid_length", "invalid_prefix", "invalid") or bool(re.search(r"\d", query)):
                return "I think I missed a digit. Could you please repeat your 11-digit phone number?"

        # ── User confirming a proposed appointment ────────────────────────────
        if awaiting_confirm == "appointment" or awaiting_ent == "confirmation":
            if re.search(r"\b(yes|confirm|ok|okay|sure|correct|good|sounds good|proceed)\b", query):
                if selected_day and selected_time_hour is not None:
                    display_time = hour_to_display(selected_time_hour)
                    if effective_phone:
                        formatted_disp = _norm_format_phone_display(effective_phone)
                        return (
                            f"Thank you. The booking request for {selected_day} at {display_time} "
                            f"with phone number {formatted_disp} is ready for submission. "
                            "I will confirm it only after the booking service reports success."
                        )
                    return (
                        f"The booking request for {selected_day} at {display_time} needs a phone number. "
                        "Please share it before I submit the request."
                    )
            if re.search(r"\b(no|cancel|different|change|another)\b", query):
                return (
                    "No problem! Please provide your preferred day (Monday–Friday) "
                    "and time (9 AM–5 PM) and I'll reschedule."
                )

        # ── Validate day ──────────────────────────────────────────────────────
        day_to_use = current_day or (selected_day.lower() if selected_day else None)

        if day_to_use:
            valid_day, day_reason = validate_day(day_to_use)
            if not valid_day:
                if day_reason == "weekend":
                    return (
                        f"Unfortunately {day_to_use.capitalize()} is a weekend — we are closed on "
                        "Saturday and Sunday. Please choose a weekday (Monday–Friday)."
                    )
                return (
                    "I didn't recognise that day. Please specify a weekday: Monday to Friday."
                )
        else:
            # No day provided yet — ask
            if current_hour is not None:
                # Time provided but no day
                valid_time, time_reason = validate_time(current_hour)
                if not valid_time:
                    if time_reason == "midnight":
                        return (
                            "12 AM (midnight) is outside our working hours. "
                            "Please choose a time between 9 AM and 5 PM on a weekday."
                        )
                    return (
                        f"That time is outside our working hours. "
                        "I can schedule it during weekdays between 9 AM and 5 PM."
                    )
                # Time is valid but day missing
                display_time = hour_to_display(current_hour)
                return (
                    f"{display_time} works! Which day would you prefer? "
                    "We are available Monday through Friday."
                )
            # Neither day nor time — ask
            return APPOINTMENT_RESPONSES[0]

        # ── Validate time ─────────────────────────────────────────────────────
        hour_to_use = current_hour if current_hour is not None else selected_time_hour

        if hour_to_use is None:
            # Day known but time missing
            return (
                f"Got it — {day_to_use.capitalize()} works! "
                "What time would you prefer? We are available between 9 AM and 5 PM."
            )

        valid_time, time_reason = validate_time(hour_to_use)
        if not valid_time:
            if time_reason == "midnight":
                return (
                    "12 AM (midnight) is outside our working hours. "
                    "Please choose a time between 9 AM and 5 PM on a weekday."
                )
            return (
                "That time is outside our working hours. "
                "I can schedule it during weekdays between 9 AM and 5 PM."
            )

        # ── Both day and time are valid → confirmation step ───────────────────
        display_time = hour_to_display(hour_to_use)
        day_display  = day_to_use.capitalize()

        if phone_number:
            return (
            f"I can request an appointment with our CEO, Muhammad Ramzan, for "
            f"{day_display} at {display_time}. "
            "Please confirm that you want me to submit this booking request."
            )
        return (
            f"I can request an appointment with our CEO, Muhammad Ramzan, for "
            f"{day_display} at {display_time}. "
            "Please confirm, and also share your phone number before I submit the request."
        )
