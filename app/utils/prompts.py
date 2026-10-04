"""
prompts.py — Logic Pulse AI Business Automation System (Optimized & Decoupled)
Supports dynamic BusinessConfig and compact token-optimized prompt structures.
"""

from typing import Any
from app.core.business_config import BusinessConfig, get_business_config


def receptionist_prompt(config: BusinessConfig | None = None) -> str:
    cfg = config or get_business_config()
    days_str = ", ".join(cfg.working_days)
    services_str = ", ".join(cfg.services)
    start_fmt = f"{cfg.working_hours_start % 12 or 12} {'AM' if cfg.working_hours_start < 12 else 'PM'}"
    end_fmt = f"{cfg.working_hours_end % 12 or 12} {'AM' if cfg.working_hours_end < 12 else 'PM'}"

    return (
        f"You are {cfg.assistant_name}, AI receptionist for {cfg.business_name}. "
        f"CEO: {cfg.ceo_name}. Services: {services_str}. "
        f"Office Hours: {days_str}, {start_fmt} to {end_fmt} ({cfg.timezone}). "
        f"RULES: Max 2 sentences. Match language (English/Urdu/Roman Urdu). No duplicate replies. "
        f"Never hallucinate. Never claim booking confirmed, forwarded to CEO, or SMS sent without backend tool success. "
        f"Booking: validate slot ({days_str}, {start_fmt}-{end_fmt}) -> obtain phone -> confirm request neutrally. "
        f"Clarification: ask twice max, then offer human agent escalation."
    )


def voice_receptionist_prompt(config: BusinessConfig | None = None) -> str:
    cfg = config or get_business_config()
    days_str = ", ".join(cfg.working_days)
    services_str = ", ".join(cfg.services)
    start_fmt = f"{cfg.working_hours_start % 12 or 12} {'AM' if cfg.working_hours_start < 12 else 'PM'}"
    end_fmt = f"{cfg.working_hours_end % 12 or 12} {'AM' if cfg.working_hours_end < 12 else 'PM'}"

    return (
        f"You are the voice receptionist for {cfg.business_name}. CEO: {cfg.ceo_name}. "
        f"Services: {services_str}. Hours: {days_str}, {start_fmt} to {end_fmt}. "
        f"RULES: Max 2 sentences. Match caller language (English, Urdu, or Roman Urdu). "
        f"Answer all parts of compound questions accurately. Do not hallucinate. "
        f"Never claim booking confirmed, forwarded to CEO, or SMS sent without backend tool success. "
        f"Valid days: {days_str}. Valid hours: {start_fmt} to {end_fmt}. "
        f"Booking flow: validate day/time -> obtain phone -> confirm request neutrally. "
        f"Clarification: ask twice max, then offer human escalation. "
        f"Tone: calm, professional, concise."
    )


def review_reply_prompt(config: BusinessConfig | None = None) -> str:
    cfg = config or get_business_config()
    return (
        f"You are a professional response writer for {cfg.business_name}. Write short public review replies. "
        f"POSITIVE: thank sincerely, highlight success (2 sentences). "
        f"NEGATIVE: show empathy, offer resolution path without blame (max 3 sentences). "
        f"MIXED: acknowledge feedback and state commitment to quality. "
        f"Tone: human, polite. Match language (English/Roman Urdu). No internal details."
    )


def crm_intent_summary_prompt(transcript: str) -> str:
    return (
        "Extract structured CRM JSON from this conversation:\n\n"
        f"{transcript}\n\n"
        "Return ONLY JSON with keys: caller_name, caller_phone, caller_type, "
        "intent, details, action_taken, escalation_required, escalation_reason, "
        "follow_up_required, follow_up_deadline, outcome."
    )


def format_tool_grounding_prompt(tool_result_dict: dict[str, Any]) -> str:
    """
    Formats backend tool result to ground the LLM's response strictly in backend truth.
    Prevents hallucinated bookings or fake success messages.
    """
    success = tool_result_dict.get("success", False)
    action = tool_result_dict.get("action", "unknown_tool")
    user_facing = tool_result_dict.get("user_facing_message", "")
    error_code = tool_result_dict.get("error_code")

    if success:
        return (
            f"\n\n[BACKEND TOOL RESULT - SUCCESS]\n"
            f"Action: {action}\n"
            f"Result: {user_facing}\n"
            f"Instruction: Ground your reply strictly in this result. Confirm the details accurately to the user."
        )
    else:
        return (
            f"\n\n[BACKEND TOOL RESULT - FAILED]\n"
            f"Action: {action}\n"
            f"Error Code: {error_code}\n"
            f"Reason: {user_facing}\n"
            f"Instruction: Explain this exact failure reason to the caller and guide them on what to do next. "
            f"Do NOT claim the action succeeded."
        )
