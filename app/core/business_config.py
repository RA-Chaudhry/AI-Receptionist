from typing import Any
from pydantic import BaseModel, Field


class BusinessConfig(BaseModel):
    business_name: str = Field(default="Logic Pulse", description="Company or business name")
    ceo_name: str = Field(default="Muhammad Ramzan", description="CEO or principal contact name")
    assistant_name: str = Field(default="Pulse Concierge AI", description="AI Receptionist persona name")
    working_days: list[str] = Field(
        default_factory=lambda: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        description="Authorized working days for scheduling",
    )
    working_hours_start: int = Field(default=9, description="Working hours start (24h format, 9 for 9 AM)")
    working_hours_end: int = Field(default=17, description="Working hours end (24h format, 17 for 5 PM)")
    timezone: str = Field(default="Asia/Karachi", description="Operating timezone")
    services: list[str] = Field(
        default_factory=lambda: [
            "Custom Software Development",
            "AI and Automation Solutions",
            "Business Systems & Dashboards",
            "Chatbots and AI Agents",
            "Web and Mobile Applications",
        ],
        description="Offered business services",
    )
    contact_phone: str = Field(default="+92 300 1234567", description="Business contact phone")
    contact_email: str = Field(default="contact@logicpulse.com", description="Business contact email")
    appointment_rules: dict[str, Any] = Field(
        default_factory=lambda: {
            "requires_phone": True,
            "confirmation_required": True,
            "max_advance_days": 30,
        },
        description="Business appointment constraints",
    )


_current_business_config: BusinessConfig = BusinessConfig()


def get_business_config() -> BusinessConfig:
    """Returns the globally active BusinessConfig."""
    return _current_business_config


def set_business_config(config: BusinessConfig) -> None:
    """Updates the globally active BusinessConfig."""
    global _current_business_config
    _current_business_config = config


def reset_business_config() -> None:
    """Resets to default Logic Pulse configuration."""
    global _current_business_config
    _current_business_config = BusinessConfig()
