from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Receptionist"
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_receptionist"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_max_tokens: int = 200
    short_memory_messages: int = 12
    allowed_origins: str = "http://127.0.0.1:8000,http://localhost:8000"
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    admin_api_key: str = ""
    voice_websocket_token: str = ""
    public_base_url: str = ""
    voice_enabled: bool = True
    stt_provider: str = "groq_whisper"  # "faster_whisper", "groq_whisper"
    groq_whisper_model: str = "whisper-large-v3-turbo"
    whisper_model: str = "base"
    whisper_language: str = "en"
    vocabulary_corrections: list[str] = [
        "logicals",
        "logical",
        "ligit pulse",
        "logic pals",
        "logic pulls",
        "lodge pulse",
    ]
    vocabulary_target: str = "Logic Pulse"
    vad_min_rms: float = 100.0
    min_stt_rms: float = 70.0
    min_stt_duration_ms: int = 400
    vad_aggressiveness: int = 3
    vad_speech_threshold: float = 0.3
    vad_silence_duration_ms: int = 450
    vad_min_speech_duration_ms: int = 60
    vad_max_turn_duration_ms: int = 15000
    vad_pre_buffer_ms: int = 200
    vad_post_buffer_ms: int = 150
    chunk_duration_ms: int = 500
    tts_voice: str = "en-US-JennyNeural"
    silence_timeout_sec: int = 30
    voice_max_packet_bytes: int = 64 * 1024
    voice_max_buffer_bytes: int = 1024 * 1024

    outbox_retry_base_seconds: int = 5
    outbox_retry_max_seconds: int = 300
    outbox_handler_timeout_seconds: float = 30.0

    # Vonage Configuration
    vonage_api_key: str = ""
    vonage_api_secret: str = ""
    vonage_from_number: str = ""

    # Email Configuration
    email_provider: str = "smtp"  # "smtp", "sendgrid", "resend"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    sendgrid_api_key: str = ""
    resend_api_key: str = ""
    from_email: str = "noreply@logicpulse.com"

    # Telemetry and VAD / Barge-In Configuration
    telemetry_log_file: str = "logs/telemetry.jsonl"
    barge_in_guard_ms: int = 800
    barge_in_min_speech_ms: int = 450
    barge_in_min_rms: float = 120.0
    barge_in_rms_multiplier: float = 3.0
    demo_mode: bool = False

    # SMS Configuration
    sms_rate_limit_count: int = 5
    sms_rate_limit_window_seconds: int = 300


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def VOICE_ENABLED(self) -> bool:
        return self.voice_enabled

    @property
    def WHISPER_MODEL(self) -> str:
        return self.whisper_model

    @property
    def VAD_AGGRESSIVENESS(self) -> int:
        return self.vad_aggressiveness

    @property
    def CHUNK_DURATION_MS(self) -> int:
        return self.chunk_duration_ms

    @property
    def TTS_VOICE(self) -> str:
        return self.tts_voice

    @property
    def SILENCE_TIMEOUT_SEC(self) -> int:
        return self.silence_timeout_sec


settings = Settings()
