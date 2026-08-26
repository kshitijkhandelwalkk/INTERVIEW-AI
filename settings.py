from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration.
    Environment variables are loaded automatically from a .env file.
    Defaults target a local LM Studio + speech-to-speech setup.
    """

    # LLM (scenario generation + feedback scoring, and the s2s pipeline brain).
    # Defaults to LM Studio's OpenAI-compatible server.
    LLM_BASE_URL: str = Field("http://127.0.0.1:1234/v1", description="OpenAI-compatible base URL")
    LLM_MODEL: str = Field("google/gemma-4-e4b", description="Model name")
    LLM_API_KEY: SecretStr = Field(SecretStr("lm-studio"), description="API key (LM Studio ignores it)")

    # speech-to-speech pipeline (Hugging Face speech-to-speech, `s2s` conda env)
    S2S_CONDA_ENV: str = Field("s2s", description="Conda env name containing speech-to-speech")
    S2S_PYTHON: str = Field(
        "", description="Explicit python.exe of the s2s env; overrides S2S_CONDA_ENV lookup when set"
    )
    S2S_WS_HOST: str = Field("127.0.0.1", description="Host the s2s WebSocket streamer binds to")
    S2S_WS_PORT: int = Field(8765, description="Port for the s2s WebSocket streamer")
    S2S_STT: str = Field("parakeet-tdt", description="STT backend for the s2s pipeline")
    S2S_TTS: str = Field("qwen3", description="TTS backend for the s2s pipeline")
    S2S_TTS_REF_AUDIO: str = Field(
        str(Path.home() / "Desktop" / "main_voice.wav"),
        description="Reference audio for qwen3 voice cloning; blank = stock speaker",
    )
    S2S_DEVICE: str = Field("", description="Optional device override for s2s handlers (e.g. cuda)")
    S2S_EXTRA_ARGS: str = Field("", description="Extra CLI args appended to the s2s command")

    # LiveAvatar (optional avatar / fully-online mode)
    LIVEAVATAR_API_KEY: SecretStr = Field(SecretStr(""), description="LiveAvatar API key")
    LIVEAVATAR_API_URL: str = Field("https://api.liveavatar.com", description="LiveAvatar API base URL")
    LIVEAVATAR_AVATAR_ID: str = Field("", description="Default avatar id (blank = first public avatar)")
    LIVEAVATAR_SANDBOX: bool = Field(False, description="Use LiveAvatar sandbox sessions (no credits)")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
