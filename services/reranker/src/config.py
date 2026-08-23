from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Reranker service settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    hf_token: str | None = None  # model is not gated; only needed for private/gated models

    grpc_port: int = 8352
    grpc_host: str = "0.0.0.0"
    device: str = "auto"  # auto, cuda, cpu

    log_level: str = "INFO"


settings = Settings()
