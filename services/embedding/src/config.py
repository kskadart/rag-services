from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Embedding service settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    embedding_model_name: str = "google/embeddinggemma-300m"
    hf_token: str

    grpc_port: int = 8351
    grpc_host: str = "0.0.0.0"
    device: str = "auto"  # auto, cuda, cpu

    log_level: str = "INFO"


settings = Settings()
