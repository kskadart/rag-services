"""Configuration management for embedding service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Embedding service settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Model settings (server-level, via env/compose)
    embedding_model_name: str = "google/embeddinggemma-300m"
    
    # Server settings (via env/compose)
    grpc_port: int = 8351
    grpc_host: str = "0.0.0.0"
    device: str = "auto"  # auto, cuda, cpu
    
    # Processing parameters are controlled by client per request
    # No server-level defaults for max_length, batch_size, normalize, pooling_strategy
    
    # Logging (via env/compose)
    log_level: str = "INFO"


# Global settings instance
settings = Settings()
