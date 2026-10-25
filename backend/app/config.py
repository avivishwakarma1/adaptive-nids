from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    nids_env: str = "development"
    nids_api_prefix: str = "/api/v1"
    mongodb_uri: str = ""
    mongodb_database: str = "adaptive_nids"
    jwt_secret_key: str = ""
    jwt_access_token_minutes: int = Field(default=30, ge=1, le=1440)
    jwt_issuer: str = "adaptive-ai-nids"
    jwt_audience: str = "adaptive-ai-nids-api"
    cors_origins: str = "http://localhost:5173"
    max_upload_mb: int = Field(default=25, ge=1, le=512)
    max_rows_per_dataset: int = Field(default=100000, ge=1)
    artifact_storage_dir: str = "./var/artifacts"
    model_artifact_dir: str = "./var/models"
    active_model_version: str = "unavailable"
    log_level: str = "INFO"
    allow_degraded_model: bool = True

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_secret(cls, value: str) -> str:
        # Empty is allowed for health-only startup/tests; token issuance refuses it.
        if value and len(value.encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET_KEY must contain at least 32 bytes")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()
