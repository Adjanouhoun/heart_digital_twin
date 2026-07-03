from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    environment: str = Field(default="development")
    database_url: str = Field(default="postgresql+asyncpg://cdt:cdt_local_2024@db:5432/cdt")
    database_url_sync: str = Field(default="postgresql+psycopg2://cdt:cdt_local_2024@db:5432/cdt")
    minio_endpoint: str = Field(default="minio:9000")
    minio_access_key: str = Field(default="cdt_admin")
    minio_secret_key: str = Field(default="cdt_minio_2024")
    minio_use_ssl: bool = Field(default=False)
    bucket_dicom: str = Field(default="cdt-dicom")
    bucket_ecg: str = Field(default="cdt-ecg")
    bucket_eam: str = Field(default="cdt-eam")
    redis_url: str = Field(default="redis://:cdt_redis_2024@redis:6379/0")
    mlflow_tracking_uri: str = Field(default="http://localhost:5001")
    twin_id_seed: str = Field(default="CHANGE_ME_IN_PRODUCTION")
    ingestion_timeout_seconds: float = Field(default=2.0)
    max_dicom_size_mb: int = Field(default=2048)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
