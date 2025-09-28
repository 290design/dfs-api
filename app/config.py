from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "DFS API Service"
    DEBUG: bool = False

    # Database (for Phase 2)
    DATABASE_URL: Optional[str] = None

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Optimizer
    OPTIMIZER_TIMEOUT: int = 120
    MAX_LINEUPS: int = 100

    class Config:
        env_file = ".env"


settings = Settings()