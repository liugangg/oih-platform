"""
OIH Platform Configuration
"""
import os
from pydantic_settings import BaseSettings
from pydantic import model_validator
from pathlib import Path


class Settings(BaseSettings):
    # Server
    SERVER_HOST: str = os.environ.get("OIH_SERVER_HOST", "localhost")
    API_PORT: int = 8080

    # GPU Assignment
    QWEN_GPU: str = "0"       # GPU0 → Qwen3-14B
    COMPUTE_GPU: str = "1"    # GPU1 → All bio tools

    # Qwen endpoint (vLLM / Ollama on GPU0) — derived from SERVER_HOST if not set
    QWEN_BASE_URL: str = ""
    QWEN_MODEL: str = "Qwen3-14B"

    # OIH API base URL — derived from SERVER_HOST if not set
    OIH_API_BASE: str = ""

    # LLM Backend: "local" | "anthropic" | "openai" | "openrouter"
    LLM_PROVIDER: str = os.environ.get("OIH_LLM_PROVIDER", "local")
    LLM_API_KEY: str = os.environ.get("OIH_LLM_API_KEY", "")
    LLM_MODEL: str = os.environ.get("OIH_LLM_MODEL", "")       # empty = provider default
    LLM_BASE_URL: str = os.environ.get("OIH_LLM_BASE_URL", "")  # custom endpoint
    OPENROUTER_API_KEY: str = ""  # OpenRouter API key (read from .env)

    # Shared data volume (mounted in all containers)
    DATA_ROOT: str = "/data/oih"
    INPUT_DIR: str = "/data/oih/inputs"
    OUTPUT_DIR: str = "/data/oih/outputs"
    TEMP_DIR: str = "/data/oih/tmp"

    # Docker container names (must match docker-compose service names)
    CONTAINER_ALPHAFOLD3: str = "oih-alphafold3"
    CONTAINER_RFDIFFUSION: str = "oih-rfdiffusion"
    CONTAINER_PROTEINMPNN: str = "oih-proteinmpnn"
    CONTAINER_BINDCRAFT: str = "oih-bindcraft"
    CONTAINER_FPOCKET: str = "oih-fpocket"
    CONTAINER_P2RANK: str = "oih-p2rank"
    CONTAINER_VINA_GPU: str = "oih-vina-gpu"
    CONTAINER_AUTODOCK_GPU: str = "oih-autodock-gpu"
    CONTAINER_GNINA: str = "oih-gnina"
    CONTAINER_DIFFDOCK: str = "oih-diffdock"
    CONTAINER_GROMACS: str = "oih-gromacs"

    # Task timeouts (seconds)
    TIMEOUT_ALPHAFOLD3: int = 7200      # 2h
    TIMEOUT_RFDIFFUSION: int = 7200     # 2h (large targets with many hotspots need more time)
    TIMEOUT_PROTEINMPNN: int = 600      # 10min
    TIMEOUT_BINDCRAFT: int = 7200       # 2h
    TIMEOUT_FPOCKET: int = 60
    TIMEOUT_P2RANK: int = 120
    TIMEOUT_DOCKING: int = 1800         # 30min
    TIMEOUT_DIFFDOCK: int = 1800
    TIMEOUT_GROMACS: int = 86400        # 24h

    CONTAINER_ESM: str = "oih-esm"
    CONTAINER_CHEMPROP: str = "oih-chemprop"
    CONTAINER_DISCOTOPE3: str = "oih-discotope3"
    CONTAINER_IGFOLD: str = "oih-igfold"
    TIMEOUT_ESM: int = 1800      # ESM-1v mutant scan can be slow on first load
    TIMEOUT_CHEMPROP: int = 3600
    TIMEOUT_DISCOTOPE3: int = 600
    TIMEOUT_IGFOLD: int = 300       # ~2s/seq but batch may be large

    @model_validator(mode="after")
    def derive_urls(self) -> "Settings":
        if not self.QWEN_BASE_URL:
            self.QWEN_BASE_URL = f"http://{self.SERVER_HOST}:8002/v1"
        if not self.OIH_API_BASE:
            self.OIH_API_BASE = f"http://{self.SERVER_HOST}:8080/api/v1"
        return self

    class Config:
        env_file = ".env"


settings = Settings()
