# Project level settings / configurations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VERBOSE: bool = True
    DATA_DIR: str = "data/raw"

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # Qdrant stuff
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "ViralLens"

    # Retriever settings
    SEARCH_TYPE: str = "mmr"
    SEARCH_K: int = 10

    # LLM stuff
    LLM_API_BASE: str = "http://127.0.0.1:1234/v1"
    # LLM_MODEL: str = "google/gemma-4-e4b"
    LLM_MODEL: str = "nvidia/nemotron-3-nano-4b"

    LLM_TEMPERATURE: float = 0.7


# Singleton settings class, do not reinstatiate
settings = Settings()
