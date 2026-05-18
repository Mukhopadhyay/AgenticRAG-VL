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
    SEARCH_K: int = 5

    # LLM stuff - set LLM_MODEL to a LiteLLM-compatible model string.
    LLM_MODEL: str = "groq/llama-3.1-8b-instant"
    # For overriding the API base
    LLM_API_BASE: str | None = None
    # Explicit API key, if not already set
    LLM_API_KEY: str | None = None
    LLM_TEMPERATURE: float = 0.7

    MAX_RETRIES: int = 2


# Singleton settings class, do not reinstatiate
settings = Settings()
