# Project level settings / configurations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VERBOSE: bool = True
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # Qdrant stuff
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "ViralLens"


# Singleton settings class, do not reinstatiate
settings = Settings()
