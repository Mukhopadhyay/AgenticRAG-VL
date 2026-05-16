# Project level settings / configurations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VERBOSE: bool = True
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"


# Singleton settings class, do not reinstatiate
settings = Settings()
