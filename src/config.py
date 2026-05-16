# Project level settings / configurations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VERBOSE: bool = True


# Singleton settings class, do not reinstatiate
settings = Settings()
