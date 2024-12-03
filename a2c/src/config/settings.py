import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "A2C-RL"
    app_version: str = "1.0.0"
    debug: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings