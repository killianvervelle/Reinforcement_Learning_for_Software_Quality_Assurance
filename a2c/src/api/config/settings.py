from pydantic import AnyUrl, BaseSettings
from typing import Union


class Settings(BaseSettings):
    APP_NAME: str = "A2C-RL"
    API_V1_STR: str = "/api/v1"
    ENV_FILE = "../.env"

    BACKEND_CORS_ORIGINS: Union[
        list[AnyUrl] | str
    ] = []


settings = Settings(BACKEND_CORS_ORIGINS=[
                    "http://localhost:8080", "http://localhost:3000"])
