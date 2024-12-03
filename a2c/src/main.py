from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from src.api.main import api_router
from src.config.settings import Settings


app = FastAPI(
    title=Settings.PROJECT_NAME,
    openapi_url=f"{Settings.app_version}/openapi.json"
)