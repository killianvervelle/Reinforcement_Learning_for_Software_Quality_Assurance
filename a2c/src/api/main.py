from fastapi import APIRouter

from api.routes import a2c


api_router = APIRouter()
api_router.include_router(a2c.router)