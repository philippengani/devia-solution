from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    description='LangGraph-based e-commerce market analysis agent for the DevIA take-home exercise.',
    version='2.0.0',
)

app.include_router(router)
