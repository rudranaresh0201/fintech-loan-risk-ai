"""FastAPI entrypoint for EMI Predict AI backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.finance import router as finance_router
from backend.routes.health import router as health_router
from config.settings import get_settings


settings = get_settings()
app = FastAPI(
    title=settings.project_name,
    version=settings.project_version,
    description="EMI eligibility and EMI amount prediction APIs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(finance_router)
