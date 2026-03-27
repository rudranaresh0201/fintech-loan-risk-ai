"""Health route for service observability."""

from __future__ import annotations

from fastapi import APIRouter

from backend.models.model_loader import load_artifacts
from schemas.responses import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report service and artifact-loading status."""

    try:
        _ = load_artifacts()
    except Exception as exc:
        return HealthResponse(
            status="degraded",
            models_loaded=False,
            message=f"Model artifacts are unavailable: {exc}",
        )

    return HealthResponse(
        status="ok",
        models_loaded=True,
        message="Service is healthy and models are loaded.",
    )
