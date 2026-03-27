"""Response schemas for API endpoints."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Contribution(BaseModel):
    """Single feature contribution used for explainability."""

    feature: str
    contribution: float


class PredictionExplainability(BaseModel):
    """Explainability details for model output and affordability."""

    rule_based_insights: List[str]
    classification_top_contributors: List[Contribution]
    regression_top_contributors: List[Contribution]


class PredictResponse(BaseModel):
    """Prediction endpoint output."""

    eligibility_code: int
    eligibility_label: str
    max_monthly_emi_predicted: float
    confidence: Optional[float] = None
    formula_monthly_emi_for_requested_loan: float
    explainability: PredictionExplainability


class EmiCalculationResponse(BaseModel):
    """Formula-based EMI output."""

    monthly_emi: float
    total_payable: float
    total_interest: float


class HealthResponse(BaseModel):
    """Simple service health response."""

    status: str
    models_loaded: bool
    message: str
