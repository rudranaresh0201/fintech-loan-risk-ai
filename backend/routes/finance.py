"""Finance endpoints for prediction and EMI calculation."""

from __future__ import annotations

from fastapi import APIRouter

from backend.services.emi_service import calculate_emi
from backend.services.prediction_service import predict_emi_eligibility
from schemas.requests import EmiCalculationRequest, FinancialProfile
from schemas.responses import EmiCalculationResponse, PredictResponse


router = APIRouter(tags=["finance"])


@router.post("/predict", response_model=PredictResponse)
def predict(payload: FinancialProfile) -> PredictResponse:
    """Predict eligibility and maximum affordable EMI from user profile."""

    return predict_emi_eligibility(payload)


@router.post("/calculate-emi", response_model=EmiCalculationResponse)
def calculate(payload: EmiCalculationRequest) -> EmiCalculationResponse:
    """Calculate formula-based EMI from principal, rate, and tenure."""

    result = calculate_emi(
        principal=payload.principal,
        annual_interest_rate=payload.annual_interest_rate,
        tenure_months=payload.tenure_months,
    )
    return EmiCalculationResponse(
        monthly_emi=result.monthly_emi,
        total_payable=result.total_payable,
        total_interest=result.total_interest,
    )
