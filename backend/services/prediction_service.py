"""Prediction orchestration service."""

from __future__ import annotations

from typing import Optional

from backend.models.model_loader import load_artifacts
from backend.services.emi_service import calculate_emi
from ml.preprocessing import build_model_input
from schemas.requests import FinancialProfile
from schemas.responses import PredictResponse


def _get_confidence(classifier: object, scaled_input: object) -> Optional[float]:
    """Return class confidence when model supports probability estimates."""
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(scaled_input)[0]
        return float(round(max(probabilities), 4))
    return None


def predict_emi_eligibility(profile: FinancialProfile) -> PredictResponse:
    """Run end-to-end prediction: preprocess, classify, regress."""

    artifacts = load_artifacts()

    # Build input
    model_input = build_model_input(
        payload=profile,
        feature_columns=artifacts.feature_columns,
    )

    # Scale input
    scaled_input = artifacts.scaler.transform(model_input)

    # Classification
    prediction = artifacts.classifier.predict(scaled_input)[0]

    if str(prediction).lower() == "eligible":
        eligibility_code = 0
        eligibility_label = "Eligible"
    else:
        eligibility_code = 1
        eligibility_label = "Not Eligible"

    # Regression
    max_monthly_emi = max(
        0.0, float(artifacts.regressor.predict(scaled_input)[0])
    )

    # EMI formula calculation
    emi_result = calculate_emi(
        principal=profile.requested_amount,
        annual_interest_rate=profile.annual_interest_rate,
        tenure_months=profile.requested_tenure,
    )

    # Return response (with safe explainability placeholder)
    return PredictResponse(
        eligibility_code=eligibility_code,
        eligibility_label=eligibility_label,
        max_monthly_emi_predicted=round(max_monthly_emi, 2),
        confidence=_get_confidence(artifacts.classifier, scaled_input),
        formula_monthly_emi_for_requested_loan=emi_result.monthly_emi,
        explainability={
            "rule_based_insights": [],
            "classification_top_contributors": [],
            "regression_top_contributors": [],
        },
    )