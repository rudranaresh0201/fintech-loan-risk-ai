"""Default-risk inference service powered by an LSTM model."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from fastapi import HTTPException

from backend.models.model_loader import load_default_risk_artifacts
from schemas.requests import RiskSequenceRequest
from schemas.responses import RiskPredictResponse


def _risk_level(score: float) -> str:
    """Map risk score to LOW / MEDIUM / HIGH levels."""

    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def _slope(values: np.ndarray) -> float:
    """Return linear trend slope for a monthly series."""

    x_axis = np.arange(values.shape[0], dtype=np.float32)
    return float(np.polyfit(x_axis, values, deg=1)[0])


def _build_explanations(
    income: np.ndarray,
    expenses: np.ndarray,
    emi_paid: np.ndarray,
    score: float,
) -> List[str]:
    """Build simple, user-facing heuristic explanations."""

    explanations: List[str] = []

    income_slope = _slope(income)
    expense_slope = _slope(expenses)

    if expense_slope > income_slope:
        explanations.append("Expenses are growing faster than income trend, increasing default risk.")

    latest_income = float(income[-1])
    latest_expenses = float(expenses[-1])
    if latest_expenses > latest_income:
        explanations.append("Latest month expenses exceed income, indicating potential repayment stress.")

    emi_burden = float(np.mean(emi_paid / np.maximum(income, 1.0)))
    if emi_burden > 0.4:
        explanations.append("Average EMI burden is high relative to income.")

    if not explanations:
        if score >= 0.75:
            explanations.append("Pattern indicates elevated repayment stress despite stable recent ratios.")
        elif score >= 0.45:
            explanations.append("Financial pattern shows moderate risk with manageable repayment pressure.")
        else:
            explanations.append("Income and expense trends are stable with low repayment pressure.")

    return explanations


def predict_default_risk(payload: RiskSequenceRequest) -> RiskPredictResponse:
    """Run LSTM inference and return default-risk score with explanations."""

    artifacts = load_default_risk_artifacts()
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Default risk model artifacts are not available. Train and deploy the LSTM model first.",
        )

    records = payload.sequence
    if len(records) < artifacts.sequence_length:
        raise HTTPException(
            status_code=422,
            detail=f"At least {artifacts.sequence_length} monthly records are required.",
        )

    ordered_records = records[-artifacts.sequence_length :]

    sequence = np.asarray(
        [
            [
                float(row.income),
                float(row.expenses),
                float(row.emi_paid),
            ]
            for row in ordered_records
        ],
        dtype=np.float32,
    )

    normalized = (sequence - artifacts.mean) / np.where(artifacts.std == 0, 1.0, artifacts.std)
    tensor_input = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = float(artifacts.model(tensor_input).item())

    score = max(0.0, min(1.0, score))
    risk_level = _risk_level(score)

    income = sequence[:, 0]
    expenses = sequence[:, 1]
    emi_paid = sequence[:, 2]
    explanations = _build_explanations(income=income, expenses=expenses, emi_paid=emi_paid, score=score)

    return RiskPredictResponse(
        risk_score=round(score, 4),
        risk_level=risk_level,
        explanations=explanations,
    )
