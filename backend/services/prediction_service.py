"""Prediction orchestration service."""

from __future__ import annotations

import torch
from backend.models.lstm_model import LSTMModel
from backend.services.emi_service import calculate_emi
from schemas.requests import FinancialProfile
from schemas.responses import PredictResponse


#  Load model once
lstm_model = LSTMModel(input_size=4, hidden_size=32)
lstm_model.eval()


#  Sequence builder
def build_sequence_from_profile(profile: FinancialProfile):
    income = profile.monthly_salary

    expenses = (
        profile.monthly_rent
        + profile.current_emi_amount
        + profile.groceries_utilities
        + profile.travel_expenses
        + profile.other_monthly_expenses
    )

    emi_flag = 1 if profile.current_emi_amount > 0 else 0

    sequence = []

    for i in range(6):
        trend = i / 6

        income_variation = income * (1 - 0.05 * trend)
        expense_variation = expenses * (1 + 0.1 * trend)

        #  better normalization
        total = income_variation + expense_variation + 1e-6
        income_norm = income_variation / total
        expense_norm = expense_variation / total

        dti = expense_variation / (income_variation + 1e-6)

        sequence.append([
            income_norm,
            expense_norm,
            dti,
            emi_flag
        ])

    return torch.tensor([sequence], dtype=torch.float32)


#  EMI affordability
def estimate_affordable_emi(profile: FinancialProfile):
    income = profile.monthly_salary

    liabilities = (
        profile.current_emi_amount
        + profile.monthly_rent
        + profile.groceries_utilities
        + profile.travel_expenses
        + profile.other_monthly_expenses
    )

    max_allowed = 0.4 * income

    return max(0, round(max_allowed - liabilities, 2))


#  MAIN FUNCTION
def predict_emi_eligibility(profile: FinancialProfile) -> PredictResponse:

    # Build sequence
    sequence = build_sequence_from_profile(profile)

    # LSTM inference
    with torch.no_grad():
        lstm_output = lstm_model(sequence)

    risk_score = float(lstm_output.item())

    #  clamp for stability
    risk_score = max(0.01, min(risk_score, 0.99))

    print("LSTM Risk Score:", risk_score)

    # EMI calculation FIRST (you used it before defining earlier ❌)
    emi_result = calculate_emi(
        principal=profile.requested_amount,
        annual_interest_rate=profile.annual_interest_rate,
        tenure_months=profile.requested_tenure,
    )

    # EMI affordability
    max_monthly_emi = estimate_affordable_emi(profile)

    #  Better risk label
    if risk_score > 0.6:
        risk_label = "High"
    elif risk_score > 0.4:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    #  FINAL DECISION LOGIC (clean)
    if risk_score > 0.6:
        eligibility_label = "Not Eligible"
        eligibility_code = 1
    elif max_monthly_emi < emi_result.monthly_emi:
        eligibility_label = "Not Eligible"
        eligibility_code = 1
    else:
        eligibility_label = "Eligible"
        eligibility_code = 0

    # Explainability
    insights = []

    if profile.current_emi_amount > 0:
        insights.append("Existing EMI increases financial burden")

    if profile.monthly_salary < (profile.monthly_rent * 3):
        insights.append("Low income-to-rent ratio")

    if profile.credit_score < 650:
        insights.append("Low credit score increases risk")

    if risk_score > 0.5:
        insights.append("High predicted financial stress from LSTM")

    if profile.bank_balance < 20000:
        insights.append("Low bank balance indicates weak financial cushion")

    if profile.dependents > 2:
        insights.append("High number of dependents increases financial pressure")

    #  Decision reasoning
    decision_factors = []

    if risk_score > 0.6:
        decision_factors.append("High risk score")

    if max_monthly_emi < emi_result.monthly_emi:
        decision_factors.append("Requested EMI exceeds affordable limit")

    return PredictResponse(
        eligibility_code=eligibility_code,
        eligibility_label=eligibility_label,
        max_monthly_emi_predicted=round(max_monthly_emi, 2),
        confidence=round(risk_score, 4),
        risk_percentage=round(risk_score * 100, 2),
        formula_monthly_emi_for_requested_loan=emi_result.monthly_emi,
        risk_label=risk_label,
        explainability={
            "rule_based_insights": insights,
            "decision_factors": decision_factors,
            "classification_top_contributors": [],
            "regression_top_contributors": [],
        },
    )