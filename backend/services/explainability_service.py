"""Explainability helpers for financial prediction outputs."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from schemas.requests import FinancialProfile
from schemas.responses import Contribution, PredictionExplainability


def build_rule_based_insights(profile: FinancialProfile) -> List[str]:
    """Generate human-readable affordability and risk signals."""

    total_expenses = (
        profile.monthly_rent
        + profile.school_fees
        + profile.college_fees
        + profile.travel_expenses
        + profile.groceries_utilities
        + profile.other_monthly_expenses
    )
    disposable_income = profile.monthly_salary - total_expenses
    dti = profile.current_emi_amount / profile.monthly_salary

    insights: list[str] = []

    if profile.credit_score < 650:
        insights.append("Credit score is below 650, which increases lending risk.")
    elif profile.credit_score >= 750:
        insights.append("Credit score is strong, which supports EMI eligibility.")

    if dti > 0.45:
        insights.append(
            "Current debt-to-income ratio is above 45%, reducing repayment headroom."
        )
    else:
        insights.append("Debt-to-income ratio is within a safer lending range.")

    if disposable_income <= 0:
        insights.append("Disposable income is non-positive after monthly expenses.")
    else:
        insights.append(
            f"Estimated disposable income is {disposable_income:.2f} per month."
        )

    if profile.emergency_fund < profile.monthly_salary * 3:
        insights.append(
            "Emergency fund is below 3 months of salary, lowering financial resilience."
        )

    return insights


def top_feature_contributions(
    model: Any,
    scaled_row: np.ndarray,
    feature_columns: list[str],
    top_n: int = 5,
) -> List[Contribution]:
    """Estimate top feature effects from linear-model coefficients."""

    if not hasattr(model, "coef_"):
        return []

    coefficients = np.ravel(model.coef_)
    values = np.ravel(scaled_row)
    contribution_values = coefficients * values

    ranked_indices = np.argsort(np.abs(contribution_values))[::-1][:top_n]

    return [
        Contribution(
            feature=feature_columns[index],
            contribution=float(round(contribution_values[index], 4)),
        )
        for index in ranked_indices
    ]


def build_prediction_explainability(
    profile: FinancialProfile,
    classifier: Any,
    regressor: Any,
    scaled_row: np.ndarray,
    feature_columns: list[str],
) -> PredictionExplainability:
    """Build a single explainability object for API response."""

    return PredictionExplainability(
        rule_based_insights=build_rule_based_insights(profile),
        classification_top_contributors=top_feature_contributions(
            model=classifier,
            scaled_row=scaled_row,
            feature_columns=feature_columns,
        ),
        regression_top_contributors=top_feature_contributions(
            model=regressor,
            scaled_row=scaled_row,
            feature_columns=feature_columns,
        ),
    )
