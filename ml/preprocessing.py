"""Feature engineering and preprocessing for EMI inference."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from schemas.requests import FinancialProfile


def _build_base_features(payload: FinancialProfile) -> pd.DataFrame:
    """Build the raw numeric feature frame from request payload."""

    return pd.DataFrame(
        [
            {
                "monthly_salary": payload.monthly_salary,
                "years_of_employment": payload.years_of_employment,
                "monthly_rent": payload.monthly_rent,
                "family_size": payload.family_size,
                "dependents": payload.dependents,
                "school_fees": payload.school_fees,
                "college_fees": payload.college_fees,
                "travel_expenses": payload.travel_expenses,
                "groceries_utilities": payload.groceries_utilities,
                "other_monthly_expenses": payload.other_monthly_expenses,
                "current_emi_amount": payload.current_emi_amount,
                "credit_score": payload.credit_score,
                "bank_balance": payload.bank_balance,
                "emergency_fund": payload.emergency_fund,
                "requested_amount": payload.requested_amount,
                "requested_tenure": payload.requested_tenure,
            }
        ]
    )


def _add_derived_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features used during training."""

    salary_safe = df_input["monthly_salary"].replace(0, 1e-9)

    df_input["total_expenses"] = (
        df_input["monthly_rent"]
        + df_input["school_fees"]
        + df_input["college_fees"]
        + df_input["travel_expenses"]
        + df_input["groceries_utilities"]
        + df_input["other_monthly_expenses"]
    )
    df_input["disposable_income"] = df_input["monthly_salary"] - df_input["total_expenses"]
    df_input["debt_to_income"] = df_input["current_emi_amount"] / salary_safe
    df_input["expense_to_income"] = df_input["total_expenses"] / salary_safe
    df_input["financial_buffer"] = df_input["bank_balance"] + df_input["emergency_fund"]

    return df_input


def _build_dummy_columns(payload: FinancialProfile) -> Dict[str, int]:
    """Build one-hot like columns from categorical request fields."""

    return {
        f"gender_{payload.gender}": 1,
        f"marital_status_{payload.marital_status}": 1,
        f"education_{payload.education}": 1,
        f"employment_type_{payload.employment_type}": 1,
        f"company_type_{payload.company_type}": 1,
        f"house_type_{payload.house_type}": 1,
        f"existing_loans_{payload.existing_loans}": 1,
        f"emi_scenario_{payload.emi_scenario}": 1,
    }


def build_model_input(payload: FinancialProfile, feature_columns: list[str]) -> pd.DataFrame:
    """Create model-ready frame with strict training-column alignment."""

    df_input = _build_base_features(payload)
    df_input = _add_derived_features(df_input)

    for column_name, value in _build_dummy_columns(payload).items():
        df_input[column_name] = value

    for column in feature_columns:
        if column not in df_input.columns:
            df_input[column] = 0

    aligned_input = df_input.reindex(columns=feature_columns, fill_value=0)
    return aligned_input
