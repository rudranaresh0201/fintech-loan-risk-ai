"""Shared pytest fixtures for EMI Predict AI tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI TestClient for API route tests."""

    return TestClient(app)


@pytest.fixture
def valid_financial_profile_payload() -> dict:
    """Return a valid profile payload accepted by FinancialProfile schema."""

    return {
        "monthly_salary": 80000,
        "years_of_employment": 5,
        "monthly_rent": 18000,
        "family_size": 4,
        "dependents": 2,
        "school_fees": 2500,
        "college_fees": 0,
        "travel_expenses": 4500,
        "groceries_utilities": 9000,
        "other_monthly_expenses": 3500,
        "current_emi_amount": 9000,
        "credit_score": 760,
        "bank_balance": 180000,
        "emergency_fund": 220000,
        "requested_amount": 600000,
        "requested_tenure": 36,
        "annual_interest_rate": 11.5,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Post Graduate",
        "employment_type": "Private",
        "company_type": "MNC",
        "house_type": "Rented",
        "existing_loans": "Yes",
        "emi_scenario": "Vehicle EMI",
    }


@pytest.fixture
def valid_emi_payload() -> dict:
    """Return a valid payload for /calculate-emi endpoint."""

    return {
        "principal": 500000,
        "annual_interest_rate": 10.0,
        "tenure_months": 60,
    }
