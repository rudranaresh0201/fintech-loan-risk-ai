"""Request schemas for prediction and EMI APIs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FinancialProfile(BaseModel):
    """Customer financial input used by both formula and ML endpoints."""

    monthly_salary: float = Field(..., gt=0, le=5_000_000)
    years_of_employment: float = Field(..., ge=0, le=60)
    monthly_rent: float = Field(..., ge=0, le=2_000_000)
    family_size: int = Field(..., ge=1, le=20)
    dependents: int = Field(..., ge=0, le=20)
    school_fees: float = Field(..., ge=0, le=1_000_000)
    college_fees: float = Field(..., ge=0, le=1_000_000)
    travel_expenses: float = Field(..., ge=0, le=500_000)
    groceries_utilities: float = Field(..., ge=0, le=500_000)
    other_monthly_expenses: float = Field(..., ge=0, le=500_000)
    current_emi_amount: float = Field(..., ge=0, le=2_000_000)
    credit_score: int = Field(..., ge=300, le=900)
    bank_balance: float = Field(..., ge=0, le=100_000_000)
    emergency_fund: float = Field(..., ge=0, le=100_000_000)
    requested_amount: float = Field(..., gt=0, le=100_000_000)
    requested_tenure: int = Field(..., ge=1, le=600)
    annual_interest_rate: float = Field(12.0, gt=0, le=60)

    gender: Literal["Male", "Female"]
    marital_status: Literal["Single", "Married"]
    education: Literal["High School", "Post Graduate", "Professional"]
    employment_type: Literal["Private", "Self-employed"]
    company_type: Literal["MNC", "Mid-size", "Small", "Startup"]
    house_type: Literal["Own", "Rented"]
    existing_loans: Literal["Yes", "No"]
    emi_scenario: Literal[
        "Education EMI",
        "Home Appliances EMI",
        "Personal Loan EMI",
        "Vehicle EMI",
    ]

    @model_validator(mode="after")
    def validate_financial_consistency(self) -> "FinancialProfile":
        """Apply cross-field validation for realistic financial data."""

        if self.dependents > self.family_size:
            raise ValueError("Dependents cannot exceed family size.")

        if self.current_emi_amount > self.monthly_salary:
            raise ValueError("Current EMI cannot be greater than monthly salary.")

        base_expenses = (
            self.monthly_rent
            + self.school_fees
            + self.college_fees
            + self.travel_expenses
            + self.groceries_utilities
            + self.other_monthly_expenses
            + self.current_emi_amount
        )

        if base_expenses > self.monthly_salary * 1.5:
            raise ValueError(
                "Total recurring expenses look unrealistic versus monthly salary."
            )

        return self


class EmiCalculationRequest(BaseModel):
    """Request schema for direct EMI formula calculation."""

    principal: float = Field(..., gt=0, le=100_000_000)
    annual_interest_rate: float = Field(..., gt=0, le=60)
    tenure_months: int = Field(..., ge=1, le=600)
