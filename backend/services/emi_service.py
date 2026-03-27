"""Reusable EMI formula and affordability helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmiResult:
    """Calculated EMI summary."""

    monthly_emi: float
    total_payable: float
    total_interest: float


def calculate_emi(principal: float, annual_interest_rate: float, tenure_months: int) -> EmiResult:
    """Compute monthly EMI using the standard reducing-balance formula."""

    monthly_rate = annual_interest_rate / (12 * 100)

    if monthly_rate == 0:
        monthly_emi = principal / tenure_months
    else:
        growth = (1 + monthly_rate) ** tenure_months
        monthly_emi = principal * monthly_rate * growth / (growth - 1)

    total_payable = monthly_emi * tenure_months
    total_interest = total_payable - principal

    return EmiResult(
        monthly_emi=round(monthly_emi, 2),
        total_payable=round(total_payable, 2),
        total_interest=round(total_interest, 2),
    )
