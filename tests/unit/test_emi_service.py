"""Unit tests for EMI calculation logic."""

from __future__ import annotations

from backend.services.emi_service import calculate_emi


def test_calculate_emi_standard_case() -> None:
    """EMI output should be stable and numerically correct for normal inputs."""

    result = calculate_emi(principal=100000, annual_interest_rate=12.0, tenure_months=12)

    assert result.monthly_emi == 8884.88
    assert result.total_payable == 106618.55
    assert result.total_interest == 6618.55


def test_calculate_emi_zero_interest_edge_case() -> None:
    """When interest is zero, EMI should be principal divided by tenure."""

    result = calculate_emi(principal=120000, annual_interest_rate=0.0, tenure_months=12)

    assert result.monthly_emi == 10000.0
    assert result.total_payable == 120000.0
    assert result.total_interest == 0.0


def test_calculate_emi_single_month_edge_case() -> None:
    """One-month tenure should return near principal plus one month interest."""

    result = calculate_emi(principal=10000, annual_interest_rate=12.0, tenure_months=1)

    assert result.monthly_emi == 10100.0
    assert result.total_payable == 10100.0
    assert result.total_interest == 100.0
