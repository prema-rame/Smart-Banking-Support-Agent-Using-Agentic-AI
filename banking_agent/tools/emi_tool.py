"""Reducing-balance loan EMI calculator."""

from __future__ import annotations

import math


def calculate_emi(loan_amount: float, annual_interest_rate: float, tenure_months: int) -> dict[str, float]:
    """Calculate the monthly EMI; interest rate is annual and expressed as percent."""
    if not math.isfinite(loan_amount) or loan_amount <= 0:
        raise ValueError("Loan amount must be a positive finite number.")
    if not math.isfinite(annual_interest_rate) or annual_interest_rate < 0 or annual_interest_rate > 100:
        raise ValueError("Annual interest rate must be between 0 and 100 percent.")
    if isinstance(tenure_months, bool) or not isinstance(tenure_months, int) or not 1 <= tenure_months <= 480:
        raise ValueError("Tenure must be between 1 and 480 months.")

    monthly_rate = annual_interest_rate / 12 / 100
    if monthly_rate == 0:
        emi = loan_amount / tenure_months
    else:
        growth = (1 + monthly_rate) ** tenure_months
        emi = loan_amount * monthly_rate * growth / (growth - 1)
    return {"emi": round(emi, 2)}