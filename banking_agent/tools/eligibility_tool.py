"""Eligibility checks against the customer's stored financial profile."""

from __future__ import annotations

import math

from banking_agent.database import SessionLocal
from banking_agent.models import Customer


def check_loan_eligibility(customer_id: str, new_emi: float) -> dict[str, object]:
    """Evaluate salary, credit score, and total monthly obligations."""
    if not math.isfinite(new_emi) or new_emi <= 0:
        raise ValueError("The proposed EMI must be a positive finite number.")

    with SessionLocal() as db:
        customer = db.get(Customer, customer_id)
        if customer is None:
            raise ValueError("Customer was not found.")

        obligation_ratio = (customer.existing_emi + new_emi) / customer.salary
        reasons = []
        if customer.salary < 30000:
            reasons.append("Monthly salary is below ₹30,000.")
        if customer.credit_score < 700:
            reasons.append("Credit score is below 700.")
        if obligation_ratio > 0.5:
            reasons.append("Monthly obligations would exceed 50% of salary.")

        return {
            "eligible": not reasons,
            "reason": " ".join(reasons) if reasons else "Meets salary, credit score, and monthly-obligation criteria.",
            "obligation_ratio": round(obligation_ratio, 4),
        }