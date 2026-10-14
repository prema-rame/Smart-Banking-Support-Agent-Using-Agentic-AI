"""Persist a submitted loan application."""

from __future__ import annotations

import math

from banking_agent.database import SessionLocal
from banking_agent.models import Customer, LoanApplication


def submit_loan_application(
    customer_id: str,
    loan_amount: float,
    tenure: float,
    emi: float,
) -> dict[str, str]:
    """Insert an application and return its human-readable identifier."""
    if not math.isfinite(loan_amount) or not 0 < loan_amount <= 50_000_000:
        raise ValueError("Loan amount must be between ₹1 and ₹5 crore.")
    if not math.isfinite(tenure) or not 0 < tenure <= 40:
        raise ValueError("Tenure must be greater than 0 and no more than 40 years.")
    if not math.isfinite(emi) or emi <= 0:
        raise ValueError("EMI must be a positive finite number.")

    with SessionLocal.begin() as db:
        if db.get(Customer, customer_id) is None:
            raise ValueError("Customer was not found.")
        application = LoanApplication(
            customer_id=customer_id,
            loan_amount=round(loan_amount, 2),
            tenure_years=round(tenure, 4),
            emi=round(emi, 2),
            status="Submitted",
        )
        db.add(application)
        db.flush()
        application_id = f"APP{application.application_id + 1000:04d}"

    return {"application_id": application_id, "status": "Submitted"}