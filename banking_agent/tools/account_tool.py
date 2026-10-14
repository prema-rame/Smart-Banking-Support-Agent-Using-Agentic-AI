"""Read customer account information from SQLite."""

from __future__ import annotations

from banking_agent.database import SessionLocal
from banking_agent.models import Customer


def get_account_details(customer_id: str) -> dict[str, object]:
    """Return the account information required for a loan summary."""
    with SessionLocal() as db:
        customer = db.get(Customer, customer_id)
        if customer is None:
            raise ValueError("Customer was not found.")
        return {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "account_number": customer.account_number,
            "credit_score": customer.credit_score,
        }