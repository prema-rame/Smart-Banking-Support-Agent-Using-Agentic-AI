"""SQLAlchemy models for customers, applications, and resumable chat sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from banking_agent.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    account_number: Mapped[str] = mapped_column(String(34), unique=True, nullable=False)
    salary: Mapped[float] = mapped_column(Float, nullable=False)
    existing_emi: Mapped[float] = mapped_column(Float, nullable=False)
    credit_score: Mapped[int] = mapped_column(Integer, nullable=False)


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    application_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    loan_amount: Mapped[float] = mapped_column(Float, nullable=False)
    tenure_years: Mapped[float] = mapped_column(Float, nullable=False)
    emi: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="Submitted")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    chat_history: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False, default=list)
    pending_application: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)