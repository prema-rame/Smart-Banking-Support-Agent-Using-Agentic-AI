"""SQLite configuration, session management, and idempotent demo seeding."""

from __future__ import annotations

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./banking_agent.db")
_engine_options = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_options)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables and add the fictional demo customers if absent."""
    from banking_agent.models import Customer

    Base.metadata.create_all(bind=engine)
    with SessionLocal.begin() as db:
        if db.get(Customer, "CUST1001") is None:
            db.add(
                Customer(
                    customer_id="CUST1001",
                    name="Aarav Sharma",
                    account_number="SBIN0000123456",
                    salary=60000,
                    existing_emi=5000,
                    credit_score=760,
                )
            )
        if db.get(Customer, "CUST1002") is None:
            db.add(
                Customer(
                    customer_id="CUST1002",
                    name="Meera Iyer",
                    account_number="SBIN0000654321",
                    salary=28000,
                    existing_emi=4000,
                    credit_score=680,
                )
            )