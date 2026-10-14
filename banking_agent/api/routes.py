"""HTTP API for the banking support agent and mock application workflow."""

from __future__ import annotations

import math
import re
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from banking_agent.agent import run_agent
from banking_agent.database import SessionLocal
from banking_agent.models import ChatSession, Customer
from banking_agent.tools.application_tool import submit_loan_application
from banking_agent.tools.eligibility_tool import check_loan_eligibility
from banking_agent.tools.emi_tool import calculate_emi

router = APIRouter()


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=4000)
    session_id: UUID = Field(default_factory=uuid4)


class ChatResponse(BaseModel):
    session_id: UUID
    customer_id: str
    response: str
    pending_confirmation: bool = False


class CustomerResponse(BaseModel):
    customer_id: str
    name: str
    account_number: str
    salary: float
    existing_emi: float
    credit_score: int


class LoanApplicationRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=32)
    loan_amount: float = Field(gt=0, le=50_000_000)
    tenure_years: float = Field(gt=0, le=40)
    confirmation: Literal["CONFIRM"]


class LoanApplicationResponse(BaseModel):
    application_id: str
    status: str
    emi: float


CONFIRMATIONS = {"yes", "yes submit", "yes, submit", "confirm", "confirm and submit", "proceed", "proceed with application", "submit application", "submit the application", "yes, submit the application"}
CANCELLATIONS = {"no", "no thanks", "cancel", "cancel application", "do not submit", "don't submit"}


def _is_confirmation(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.strip().lower()).rstrip(".! ")
    return normalized in CONFIRMATIONS


def _is_cancellation(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.strip().lower()).rstrip(".! ")
    return normalized in CANCELLATIONS


def _summary(
    *,
    loan_amount: float,
    tenure_months: int,
    emi: float,
    eligibility: dict[str, Any],
    account: dict[str, Any] | None,
    application_status: str,
) -> str:
    eligible = bool(eligibility.get("eligible"))
    tenure_years = tenure_months / 12
    tenure_label = f"{tenure_years:g} years" if tenure_months % 12 == 0 else f"{tenure_months} months"
    lines = [
        "Loan Summary",
        "=======================",
        f"Loan Amount: ₹{loan_amount:,.2f}",
        f"Tenure: {tenure_label}",
        f"Monthly EMI: ₹{emi:,.0f}",
        "",
        f"Eligibility: {'Approved' if eligible else 'Not approved'}",
        f"Reason: {eligibility.get('reason', 'Eligibility could not be determined.')}",
    ]
    if account:
        lines.extend(
            [
                "",
                f"Customer: {account['name']}",
                f"Account Number: {account['account_number']}",
                f"Credit Score: {account['credit_score']}",
            ]
        )
    lines.extend(["", f"Application Status: {application_status}"])
    return "\n".join(lines)


def _tool_output(trace: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    result = trace.get(name, {}).get("output")
    return result if isinstance(result, dict) and "error" not in result else None


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run a customer turn through the single agent and gate application writes."""
    session_key = str(request.session_id)
    with SessionLocal() as db:
        customer = db.get(Customer, request.customer_id)
        if customer is None:
            raise HTTPException(status_code=404, detail="Customer not found.")
        chat_session = db.get(ChatSession, session_key)
        if chat_session is not None and chat_session.customer_id != request.customer_id:
            raise HTTPException(status_code=403, detail="This chat session belongs to another customer.")
        if chat_session is None:
            chat_session = ChatSession(session_id=session_key, customer_id=request.customer_id)
            db.add(chat_session)
            db.commit()
            history: list[dict[str, str]] = []
            pending: dict[str, Any] | None = None
        else:
            history = list(chat_session.chat_history or [])
            pending = chat_session.pending_application

    confirmation = pending is not None and _is_confirmation(request.message)
    cancellation = pending is not None and _is_cancellation(request.message)
    trace: dict[str, dict[str, Any]] = {}

    if cancellation:
        answer = "The pending loan application was cancelled. No application was submitted."
        pending = None
    else:
        try:
            answer, trace = run_agent(
                customer_id=request.customer_id,
                message=request.message,
                history=history,
                authorized_application=pending if confirmation else None,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="The banking agent could not complete this request.") from exc

        if confirmation:
            submitted = _tool_output(trace, "submit_loan_application")
            if submitted and submitted.get("status") == "Submitted":
                answer = _summary(
                    loan_amount=pending["loan_amount"],
                    tenure_months=pending["tenure_months"],
                    emi=pending["emi"],
                    eligibility=pending["eligibility"],
                    account=pending["account"],
                    application_status=f"Submitted ({submitted['application_id']})",
                )
                answer += f"\n\nApplication ID: {submitted['application_id']}"
                pending = None
            else:
                answer = "I could not submit the application. The eligible offer remains pending; no application was created. Please try confirming again."
        else:
            emi_run = trace.get("calculate_emi")
            eligibility = _tool_output(trace, "check_loan_eligibility")
            if emi_run and eligibility:
                account = _tool_output(trace, "get_account_details") if eligibility.get("eligible") else None
                if not eligibility.get("eligible") or account:
                    inputs = emi_run.get("input", {})
                    loan_amount = float(inputs["loan_amount"])
                    tenure_months = int(inputs["tenure_months"])
                    emi_result = _tool_output(trace, "calculate_emi")
                    if emi_result:
                        emi = float(emi_result["emi"])
                        if eligibility.get("eligible"):
                            pending = {
                                "customer_id": request.customer_id,
                                "loan_amount": loan_amount,
                                "tenure_months": tenure_months,
                                "tenure_years": tenure_months / 12,
                                "emi": emi,
                                "eligibility": eligibility,
                                "account": account,
                            }
                            answer = _summary(
                                loan_amount=loan_amount,
                                tenure_months=tenure_months,
                                emi=emi,
                                eligibility=eligibility,
                                account=account,
                                application_status="Awaiting your confirmation",
                            )
                            answer += "\n\nDo you want me to submit the loan application?"
                        else:
                            pending = None
                            answer = _summary(
                                loan_amount=loan_amount,
                                tenure_months=tenure_months,
                                emi=emi,
                                eligibility=eligibility,
                                account=None,
                                application_status="Not submitted (not eligible)",
                            )

    history.extend([{"role": "user", "content": request.message}, {"role": "assistant", "content": answer}])
    history = history[-20:]
    with SessionLocal.begin() as db:
        chat_session = db.get(ChatSession, session_key)
        if chat_session is None:
            raise HTTPException(status_code=404, detail="Chat session was not found.")
        chat_session.chat_history = history
        chat_session.pending_application = pending

    return ChatResponse(
        session_id=request.session_id,
        customer_id=request.customer_id,
        response=answer,
        pending_confirmation=pending is not None,
    )


@router.get("/customer/{customer_id}", response_model=CustomerResponse)
def get_customer(customer_id: str) -> CustomerResponse:
    """Return a customer's mock profile."""
    with SessionLocal() as db:
        customer = db.get(Customer, customer_id)
        if customer is None:
            raise HTTPException(status_code=404, detail="Customer not found.")
        return CustomerResponse(
            customer_id=customer.customer_id,
            name=customer.name,
            account_number=customer.account_number,
            salary=customer.salary,
            existing_emi=customer.existing_emi,
            credit_score=customer.credit_score,
        )


@router.post("/loan/apply", response_model=LoanApplicationResponse, status_code=status.HTTP_201_CREATED)
def apply_for_loan(request: LoanApplicationRequest) -> LoanApplicationResponse:
    """Submit a directly requested mock application after explicit confirmation."""
    with SessionLocal() as db:
        customer = db.get(Customer, request.customer_id)
        if customer is None:
            raise HTTPException(status_code=404, detail="Customer not found.")

    tenure_months_exact = request.tenure_years * 12
    if not math.isclose(tenure_months_exact, round(tenure_months_exact), abs_tol=1e-9):
        raise HTTPException(status_code=422, detail="Tenure must resolve to whole months.")
    tenure_months = round(tenure_months_exact)
    try:
        emi = calculate_emi(request.loan_amount, 10.0, tenure_months)["emi"]
        eligibility = check_loan_eligibility(request.customer_id, emi)
        if not eligibility["eligible"]:
            raise HTTPException(status_code=409, detail={"eligible": False, "reason": eligibility["reason"]})
        result = submit_loan_application(request.customer_id, request.loan_amount, request.tenure_years, emi)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (TypeError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail="Invalid loan application values.") from exc
    return LoanApplicationResponse(application_id=result["application_id"], status=result["status"], emi=emi)