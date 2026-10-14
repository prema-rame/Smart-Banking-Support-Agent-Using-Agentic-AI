"""Focused tests for banking calculations, confirmation, and HTTP validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

_TEST_DATABASE = Path(tempfile.gettempdir()) / f"banking-agent-{uuid4().hex}.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DATABASE}"

from fastapi.testclient import TestClient

from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from banking_agent.agent import SYSTEM_PROMPT, ToolRun, _current_run, _toolset
from banking_agent.app import app
from banking_agent.database import SessionLocal, init_db
from banking_agent.models import LoanApplication
from banking_agent.tools.account_tool import get_account_details
from banking_agent.tools.application_tool import submit_loan_application
from banking_agent.tools.eligibility_tool import check_loan_eligibility
from banking_agent.tools.emi_tool import calculate_emi


def _application_count() -> int:
    with SessionLocal() as db:
        return db.query(LoanApplication).count()


def test_emi_and_eligibility_rules() -> None:
    emi = calculate_emi(500000, 10, 36)["emi"]
    assert emi == 16133.59

    init_db()
    approved = check_loan_eligibility("CUST1001", emi)
    declined = check_loan_eligibility("CUST1002", emi)
    assert approved["eligible"] is True
    assert approved["obligation_ratio"] == 0.3522
    assert declined["eligible"] is False
    assert "salary" in str(declined["reason"]).lower()
    assert "credit score" in str(declined["reason"]).lower()


def test_single_agent_executor_assembles_four_tools() -> None:
    tools = _toolset("CUST1001")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    ).partial(
        customer_id="CUST1001",
        submission_state="not granted",
        submission_instructions="No submission is authorized.",
    )
    model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-test-key", temperature=0)
    agent = create_openai_tools_agent(model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=4)

    assert len(executor.tools) == 4
    assert {tool.name for tool in executor.tools} == {
        "calculate_emi",
        "check_loan_eligibility",
        "get_account_details",
        "submit_loan_application",
    }


def test_submission_tool_requires_explicit_grant() -> None:
    init_db()
    tools = {tool.name: tool for tool in _toolset("CUST1001")}
    assert tools["calculate_emi"].args_schema.model_fields["tenure_months"].is_required()
    assert not tools["calculate_emi"].args_schema.model_fields["annual_interest_rate"].is_required()
    assert "customer_id" in tools["get_account_details"].args_schema.model_fields
    before = _application_count()
    offer = {
        "customer_id": "CUST1001",
        "loan_amount": 500000,
        "tenure": 3,
        "emi": 16133.59,
    }

    denied = tools["submit_loan_application"].invoke(offer)
    assert "error" in denied
    assert _application_count() == before

    run = ToolRun(
        customer_id="CUST1001",
        authorized_application={
            "customer_id": "CUST1001",
            "loan_amount": 500000,
            "tenure_years": 3,
            "emi": 16133.59,
        },
    )
    token = _current_run.set(run)
    try:
        submitted = tools["submit_loan_application"].invoke(offer)
    finally:
        _current_run.reset(token)

    assert submitted == {"application_id": "APP1001", "status": "Submitted"}
    assert _application_count() == before + 1


def test_customer_and_direct_application_endpoints() -> None:
    with TestClient(app) as client:
        customer = client.get("/customer/CUST1001")
        assert customer.status_code == 200
        assert customer.json()["credit_score"] == 760

        missing_confirmation = client.post(
            "/loan/apply",
            json={"customer_id": "CUST1001", "loan_amount": 500000, "tenure_years": 3},
        )
        assert missing_confirmation.status_code == 422

        before = _application_count()
        invalid_tenure = client.post(
            "/loan/apply",
            json={
                "customer_id": "CUST1001",
                "loan_amount": 500000,
                "tenure_years": 1.01,
                "confirmation": "CONFIRM",
            },
        )
        assert invalid_tenure.status_code == 422
        ineligible = client.post(
            "/loan/apply",
            json={
                "customer_id": "CUST1002",
                "loan_amount": 500000,
                "tenure_years": 3,
                "confirmation": "CONFIRM",
            },
        )
        assert ineligible.status_code == 409
        assert _application_count() == before

        submitted = client.post(
            "/loan/apply",
            json={
                "customer_id": "CUST1001",
                "loan_amount": 500000,
                "tenure_years": 3,
                "confirmation": "CONFIRM",
            },
        )
        assert submitted.status_code == 201
        assert submitted.json()["application_id"] == f"APP{before + 1001:04d}"


def test_chat_confirmation_is_bound_to_session(monkeypatch) -> None:
    init_db()
    calls: list[dict[str, object]] = []

    def fake_agent(*, customer_id, message, history, authorized_application=None):
        calls.append({"message": message, "authorized_application": authorized_application})
        if authorized_application:
            result = submit_loan_application(
                customer_id,
                authorized_application["loan_amount"],
                authorized_application["tenure_years"],
                authorized_application["emi"],
            )
            return "submitted", {"submit_loan_application": {"output": result}}

        emi_input = {"loan_amount": 500000, "annual_interest_rate": 10.0, "tenure_months": 36}
        emi = calculate_emi(**emi_input)
        eligibility = check_loan_eligibility(customer_id, emi["emi"])
        account = get_account_details(customer_id)
        return "assessment complete", {
            "calculate_emi": {"input": emi_input, "output": emi},
            "check_loan_eligibility": {
                "input": {"customer_id": customer_id, "new_emi": emi["emi"]},
                "output": eligibility,
            },
            "get_account_details": {"input": {"customer_id": customer_id}, "output": account},
        }

    monkeypatch.setattr("banking_agent.api.routes.run_agent", fake_agent)
    session_id = str(uuid4())
    with TestClient(app) as client:
        first = client.post(
            "/chat",
            json={
                "customer_id": "CUST1001",
                "session_id": session_id,
                "message": "Assess a loan of 500000 for 3 years.",
            },
        )
        assert first.status_code == 200
        assert first.json()["pending_confirmation"] is True
        assert "Do you want me to submit" in first.json()["response"]

        confirmation = client.post(
            "/chat",
            json={"customer_id": "CUST1001", "session_id": session_id, "message": "Yes, submit."},
        )
        assert confirmation.status_code == 200
        assert confirmation.json()["pending_confirmation"] is False
        assert "Application ID: APP" in confirmation.json()["response"]
        assert calls[1]["authorized_application"] is not None

        unbound = client.post(
            "/chat",
            json={"customer_id": "CUST1001", "session_id": str(uuid4()), "message": "Yes"},
        )
        assert unbound.status_code == 200
        assert calls[2]["authorized_application"] is None