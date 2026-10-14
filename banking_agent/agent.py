"""Single LangChain agent and request-scoped safety-gated banking tools."""

from __future__ import annotations

import math
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from banking_agent.tools.account_tool import get_account_details
from banking_agent.tools.application_tool import submit_loan_application
from banking_agent.tools.eligibility_tool import check_loan_eligibility
from banking_agent.tools.emi_tool import calculate_emi

load_dotenv()


@dataclass
class ToolRun:
    customer_id: str
    authorized_application: dict[str, Any] | None = None
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)


_current_run: ContextVar[ToolRun | None] = ContextVar("banking_agent_run", default=None)

SYSTEM_PROMPT = """You are a careful banking support assistant. Think through the workflow step by step internally; do not reveal private chain-of-thought. Use tool results as the source of truth, and ask concise questions for missing details.

The customer ID for this request is {customer_id}. Never use another customer ID.
For a loan assessment, first collect the requested loan amount and tenure. If either is missing, ask for it before using tools. Then call calculate_emi using an annual interest rate of 10% and tenure in months. Next call check_loan_eligibility for this customer using the calculated EMI. If and only if eligible, call get_account_details. Report the EMI, eligibility decision and reason, and account details. Ask exactly: "Do you want me to submit the loan application?" Do not claim an application was submitted before the submission tool confirms it.

The submission authorization for this invocation is {submission_state}. {submission_instructions}
When authorization is absent, never call submit_loan_application. When it is present, submit only the exact customer and offer described in the authorization context, and only because the customer explicitly confirmed. Do not infer confirmation from prior intent, silence, or a request merely to prepare the application.
"""


def _record(name: str, inputs: dict[str, Any], operation: Any) -> dict[str, Any]:
    try:
        result = operation()
    except (ValueError, TypeError, ArithmeticError) as exc:
        result = {"error": str(exc)}
    run = _current_run.get()
    if run is not None:
        run.outputs[name] = {"input": inputs, "output": result}
    return result


def _toolset(customer_id: str) -> list[StructuredTool]:
    session_customer_id = customer_id

    def emi_tool(loan_amount: float, tenure_months: int, annual_interest_rate: float = 10.0) -> dict[str, Any]:
        inputs = {
            "loan_amount": loan_amount,
            "annual_interest_rate": annual_interest_rate,
            "tenure_months": tenure_months,
        }
        return _record("calculate_emi", inputs, lambda: calculate_emi(**inputs))

    def eligibility_tool(customer_id: str, new_emi: float) -> dict[str, Any]:
        inputs = {"customer_id": customer_id, "new_emi": new_emi}
        if customer_id != session_customer_id:
            return _record("check_loan_eligibility", inputs, lambda: {"error": "Customer ID does not match this session."})
        run = _current_run.get()
        calculated = run.outputs.get("calculate_emi", {}).get("output") if run else None
        if not isinstance(calculated, dict) or "emi" not in calculated or not math.isclose(
            new_emi, calculated["emi"], rel_tol=0, abs_tol=0.01
        ):
            return _record("check_loan_eligibility", inputs, lambda: {"error": "Calculate the EMI first and use that exact value for eligibility."})
        return _record("check_loan_eligibility", inputs, lambda: check_loan_eligibility(session_customer_id, new_emi))

    def account_tool(customer_id: str) -> dict[str, Any]:
        inputs = {"customer_id": customer_id}
        run = _current_run.get()
        eligibility = run.outputs.get("check_loan_eligibility", {}).get("output") if run else None
        if customer_id != session_customer_id:
            result = {"error": "Customer ID does not match this session."}
        elif not isinstance(eligibility, dict) or not eligibility.get("eligible"):
            result = {"error": "Account details are available for this workflow only after an eligible loan check."}
        else:
            result = get_account_details(session_customer_id)
        if run is not None:
            run.outputs["get_account_details"] = {"input": inputs, "output": result}
        return result

    def application_tool(
        customer_id: str,
        loan_amount: float,
        tenure: float,
        emi: float,
    ) -> dict[str, Any]:
        inputs = {
            "customer_id": customer_id,
            "loan_amount": loan_amount,
            "tenure": tenure,
            "emi": emi,
        }

        def submit() -> dict[str, Any]:
            run = _current_run.get()
            approved = run.authorized_application if run else None
            if approved is None:
                return {"error": "Explicit customer confirmation is required before submission."}
            exact_offer = (
                customer_id == session_customer_id == approved["customer_id"]
                and math.isclose(loan_amount, approved["loan_amount"], rel_tol=0, abs_tol=0.01)
                and math.isclose(tenure, approved["tenure_years"], rel_tol=0, abs_tol=0.0001)
                and math.isclose(emi, approved["emi"], rel_tol=0, abs_tol=0.01)
            )
            if not exact_offer:
                return {"error": "Submission terms do not match the confirmed eligible offer."}
            return submit_loan_application(session_customer_id, loan_amount, tenure, emi)

        return _record("submit_loan_application", inputs, submit)

    return [
        StructuredTool.from_function(
            func=emi_tool,
            name="calculate_emi",
            description="Calculate a loan's monthly EMI. Provide loan amount in rupees, annual interest rate in percent, and tenure in months.",
        ),
        StructuredTool.from_function(
            func=eligibility_tool,
            name="check_loan_eligibility",
            description="Check eligibility using the customer's stored salary, existing EMI, and credit score, plus the proposed new EMI.",
        ),
        StructuredTool.from_function(
            func=account_tool,
            name="get_account_details",
            description="Retrieve the current customer's name, account number, and credit score after a passing eligibility check.",
        ),
        StructuredTool.from_function(
            func=application_tool,
            name="submit_loan_application",
            description="Submit the exact eligible loan offer only after explicit customer confirmation. This tool rejects unauthorized submissions.",
        ),
    ]


def _chat_messages(history: list[dict[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history[-20:]:
        role, content = item.get("role"), item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def run_agent(
    *,
    customer_id: str,
    message: str,
    history: list[dict[str, str]],
    authorized_application: dict[str, Any] | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Invoke the one agent and return its answer and this invocation's tool results."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    run = ToolRun(customer_id=customer_id, authorized_application=authorized_application)
    state = "granted" if authorized_application else "not granted"
    instructions = (
        f"The customer explicitly confirmed this pending offer: {authorized_application}."
        if authorized_application
        else "No application submission is authorized in this invocation."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    tools = _toolset(customer_id)
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
        timeout=30,
        max_retries=2,
    )
    agent = create_openai_tools_agent(llm, tools, prompt.partial(
        customer_id=customer_id,
        submission_state=state,
        submission_instructions=instructions,
    ))
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=8, handle_parsing_errors=True, verbose=False)

    token = _current_run.set(run)
    try:
        result = executor.invoke({"input": message, "chat_history": _chat_messages(history)})
    finally:
        _current_run.reset(token)
    return str(result["output"]), run.outputs