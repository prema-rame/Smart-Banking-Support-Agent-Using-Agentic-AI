# Smart Banking Support Agent

A single LangChain agent coordinates four ordinary Python tools for loan EMI calculation, eligibility checks, account lookup, and application submission. FastAPI exposes the workflow, and SQLite stores fictional sample customers, chat state, and submitted applications.

## Requirements

- Python 3.11 or newer
- An OpenAI API key

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add your `OPENAI_API_KEY` to `.env`. The sample database is created and seeded when the application starts. The included customers and account numbers are fictitious.

## Run

```bash
uvicorn banking_agent.app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. The default sample customer is `CUST1001` (salary ₹60,000, existing EMI ₹5,000, credit score 760).

## Chat example

Start a loan assessment:

```bash
curl -s http://127.0.0.1:8000/chat \
	-H 'Content-Type: application/json' \
	-d '{"customer_id":"CUST1001","message":"I want a personal loan of 500000 for 3 years. My monthly salary is 60000 and existing EMI is 5000. Calculate my EMI, check eligibility, and retrieve my account details."}'
```

The response includes a `session_id` and asks whether to submit. Pass that same ID with an explicit confirmation in a second request:

```bash
curl -s http://127.0.0.1:8000/chat \
	-H 'Content-Type: application/json' \
	-d '{"customer_id":"CUST1001","session_id":"SESSION_ID_FROM_FIRST_RESPONSE","message":"Yes, submit the application."}'
```

The service persists the pending offer and chat history in SQLite. Confirmation is accepted only for an eligible offer already presented in that same customer session. The agent's submission tool additionally checks a request-scoped authorization grant and the exact offer terms, so prompt instructions alone are not the safety boundary. “No” or “cancel” clears the pending offer.

## API

- `POST /chat`: agent conversation; body fields are `customer_id`, `message`, and optional `session_id`.
- `GET /customer/{id}`: retrieve the customer's stored profile.
- `POST /loan/apply`: direct application submission; requires `confirmation: "CONFIRM"`, rechecks eligibility, and computes EMI server-side.
- `GET /health`: process health check.

`/loan/apply` accepts `customer_id`, `loan_amount`, `tenure_years`, and the explicit `confirmation` field. It is a mock banking API, not a production lending integration.

## Loan rules

EMI uses the reducing-balance formula with a default annual interest rate of 10%. Eligibility uses the customer's stored salary, existing EMI, and credit score: salary must be at least ₹30,000, credit score at least 700, and `(existing EMI + new EMI) / salary` no greater than 0.5. Account details are fetched only after eligibility passes. The model cannot submit an application without an eligible, pending offer and explicit confirmation.

## Project layout

```text
banking_agent/
├── app.py
├── agent.py
├── database.py
├── models.py
├── api/
│   └── routes.py
└── tools/
		├── account_tool.py
		├── application_tool.py
		├── eligibility_tool.py
		└── emi_tool.py
```

For a real bank deployment, add customer authentication and authorization, use a managed database and secrets store, protect account details, and replace the mock application write with an authorized banking API. The SQLite chat session is state storage, not proof of customer identity.