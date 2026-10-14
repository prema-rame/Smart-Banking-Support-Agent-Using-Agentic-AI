"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from banking_agent.api.routes import router
from banking_agent.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Smart Banking Support Agent",
    description="A single LangChain agent backed by guarded banking tools and SQLite demo data.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}