"""FastAPI endpoint exposing the agent as ``POST /ask``."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agentic_warehouse_ops.agent.graph import build_agent
from agentic_warehouse_ops.agent.models import AgentAnswer

app = FastAPI(title="Agentic Warehouse Ops Agent", version="0.1.0")


class AskRequest(BaseModel):
    """Request body for /ask."""

    question: str = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def _get_agent() -> Callable[..., AgentAnswer]:
    """Build (once) the agent bound to local defaults."""
    return build_agent()


@app.post("/ask", response_model=AgentAnswer)
def ask(request: AskRequest) -> AgentAnswer:
    """Answer a natural-language question about the governed marts."""
    return _get_agent()(request.question)
