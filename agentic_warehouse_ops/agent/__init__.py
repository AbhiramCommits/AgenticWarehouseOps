"""Agentic layer: typed tool-calling agent over the governed marts.

The agent answers natural-language questions by choosing among typed tools
(``query_mart``, ``vector_search_tickets``, ``lookup_schema``); it never
emits free-form SQL, and PII columns are structurally inaccessible to it.
"""

from agentic_warehouse_ops.agent.graph import FakeLLM, build_agent, get_llm, run_agent
from agentic_warehouse_ops.agent.index import DuckDBVectorStore, VectorStore, get_embedder
from agentic_warehouse_ops.agent.models import AgentAnswer, AgentStep, ToolCallRecord

__all__ = [
    "AgentAnswer",
    "AgentStep",
    "DuckDBVectorStore",
    "FakeLLM",
    "ToolCallRecord",
    "VectorStore",
    "build_agent",
    "get_embedder",
    "get_llm",
    "run_agent",
]
