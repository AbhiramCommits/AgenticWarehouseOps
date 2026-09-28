"""Tests for the agentic layer: typed tools, vector search, graph, embeddings."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agentic_warehouse_ops.agent.embedder import FakeEmbedder
from agentic_warehouse_ops.agent.graph import FakeLLM, run_agent
from agentic_warehouse_ops.agent.index import (
    DuckDBVectorStore,
    embed_catalog,
    embed_support_tickets,
)
from agentic_warehouse_ops.agent.tools import (
    FilterClause,
    ToolContext,
    build_tools,
    compile_query,
)
from agentic_warehouse_ops.common.warehouse import DuckDBEngine, WarehouseEngine
from agentic_warehouse_ops.governance.catalog import compute_manifest_hash


def _col(
    name: str,
    *,
    pii: bool = False,
    classification: str = "public",
    description: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "meta": {"pii": pii, "classification": classification, "owner": "analytics"},
    }


def _model(
    name: str,
    columns: dict[str, dict[str, Any]],
    *,
    schema: str = "raw_marts",
    materialized: str = "table",
    description: str = "",
) -> dict[str, Any]:
    return {
        "resource_type": "model",
        "package_name": "warehouse",
        "name": name,
        "schema": schema,
        "unique_id": f"model.warehouse.{name}",
        "config": {"materialized": materialized},
        "depends_on": {"nodes": []},
        "description": description,
        "columns": columns,
    }


def _manifest() -> dict[str, Any]:
    """Fixture manifest whose marts match the seeded warehouse tables."""
    return {
        "nodes": {
            node["unique_id"]: node
            for node in [
                _model(
                    "dim_customer",
                    {
                        "customer_id": _col("customer_id", classification="internal"),
                        "full_name": _col("full_name", pii=True, classification="restricted"),
                        "email": _col("email", pii=True, classification="restricted"),
                        "phone": _col("phone", pii=True, classification="restricted"),
                        "signup_date": _col("signup_date", classification="internal"),
                        "country": _col("country"),
                        "segment": _col("segment"),
                    },
                ),
                _model(
                    "fct_orders",
                    {
                        "order_id": _col("order_id", classification="internal"),
                        "customer_id": _col("customer_id", classification="internal"),
                        "order_ts": _col("order_ts", classification="internal"),
                        "status": _col("status"),
                        "channel": _col("channel"),
                        "currency": _col("currency"),
                        "order_total": _col("order_total"),
                        "item_count": _col("item_count"),
                        "fulfilment_latency_days": _col("fulfilment_latency_days"),
                    },
                ),
                _model(
                    "fct_order_items",
                    {
                        "order_item_id": _col("order_item_id", classification="internal"),
                        "order_id": _col("order_id", classification="internal"),
                        "sku": _col("sku", classification="internal"),
                        "quantity": _col("quantity"),
                        "line_total": _col("line_total"),
                    },
                ),
                _model(
                    "dim_product",
                    {
                        "sku": _col("sku", classification="internal"),
                        "product_name": _col("product_name"),
                        "category": _col("category"),
                        "supplier": _col("supplier", classification="internal"),
                        "list_price": _col("list_price"),
                    },
                ),
                _model(
                    "mart_daily_revenue",
                    {
                        "revenue_date": _col("revenue_date"),
                        "channel": _col("channel"),
                        "category": _col("category"),
                        "revenue": _col("revenue"),
                        "orders": _col("orders"),
                        "units_sold": _col("units_sold"),
                    },
                ),
                _model(
                    "mart_customer_support_summary",
                    {
                        "customer_id": _col("customer_id", classification="internal"),
                        "ticket_count": _col("ticket_count"),
                        "first_ticket_ts": _col("first_ticket_ts", classification="internal"),
                        "last_ticket_ts": _col("last_ticket_ts", classification="internal"),
                        "open_ticket_count": _col("open_ticket_count"),
                    },
                ),
            ]
        },
        "sources": {},
    }


@pytest.fixture
def manifest() -> dict[str, Any]:
    return _manifest()


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[WarehouseEngine]:
    yield DuckDBEngine(str(tmp_path / "store.duckdb"))


@pytest.fixture
def seeded_engine(engine: WarehouseEngine) -> WarehouseEngine:
    """Warehouse with marts tables matching the fixture manifest."""
    return seed_tables(engine)


def seed_tables(engine: WarehouseEngine) -> WarehouseEngine:
    """Create marts + support-ticket tables matching the fixture manifest."""
    engine.execute("CREATE SCHEMA raw_marts")
    engine.execute(
        "CREATE TABLE raw_marts.dim_customer (customer_id BIGINT, full_name VARCHAR,"
        " email VARCHAR, phone VARCHAR, signup_date DATE, country VARCHAR, segment VARCHAR)"
    )
    engine.execute(
        "INSERT INTO raw_marts.dim_customer VALUES"
        " (1, 'Ava Smith', 'ava@example.com', '+15551234567', DATE '2026-08-29', 'US', 'premium'),"
        " (2, 'Liam Jones', 'liam@example.com', '+15559876543', DATE '2026-09-01', 'UK', 'standard'),"
        " (3, 'Mia Lee', 'mia@example.com', '+15551112233', DATE '2026-09-10', 'DE', 'new')"
    )
    engine.execute(
        "CREATE TABLE raw_marts.fct_orders (order_id BIGINT, customer_id BIGINT,"
        " order_ts TIMESTAMP, status VARCHAR, channel VARCHAR, currency VARCHAR,"
        " order_total DOUBLE, item_count BIGINT, fulfilment_latency_days BIGINT)"
    )
    engine.execute(
        "INSERT INTO raw_marts.fct_orders VALUES"
        " (101, 1, TIMESTAMP '2026-08-30 10:00:00', 'delivered', 'web', 'USD', 123.45, 2, NULL),"
        " (102, 2, TIMESTAMP '2026-09-05 09:30:00', 'shipped', 'mobile', 'EUR', 67.80, 1, 21)"
    )
    engine.execute("CREATE SCHEMA raw")
    engine.execute(
        "CREATE TABLE raw.support_tickets (ticket_id BIGINT, customer_id BIGINT,"
        " created_ts TIMESTAMP, channel VARCHAR, subject VARCHAR, body_text VARCHAR)"
    )
    engine.execute(
        "INSERT INTO raw.support_tickets VALUES"
        " (1, 1, TIMESTAMP '2026-09-01 08:00:00', 'web', 'Delivery delay',"
        "  'Hi, my order is late and the shipping delay is really frustrating.'),"
        " (2, 2, TIMESTAMP '2026-09-02 09:00:00', 'email', 'Refund request',"
        "  'Please process a refund for order 102, contact me at liam@example.com.'),"
        " (3, 3, TIMESTAMP '2026-09-03 10:00:00', 'web', 'Login issue',"
        "  'I cannot log in to my account, password reset never arrives.')"
    )
    return engine


def _tools(
    engine: WarehouseEngine, manifest: dict[str, Any], embedder: FakeEmbedder
) -> dict[str, Any]:
    store = DuckDBVectorStore(engine, embedder)
    return build_tools(
        ToolContext(engine=engine, manifest=manifest, vector_store=store, embedder=embedder)
    )


def test_compile_query_for_each_filter_operator(manifest: dict[str, Any]) -> None:
    """Every operator compiles to the expected parameterised SQL shape."""
    node = manifest["nodes"]["model.warehouse.dim_customer"]
    cases = [
        (FilterClause(column="country", op="eq", value="US"), "country = ?", ["US"]),
        (FilterClause(column="segment", op="ne", value="new"), "segment != ?", ["new"]),
        (FilterClause(column="customer_id", op="gt", value=5), "customer_id > ?", [5]),
        (FilterClause(column="customer_id", op="gte", value=5), "customer_id >= ?", [5]),
        (FilterClause(column="customer_id", op="lt", value=9), "customer_id < ?", [9]),
        (FilterClause(column="customer_id", op="lte", value=9), "customer_id <= ?", [9]),
        (FilterClause(column="segment", op="like", value="pre%"), "segment LIKE ?", ["pre%"]),
        (
            FilterClause(column="country", op="in", value=["US", "UK"]),
            "country IN (?, ?)",
            ["US", "UK"],
        ),
        (FilterClause(column="signup_date", op="is_null", value=None), "signup_date IS NULL", []),
        (
            FilterClause(column="signup_date", op="is_not_null", value=None),
            "signup_date IS NOT NULL",
            [],
        ),
    ]
    for clause, expected_fragment, expected_params in cases:
        sql, params = compile_query(node, ["country"], [clause], [], [], 50)
        assert expected_fragment in sql
        assert params == expected_params


def test_compile_query_rejects_unknown_mart(
    manifest: dict[str, Any], engine: WarehouseEngine
) -> None:
    """Unknown marts and unknown/restricted columns are rejected before SQL runs."""
    tools = _tools(engine, manifest, FakeEmbedder())
    with pytest.raises(Exception, match="dim_customer"):
        tools["query_mart"].invoke({"mart": "customer_secrets", "select": ["country"], "limit": 5})
    with pytest.raises(Exception, match="unknown column"):
        tools["query_mart"].invoke({"mart": "dim_customer", "select": ["not_a_column"], "limit": 5})
    with pytest.raises(Exception, match="restricted"):
        tools["query_mart"].invoke({"mart": "dim_customer", "select": ["email"], "limit": 5})


def test_query_mart_parameterizes_injection_attempt(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """Injection payloads in string values are bound as parameters, never inlined."""
    tools = _tools(seeded_engine, manifest, FakeEmbedder())
    payload = "x'; DROP TABLE raw_marts.dim_customer;--"
    result = tools["query_mart"].invoke(
        {
            "mart": "dim_customer",
            "select": ["country", "segment"],
            "filters": [{"column": "country", "op": "eq", "value": payload}],
            "limit": 5,
        }
    )
    assert result["row_count"] == 0
    sql = result["sql"]
    assert "?" in sql
    assert payload not in sql
    assert "DROP" not in sql
    # The table survived the attempted injection.
    rows = seeded_engine.execute("SELECT COUNT(*) AS n FROM raw_marts.dim_customer")
    assert int(rows.iloc[0]["n"]) == 3


def test_vector_search_returns_relevant_tickets(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """Vector search ranks the lexically closest ticket first and redacts PII."""
    embedder = FakeEmbedder()
    embed_support_tickets(seeded_engine, embedder=embedder)
    tools = _tools(seeded_engine, manifest, embedder)

    result = tools["vector_search_tickets"].invoke({"query": "shipping delay", "k": 3})
    assert result["results"][0]["ticket_id"] == "1"

    result = tools["vector_search_tickets"].invoke(
        {"query": "log in account", "k": 3, "channel": "web"}
    )
    assert result["results"][0]["ticket_id"] == "3"

    refund = tools["vector_search_tickets"].invoke({"query": "refund contact email", "k": 3})
    snippet = refund["results"][0]["snippet"]
    assert "liam@example.com" not in snippet
    assert "[EMAIL]" in snippet


def test_lookup_schema_flags_restricted_columns(
    engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """Restricted columns are listed by name but flagged unavailable."""
    tools = _tools(engine, manifest, FakeEmbedder())
    result = tools["lookup_schema"].invoke({"model": "dim_customer"})
    model = result["models"][0]
    by_name = {column["name"]: column for column in model["columns"]}
    assert by_name["email"]["available"] is False
    assert by_name["country"]["available"] is True
    assert "data_type" not in by_name["email"]


def test_graph_terminates_within_iteration_cap(
    engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """A greedy agent that always asks for another tool stops at 5 iterations."""
    tools = _tools(engine, manifest, FakeEmbedder())
    step = {"action": "tool", "tool": "lookup_schema", "args": {}, "reason": "keep looking"}
    llm = FakeLLM(script=[json.dumps(step)] * 5 + ["Not enough information found."])
    answer = run_agent(
        "What does the schema look like?",
        tools=tools,
        llm=llm,
        manifest=manifest,
        run_id="cap-test",
    )
    assert len(answer.tool_calls) == 5
    assert answer.answer == "Not enough information found."
    assert answer.run_id == "cap-test"


def test_end_to_end_ask_with_fake_llm(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """A full ask with FakeLLM returns a populated AgentAnswer."""
    tools = _tools(seeded_engine, manifest, FakeEmbedder())
    llm = FakeLLM(
        script=[
            json.dumps(
                {
                    "action": "tool",
                    "tool": "query_mart",
                    "args": {"mart": "dim_customer", "select": ["country", "segment"], "limit": 3},
                    "reason": "inspect customer segments",
                }
            ),
            json.dumps({"action": "finish", "reason": "have the answer"}),
            "There are 3 customers across US, UK, and DE.",
        ]
    )
    answer = run_agent(
        "How many customers do we have and where are they?",
        tools=tools,
        llm=llm,
        manifest=manifest,
        run_id="e2e-1",
    )
    assert answer.answer == "There are 3 customers across US, UK, and DE."
    assert len(answer.tool_calls) == 1
    call = answer.tool_calls[0]
    assert call.name == "query_mart"
    assert call.row_count == 3
    assert call.error is None
    assert len(answer.sql_executed) == 1
    assert "FROM raw_marts.dim_customer" in answer.sql_executed[0]
    assert answer.manifest_hash == compute_manifest_hash(manifest)
    assert answer.run_id == "e2e-1"


class _CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that counts embedding calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.texts_embedded: list[str] = []

    def embed(self, texts: list[str]) -> Any:
        self.calls += 1
        self.texts_embedded.extend(texts)
        return super().embed(texts)


def test_embed_jobs_skip_unchanged_rows(
    seeded_engine: WarehouseEngine, manifest: dict[str, Any]
) -> None:
    """Re-running embedding jobs embeds nothing new (content-hash skip)."""
    embedder = _CountingEmbedder()
    first = embed_support_tickets(seeded_engine, embedder=embedder)
    assert first.total == 3
    assert first.embedded == 3
    calls_after_first = embedder.calls

    second = embed_support_tickets(seeded_engine, embedder=embedder)
    assert second.embedded == 0
    assert second.skipped == 3
    assert embedder.calls == calls_after_first

    catalog = embed_catalog(seeded_engine, manifest, embedder=embedder)
    assert catalog.embedded > 0
    again = embed_catalog(seeded_engine, manifest, embedder=embedder)
    assert again.embedded == 0
