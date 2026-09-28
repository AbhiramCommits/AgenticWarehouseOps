"""Typed agent tools over the governed marts.

There is deliberately no free-text SQL parameter anywhere: every argument is
schema-validated by pydantic, and :func:`compile_query` constructs the only
SQL shape this layer ever executes — a single parameterised ``SELECT``
against a known mart model, with identifiers whitelisted and every value
bound as a ``?`` placeholder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

from agentic_warehouse_ops.agent.embedder import Embedder
from agentic_warehouse_ops.agent.index import VectorStore
from agentic_warehouse_ops.common.warehouse import WarehouseEngine

MART_NAMES: tuple[str, ...] = (
    "dim_customer",
    "dim_product",
    "fct_orders",
    "fct_order_items",
    "mart_daily_revenue",
    "mart_customer_support_summary",
)

FILTER_OPS: tuple[str, ...] = (
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "like",
    "in",
    "is_null",
    "is_not_null",
)

_MAX_RECORDS = 200
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d{7,15}")


class ToolError(Exception):
    """Raised when a tool argument fails structural validation."""


def _is_restricted(column: dict[str, Any]) -> bool:
    meta = column.get("meta") or {}
    return meta.get("classification") == "restricted" or meta.get("pii") is True


def _json_safe(value: Any) -> Any:
    """Convert a pandas scalar into a JSON-serialisable value."""
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def redact_text(text: str) -> str:
    """Mask email addresses and phone numbers in free text."""
    masked = _EMAIL_RE.sub("[EMAIL]", text)
    return _PHONE_RE.sub("[PHONE]", masked)


class FilterClause(BaseModel):
    """One typed filter condition; values are always parameterised."""

    column: str = Field(min_length=1, max_length=64)
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "like", "in", "is_null", "is_not_null"]
    value: Any = None

    @model_validator(mode="after")
    def _validate_value(self) -> FilterClause:
        if self.op == "in":
            if not isinstance(self.value, list) or not self.value:
                raise ValueError("'in' filter requires a non-empty list value")
        elif self.op not in ("is_null", "is_not_null") and self.value is None:
            raise ValueError(f"'{self.op}' filter requires a value")
        if self.op == "like" and not isinstance(self.value, str):
            raise ValueError("'like' filter requires a string value")
        return self


class OrderByClause(BaseModel):
    """One order-by term."""

    column: str = Field(min_length=1, max_length=64)
    direction: Literal["asc", "desc"] = "asc"


class QueryMartInput(BaseModel):
    """Arguments for the query_mart tool. No SQL is ever accepted here."""

    mart: Literal[
        "dim_customer",
        "dim_product",
        "fct_orders",
        "fct_order_items",
        "mart_daily_revenue",
        "mart_customer_support_summary",
    ]
    select: list[str] = Field(min_length=1, max_length=20)
    filters: list[FilterClause] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list, max_length=10)
    order_by: list[OrderByClause] = Field(default_factory=list, max_length=10)
    limit: int = Field(default=50, ge=1, le=1000)


class VectorSearchTicketsInput(BaseModel):
    """Arguments for the vector_search_tickets tool."""

    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=25)
    start_date: date | None = None
    end_date: date | None = None
    channel: Literal["web", "email", "phone", "chat"] | None = None


class LookupSchemaInput(BaseModel):
    """Arguments for the lookup_schema tool."""

    model: str | None = Field(default=None, max_length=64)


@dataclass
class ToolContext:
    """Runtime dependencies shared by the tools."""

    engine: WarehouseEngine
    manifest: dict[str, Any]
    vector_store: VectorStore
    embedder: Embedder
    catalog: dict[str, Any] | None = field(default=None)


def _mart_node(manifest: dict[str, Any], mart: str) -> dict[str, Any]:
    """Return the manifest node for a mart name, or raise :class:`ToolError`."""
    for node in manifest.get("nodes", {}).values():
        if node.get("resource_type") == "model" and node.get("name") == mart:
            return node
    raise ToolError(f"unknown mart '{mart}'")


def _validate_identifier(name: str, columns: dict[str, Any], role: str) -> None:
    if not _IDENTIFIER_RE.match(name):
        raise ToolError(f"invalid {role} identifier: {name!r}")
    column = columns.get(name)
    if column is None:
        raise ToolError(f"unknown column '{name}' for {role}")
    if _is_restricted(column):
        raise ToolError(f"column '{name}' is restricted and cannot be queried")


def compile_query(
    model: dict[str, Any],
    select: list[str],
    filters: list[FilterClause],
    group_by: list[str],
    order_by: list[OrderByClause],
    limit: int,
) -> tuple[str, list[Any]]:
    """Compile typed arguments into one parameterised SELECT.

    Args:
        model: Manifest node of the target mart.
        select: Whitelisted, non-restricted columns.
        filters: Typed filter clauses; values become ``?`` parameters.
        group_by: Optional grouping columns (must contain ``select``).
        order_by: Optional order-by terms (must be in ``select``).
        limit: Row limit, clamped to ``[1, 1000]``.

    Returns:
        ``(sql, params)`` where every user value is a bound parameter.

    Raises:
        ToolError: On any unknown column, restricted column, or malformed
            identifier — the SQL is never executed in that case.
    """
    columns = model.get("columns") or {}
    if group_by and not set(select).issubset(set(group_by)):
        raise ToolError("select columns must be a subset of group_by when grouping")
    for column in select:
        _validate_identifier(column, columns, "select")
    for clause in filters:
        _validate_identifier(clause.column, columns, "filter")
    for term in order_by:
        _validate_identifier(term.column, columns, "order_by")
        if term.column not in select:
            raise ToolError(f"order_by column '{term.column}' must be in select")

    sql = f"SELECT {', '.join(select)} FROM {model['schema']}.{model['name']}"
    params: list[Any] = []
    if filters:
        clauses: list[str] = []
        for clause in filters:
            column = clause.column
            if clause.op == "eq":
                clauses.append(f"{column} = ?")
                params.append(clause.value)
            elif clause.op == "ne":
                clauses.append(f"{column} != ?")
                params.append(clause.value)
            elif clause.op == "gt":
                clauses.append(f"{column} > ?")
                params.append(clause.value)
            elif clause.op == "gte":
                clauses.append(f"{column} >= ?")
                params.append(clause.value)
            elif clause.op == "lt":
                clauses.append(f"{column} < ?")
                params.append(clause.value)
            elif clause.op == "lte":
                clauses.append(f"{column} <= ?")
                params.append(clause.value)
            elif clause.op == "like":
                clauses.append(f"{column} LIKE ?")
                params.append(clause.value)
            elif clause.op == "in":
                clauses.append(f"{column} IN ({', '.join('?' for _ in clause.value)})")
                params.extend(clause.value)
            elif clause.op == "is_null":
                clauses.append(f"{column} IS NULL")
            elif clause.op == "is_not_null":
                clauses.append(f"{column} IS NOT NULL")
            else:  # pragma: no cover - Literal prevents this
                raise ToolError(f"unsupported filter op '{clause.op}'")
        sql += " WHERE " + " AND ".join(clauses)
    if group_by:
        sql += " GROUP BY " + ", ".join(group_by)
    if order_by:
        sql += " ORDER BY " + ", ".join(
            f"{term.column} {'DESC' if term.direction == 'desc' else 'ASC'}" for term in order_by
        )
    sql += f" LIMIT {max(1, min(limit, 1000))}"
    return sql, params


def _query_mart(
    mart: str,
    select: list[str],
    filters: list[FilterClause],
    group_by: list[str],
    order_by: list[OrderByClause],
    limit: int,
    *,
    engine: WarehouseEngine,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Run a typed query against one mart; returns records, count, and the SQL."""
    node = _mart_node(manifest, mart)
    sql, params = compile_query(node, select, filters, group_by, order_by, limit)
    frame = engine.execute(sql, params)
    records = [
        {name: _json_safe(value) for name, value in record.items()}
        for record in frame.head(_MAX_RECORDS).to_dict("records")
    ]
    return {"records": records, "row_count": int(len(frame)), "sql": sql}


def _vector_search_tickets(
    query: str,
    k: int,
    start_date: date | None,
    end_date: date | None,
    channel: Literal["web", "email", "phone", "chat"] | None,
    *,
    vector_store: VectorStore,
) -> dict[str, Any]:
    """Search embedded support tickets; snippets are redacted."""
    filters: dict[str, Any] = {"kind": "ticket"}
    if channel:
        filters["channel"] = channel
    hits = vector_store.search(query, k=25, filters=filters)
    results: list[dict[str, Any]] = []
    for hit in hits:
        metadata = hit["metadata"]
        created = metadata.get("created_ts")
        if start_date and (created is None or created < start_date.isoformat()):
            continue
        if end_date and (created is None or created > end_date.isoformat()):
            continue
        results.append(
            {
                "ticket_id": metadata.get("ticket_id"),
                "snippet": redact_text(str(hit["content"]))[:200],
                "channel": metadata.get("channel"),
                "created_ts": created,
                "score": round(float(hit["score"]), 4),
            }
        )
        if len(results) >= k:
            break
    return {"results": results, "row_count": len(results)}


def _lookup_schema(
    model: str | None,
    *,
    manifest: dict[str, Any],
    catalog: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe models and columns; restricted columns are flagged unavailable."""
    nodes = [
        node
        for node in manifest.get("nodes", {}).values()
        if node.get("resource_type") == "model" and node.get("package_name") == "warehouse"
    ]
    if model:
        nodes = [node for node in nodes if node.get("name") == model]
        if not nodes:
            raise ToolError(f"unknown model '{model}'")
    entries: list[dict[str, Any]] = []
    for node in sorted(nodes, key=lambda item: item.get("name", "")):
        columns: list[dict[str, Any]] = []
        for col_name, column in (node.get("columns") or {}).items():
            meta = column.get("meta") or {}
            restricted = _is_restricted(column)
            entry: dict[str, Any] = {
                "name": col_name,
                "description": column.get("description", ""),
                "classification": meta.get("classification", "public"),
                "available": not restricted,
            }
            if not restricted:
                catalog_node = (
                    (catalog or {}).get("nodes", {}).get(node.get("unique_id", ""), {})
                    if catalog
                    else {}
                )
                data_type = (catalog_node.get("columns") or {}).get(col_name, {}).get("type")
                entry["data_type"] = data_type
            columns.append(entry)
        entries.append(
            {
                "name": node.get("name"),
                "schema": node.get("schema"),
                "materialized": (node.get("config") or {}).get("materialized"),
                "description": node.get("description", ""),
                "columns": columns,
            }
        )
    return {"models": entries, "row_count": len(entries)}


def build_tools(context: ToolContext) -> dict[str, StructuredTool]:
    """Build the three typed tools bound to a :class:`ToolContext`.

    Args:
        context: Runtime dependencies (engine, manifest, vector store,
            embedder, optional catalog).

    Returns:
        Tools keyed by name: ``query_mart``, ``vector_search_tickets``,
        ``lookup_schema``.
    """

    def query_mart(**kwargs: Any) -> dict[str, Any]:
        return _query_mart(**kwargs, engine=context.engine, manifest=context.manifest)

    def vector_search_tickets(**kwargs: Any) -> dict[str, Any]:
        return _vector_search_tickets(**kwargs, vector_store=context.vector_store)

    def lookup_schema(**kwargs: Any) -> dict[str, Any]:
        return _lookup_schema(**kwargs, manifest=context.manifest, catalog=context.catalog)

    tools: dict[str, StructuredTool] = {}
    tools["query_mart"] = StructuredTool.from_function(
        func=query_mart,
        name="query_mart",
        description=(
            "Run a typed, parameterised SELECT against one governed mart. "
            "Only the declared columns may be selected; restricted (PII) columns are "
            "rejected. Filters are typed and parameterised — never write SQL yourself. "
            f"Available marts: {', '.join(MART_NAMES)}. Filter ops: {', '.join(FILTER_OPS)}."
        ),
        args_schema=QueryMartInput,
    )
    tools["vector_search_tickets"] = StructuredTool.from_function(
        func=vector_search_tickets,
        name="vector_search_tickets",
        description=(
            "Semantic search over support-ticket text. Returns redacted snippets and "
            "ticket ids; never returns raw ticket body text."
        ),
        args_schema=VectorSearchTicketsInput,
    )
    tools["lookup_schema"] = StructuredTool.from_function(
        func=lookup_schema,
        name="lookup_schema",
        description=(
            "Look up model/column names, types, descriptions, and classifications from "
            "the dbt manifest. Restricted columns are listed but flagged unavailable."
        ),
        args_schema=LookupSchemaInput,
    )
    return tools


__all__ = [
    "FILTER_OPS",
    "MART_NAMES",
    "FilterClause",
    "LookupSchemaInput",
    "OrderByClause",
    "QueryMartInput",
    "ToolContext",
    "ToolError",
    "VectorSearchTicketsInput",
    "build_tools",
    "compile_query",
    "redact_text",
]
