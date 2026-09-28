"""Agent audit log: ``meta.agent_audit`` (one row per tool call) and
``meta.agent_questions`` (one row per question, written even on failure)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from agentic_warehouse_ops.common.warehouse import WarehouseEngine

AGENT_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS meta.agent_audit (
    audit_id VARCHAR PRIMARY KEY,
    question_id VARCHAR,
    run_id VARCHAR,
    ts TIMESTAMP,
    tool_name VARCHAR,
    tool_args JSON,
    compiled_sql VARCHAR,
    rows_returned BIGINT,
    bytes_scanned BIGINT,
    duration_ms BIGINT,
    guardrail_verdict VARCHAR,
    caller_role VARCHAR,
    llm_model VARCHAR,
    manifest_hash VARCHAR
)
"""

AGENT_QUESTIONS_DDL = """
CREATE TABLE IF NOT EXISTS meta.agent_questions (
    question_id VARCHAR PRIMARY KEY,
    run_id VARCHAR,
    question VARCHAR,
    answer VARCHAR,
    status VARCHAR,
    tool_call_count BIGINT,
    total_cost BIGINT,
    created_ts TIMESTAMP,
    manifest_hash VARCHAR,
    llm_model VARCHAR,
    caller_role VARCHAR
)
"""


def _utcnow() -> datetime:
    """Return the current UTC time as a naive datetime."""
    return datetime.now(UTC).replace(tzinfo=None)


def migrate_agent_audit(engine: WarehouseEngine) -> None:
    """Create the agent audit tables, idempotently.

    Args:
        engine: Warehouse engine receiving the tables.
    """
    engine.execute("CREATE SCHEMA IF NOT EXISTS meta")
    engine.execute(AGENT_AUDIT_DDL)
    engine.execute(AGENT_QUESTIONS_DDL)


def record_tool_call(
    engine: WarehouseEngine,
    *,
    audit_id: str,
    question_id: str,
    run_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    compiled_sql: str | None,
    rows_returned: int,
    bytes_scanned: int,
    duration_ms: int,
    guardrail_verdict: str,
    caller_role: str,
    llm_model: str,
    manifest_hash: str,
) -> None:
    """Write one ``meta.agent_audit`` row.

    Args:
        engine: Warehouse engine.
        audit_id: Unique id for this tool call.
        question_id: Owning question id.
        run_id: Owning run id.
        tool_name: Tool invoked.
        tool_args: Serialised tool arguments.
        compiled_sql: SQL the tool compiled (when applicable).
        rows_returned: Rows the tool returned (0 for denied calls).
        bytes_scanned: Estimated scanned bytes for the call.
        duration_ms: Wall-clock duration of the call.
        guardrail_verdict: ``"allowed"`` or ``"denied: <reason>"``.
        caller_role: Role on whose behalf the agent ran.
        llm_model: LLM model identifier.
        manifest_hash: dbt manifest hash the agent ran against.
    """
    import json

    engine.execute(
        "INSERT INTO meta.agent_audit"
        " (audit_id, question_id, run_id, ts, tool_name, tool_args, compiled_sql,"
        "  rows_returned, bytes_scanned, duration_ms, guardrail_verdict,"
        "  caller_role, llm_model, manifest_hash)"
        " VALUES (?, ?, ?, ?, ?, ?::JSON, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            audit_id,
            question_id,
            run_id,
            _utcnow(),
            tool_name,
            json.dumps(tool_args, default=str),
            compiled_sql,
            rows_returned,
            bytes_scanned,
            duration_ms,
            guardrail_verdict,
            caller_role,
            llm_model,
            manifest_hash,
        ],
    )


def record_question(
    engine: WarehouseEngine,
    *,
    question_id: str,
    run_id: str,
    question: str,
    answer: str,
    status: str,
    tool_call_count: int,
    total_cost: int,
    manifest_hash: str,
    llm_model: str,
    caller_role: str,
) -> None:
    """Write (or overwrite) one ``meta.agent_questions`` row.

    Args:
        engine: Warehouse engine.
        question_id: Unique question id.
        run_id: Owning run id.
        question: The natural-language question.
        answer: The final answer (may be partial on abort/failure).
        status: ``success``, ``aborted``, ``failed``, or ``rejected``.
        tool_call_count: Number of tool invocations (including denied).
        total_cost: Cumulative estimated bytes scanned.
        manifest_hash: dbt manifest hash the agent ran against.
        llm_model: LLM model identifier.
        caller_role: Role on whose behalf the agent ran.
    """
    engine.execute("DELETE FROM meta.agent_questions WHERE question_id = ?", [question_id])
    engine.execute(
        "INSERT INTO meta.agent_questions"
        " (question_id, run_id, question, answer, status, tool_call_count, total_cost,"
        "  created_ts, manifest_hash, llm_model, caller_role)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            question_id,
            run_id,
            question,
            answer,
            status,
            tool_call_count,
            total_cost,
            _utcnow(),
            manifest_hash,
            llm_model,
            caller_role,
        ],
    )


def list_audit(
    engine: WarehouseEngine,
    limit: int = 20,
    question_id: str | None = None,
) -> pd.DataFrame:
    """Return recent audit rows, optionally filtered by question id.

    Args:
        engine: Warehouse engine.
        limit: Maximum rows (most recent first).
        question_id: Optional question filter.

    Returns:
        A DataFrame of ``meta.agent_audit`` rows.
    """
    sql = "SELECT * FROM meta.agent_audit"
    params: list[Any] = []
    if question_id:
        sql += " WHERE question_id = ?"
        params.append(question_id)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    return engine.execute(sql, params)


def list_questions(
    engine: WarehouseEngine,
    limit: int = 20,
) -> pd.DataFrame:
    """Return recent question rows, most recent first.

    Args:
        engine: Warehouse engine.
        limit: Maximum rows.

    Returns:
        A DataFrame of ``meta.agent_questions`` rows.
    """
    return engine.execute(
        "SELECT * FROM meta.agent_questions ORDER BY created_ts DESC LIMIT ?", [limit]
    )


def get_question(engine: WarehouseEngine, question_id: str) -> dict[str, Any] | None:
    """Return one question row as a dict, or ``None``.

    Args:
        engine: Warehouse engine.
        question_id: Question id to fetch.

    Returns:
        The row as a dict, if present.
    """
    frame = engine.execute(
        "SELECT * FROM meta.agent_questions WHERE question_id = ?", [question_id]
    )
    if frame.empty:
        return None
    return {str(name): value for name, value in frame.iloc[0].items()}


def get_tool_calls(engine: WarehouseEngine, question_id: str) -> pd.DataFrame:
    """Return the tool-call audit rows for one question, oldest first.

    Args:
        engine: Warehouse engine.
        question_id: Question id.

    Returns:
        A DataFrame of ``meta.agent_audit`` rows.
    """
    return engine.execute(
        "SELECT * FROM meta.agent_audit WHERE question_id = ? ORDER BY ts ASC", [question_id]
    )
