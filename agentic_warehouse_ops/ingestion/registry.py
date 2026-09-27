"""Run registry: ``meta.pipeline_runs`` / ``meta.load_audit`` plus raw DDL migrations.

The registry makes every load replayable and auditable: each DAG run writes
exactly one ``meta.pipeline_runs`` row and one ``meta.load_audit`` row per
source table.
"""

from __future__ import annotations

import types
import uuid
from datetime import UTC, date, datetime
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from agentic_warehouse_ops.common.warehouse import WarehouseEngine
from agentic_warehouse_ops.ingestion.schemas import TABLE_SCHEMAS

PIPELINE_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS meta.pipeline_runs (
    run_id VARCHAR PRIMARY KEY,
    dag_id VARCHAR,
    logical_date TIMESTAMP,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    status VARCHAR,
    git_sha VARCHAR,
    dbt_manifest_hash VARCHAR,
    UNIQUE (dag_id, logical_date)
)
"""

LOAD_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS meta.load_audit (
    run_id VARCHAR,
    table_name VARCHAR,
    partition_dt DATE,
    rows_loaded BIGINT,
    rows_rejected BIGINT,
    source_keys JSON,
    duration_ms BIGINT
)
"""

LOAD_COLUMNS: tuple[str, ...] = ("_ingested_at", "_source_key", "_run_id", "_file_hash")

_SQL_TYPES: dict[type[Any], str] = {
    int: "BIGINT",
    str: "VARCHAR",
    float: "DOUBLE",
    datetime: "TIMESTAMP",
    date: "DATE",
}


def _utcnow() -> datetime:
    """Return the current UTC time as a naive datetime for TIMESTAMP columns."""
    return datetime.now(UTC).replace(tzinfo=None)


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Split an annotation into (inner_type, is_required).

    Handles ``Optional[X]`` / ``X | None``; returns ``(annotation, True)``
    for everything else.
    """
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        args = get_args(annotation)
        inner = next(a for a in args if a is not type(None))
        return inner, False
    return annotation, True


def _sql_type(annotation: Any) -> str:
    """Map a row-schema annotation to a portable SQL column type."""
    inner, _ = _unwrap_optional(annotation)
    return _SQL_TYPES[inner]


def _raw_table_ddl(table: str, model: type[BaseModel]) -> str:
    """Build the ``raw.<table>`` DDL from a pydantic row schema."""
    columns: list[str] = []
    for name, field in model.model_fields.items():
        _, required = _unwrap_optional(field.annotation)
        suffix = " NOT NULL" if required else ""
        columns.append(f"{name} {_sql_type(field.annotation)}{suffix}")
    columns.append("dt DATE")
    columns.extend(
        f"{col} VARCHAR" if col != "_ingested_at" else f"{col} TIMESTAMP" for col in LOAD_COLUMNS
    )
    return f"CREATE TABLE IF NOT EXISTS raw.{table} ({', '.join(columns)})"


def _reject_table_ddl(table: str, model: type[BaseModel]) -> str:
    """Build the ``raw.<table>_rejects`` DDL; business columns are VARCHAR."""
    columns = [f"{name} VARCHAR" for name in model.model_fields]
    columns.append("dt DATE")
    columns.extend(
        f"{col} VARCHAR" if col != "_ingested_at" else f"{col} TIMESTAMP" for col in LOAD_COLUMNS
    )
    columns.append("reject_reason VARCHAR")
    return f"CREATE TABLE IF NOT EXISTS raw.{table}_rejects ({', '.join(columns)})"


def migrate_registry(engine: WarehouseEngine) -> None:
    """Create the ``raw``/``meta`` schemas and all loader tables, idempotently.

    Creates ``raw.<table>`` and ``raw.<table>_rejects`` for every source table
    plus the ``meta.pipeline_runs`` and ``meta.load_audit`` registry tables.

    Args:
        engine: Warehouse engine to migrate.
    """
    engine.execute("CREATE SCHEMA IF NOT EXISTS raw")
    engine.execute("CREATE SCHEMA IF NOT EXISTS meta")
    for table, model in TABLE_SCHEMAS.items():
        engine.execute(_raw_table_ddl(table, model))
        engine.execute(_reject_table_ddl(table, model))
    engine.execute(PIPELINE_RUNS_DDL)
    engine.execute(LOAD_AUDIT_DDL)


def open_run(
    engine: WarehouseEngine,
    dag_id: str,
    logical_date: datetime,
    git_sha: str | None = None,
    dbt_manifest_hash: str | None = None,
) -> str:
    """Start (or resume) a pipeline run and return its ``run_id``.

    Idempotent per ``(dag_id, logical_date)``: if a row already exists (e.g.
    after a task retry) its ``run_id`` is returned instead of inserting again,
    guaranteeing exactly one ``meta.pipeline_runs`` row per DAG run.

    Args:
        engine: Warehouse engine.
        dag_id: DAG identifier.
        logical_date: DAG run logical date.
        git_sha: Optional source revision.
        dbt_manifest_hash: Optional dbt manifest hash.

    Returns:
        The run id for this DAG run.
    """
    if logical_date.tzinfo is not None:
        logical_date = logical_date.replace(tzinfo=None)
    existing = engine.execute(
        "SELECT run_id FROM meta.pipeline_runs WHERE dag_id = ? AND logical_date = ?",
        [dag_id, logical_date],
    )
    if not existing.empty:
        return str(existing.iloc[0]["run_id"])
    run_id = uuid.uuid4().hex
    engine.execute(
        "INSERT INTO meta.pipeline_runs"
        " (run_id, dag_id, logical_date, started_at, ended_at, status, git_sha, dbt_manifest_hash)"
        " VALUES (?, ?, ?, ?, NULL, 'running', ?, ?)",
        [run_id, dag_id, logical_date, _utcnow(), git_sha, dbt_manifest_hash],
    )
    return run_id


def close_run(engine: WarehouseEngine, run_id: str, status: str) -> None:
    """Finalize a pipeline run with its terminal ``status``.

    Args:
        engine: Warehouse engine.
        run_id: Run id previously returned by :func:`open_run`.
        status: Terminal status, e.g. ``"success"`` or ``"failed"``.
    """
    engine.execute(
        "UPDATE meta.pipeline_runs SET ended_at = ?, status = ? WHERE run_id = ?",
        [_utcnow(), status, run_id],
    )


def record_load_audit(
    engine: WarehouseEngine,
    run_id: str,
    table_name: str,
    partition_dt: date,
    rows_loaded: int,
    rows_rejected: int,
    source_keys: list[str],
    duration_ms: int,
) -> None:
    """Record one ``meta.load_audit`` row for a (table, partition) load.

    Args:
        engine: Warehouse engine.
        run_id: Run id the load belongs to.
        table_name: Source table name.
        partition_dt: Partition date that was loaded.
        rows_loaded: Number of valid rows written to ``raw.<table>``.
        rows_rejected: Number of invalid rows written to ``raw.<table>_rejects``.
        source_keys: S3 keys that were read for this partition.
        duration_ms: Load wall-clock duration in milliseconds.
    """
    import json

    engine.execute(
        "INSERT INTO meta.load_audit"
        " (run_id, table_name, partition_dt, rows_loaded, rows_rejected, source_keys, duration_ms)"
        " VALUES (?, ?, ?, ?, ?, ?::JSON, ?)",
        [
            run_id,
            table_name,
            partition_dt,
            rows_loaded,
            rows_rejected,
            json.dumps(source_keys),
            duration_ms,
        ],
    )
