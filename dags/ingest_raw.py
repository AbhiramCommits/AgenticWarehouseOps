"""Daily raw ingestion DAG.

Lands one day's S3 partitions into the warehouse raw schema, gates data
quality, and records the outcome in the ``meta.pipeline_runs`` registry:

    open_run -> [load_partition x5] -> data_quality_gate -> close_run

- Schedule: daily at midnight UTC, catchup enabled (backfill over the seeded
  window by updating ``start_date`` to match your generated data).
- Each source table loads in its own dynamically mapped task with
  retries=2 + exponential backoff.
- The data quality gate fails the DAG (and marks the run ``failed``) on row
  floor breaches, reject-ceiling breaches, or fully empty partitions.
- Every run writes exactly one ``meta.pipeline_runs`` row and one
  ``meta.load_audit`` row per table.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

from airflow.decorators import dag, task
from airflow.utils.trigger_rule import TriggerRule

from agentic_warehouse_ops.common.warehouse import WarehouseEngine, get_engine
from agentic_warehouse_ops.ingestion.quality import (
    DEFAULT_REJECT_CEILING,
    DEFAULT_ROW_FLOORS,
    evaluate_quality_gate,
)
from agentic_warehouse_ops.ingestion.registry import (
    close_run,
    migrate_registry,
    open_run,
    record_load_audit,
)
from agentic_warehouse_ops.ingestion.s3_to_warehouse import (
    LoadResult,
    load_partition,
)
from agentic_warehouse_ops.ingestion.schemas import SOURCE_TABLES

DAG_ID = "ingest_raw"

# Aligned with the default seed window (today - 30 days); update when you
# regenerate data/seeds for a different window.
START_DATE = datetime(2026, 8, 28)

DEFAULT_TASK_ARGS = {
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}


def _engine() -> WarehouseEngine:
    """Build a warehouse engine for the configured profile (DuckDB default)."""
    return get_engine(profile=os.environ.get("WAREHOUSE_PROFILE", "duckdb"))


@dag(
    dag_id=DAG_ID,
    schedule="0 0 * * *",
    start_date=START_DATE,
    catchup=True,
    max_active_runs=1,
    default_args=DEFAULT_TASK_ARGS,
    doc_md=__doc__,
    tags=["ingestion", "raw", "warehouse"],
)
def ingest_raw() -> None:
    """Define the ingest task graph."""

    @task(task_id="open_run")
    def open_run_task(**context: Any) -> str:
        """Create the ``meta.pipeline_runs`` row and return its run id."""
        engine = _engine()
        migrate_registry(engine)
        return open_run(
            engine,
            dag_id=DAG_ID,
            logical_date=context["logical_date"],
        )

    @task(task_id="load_partition")
    def load_partition_task(pipeline_run_id: str, table: str, **context: Any) -> dict[str, Any]:
        """Load one (table, logical_date) partition idempotently and audit it."""
        engine = _engine()
        partition_dt = context["logical_date"].date()
        result = load_partition(table=table, dt=partition_dt, engine=engine, run_id=pipeline_run_id)
        record_load_audit(
            engine,
            run_id=pipeline_run_id,
            table_name=result.table,
            partition_dt=result.dt,
            rows_loaded=result.rows_loaded,
            rows_rejected=result.rows_rejected,
            source_keys=result.source_keys,
            duration_ms=result.duration_ms,
        )
        return {
            "table": result.table,
            "dt": result.dt.isoformat(),
            "rows_loaded": result.rows_loaded,
            "rows_rejected": result.rows_rejected,
            "source_keys": result.source_keys,
            "duration_ms": result.duration_ms,
        }

    @task(task_id="data_quality_gate")
    def quality_gate_task(loads: list[dict[str, Any]]) -> dict[str, Any]:
        """Fail the DAG if any partition breaches floors/ceilings/emptiness."""
        results = [
            LoadResult(
                table=item["table"],
                dt=datetime.fromisoformat(item["dt"]).date(),
                rows_loaded=item["rows_loaded"],
                rows_rejected=item["rows_rejected"],
                source_keys=item["source_keys"],
                duration_ms=item["duration_ms"],
            )
            for item in loads
        ]
        return evaluate_quality_gate(
            results,
            row_floors=DEFAULT_ROW_FLOORS,
            reject_ceiling=DEFAULT_REJECT_CEILING,
        )

    @task(task_id="close_run", trigger_rule=TriggerRule.ALL_DONE)
    def close_run_task(gate_payload: dict[str, Any] | None, **context: Any) -> None:
        """Finalize the ``meta.pipeline_runs`` row; failed gate means ``failed``."""
        ti = context["ti"]
        registry_run_id = ti.xcom_pull(task_ids="open_run", key="return_value")
        status = "success" if gate_payload else "failed"
        if registry_run_id:
            close_run(_engine(), run_id=registry_run_id, status=status)

    run_id = open_run_task()
    load_results = load_partition_task.partial(pipeline_run_id=run_id).expand(
        table=list(SOURCE_TABLES)
    )
    gate = quality_gate_task(load_results)
    close_run_task(gate)


dag = ingest_raw()
