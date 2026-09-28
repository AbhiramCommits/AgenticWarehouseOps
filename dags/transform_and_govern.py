"""Transform and govern DAG.

Runs after ``ingest_raw`` finishes for the same logical date:

    wait_for_ingest -> dbt deps -> dbt build -> emit lineage -> apply grants -> stamp manifest hash

- ``dbt build`` is the gate: any test failure fails the DAG run.
- The lineage artifact lands in ``artifacts/lineage/<ingest_run_id>.json``.
- Grants (restricted-stripped analyst views) are (re)applied from the manifest
  on every run, and the ingest run's ``meta.pipeline_runs`` row is stamped
  with the manifest hash.

Heavy third-party imports (pandas, duckdb) are kept inside task bodies so the
DAG file parses quickly in the scheduler.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.sensors.external_task import ExternalTaskSensor

DAG_ID = "transform_and_govern"
START_DATE = datetime(2026, 8, 28)

DBT_DIR = "/opt/airflow/dbt"
ARTIFACTS_DIR = "/opt/airflow/artifacts/lineage"

DEFAULT_TASK_ARGS = {
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}


def _engine() -> Any:
    """Build a warehouse engine for the configured profile (DuckDB default)."""
    from agentic_warehouse_ops.common.warehouse import get_engine

    return get_engine(profile=os.environ.get("WAREHOUSE_PROFILE", "duckdb"))


def _dbt(*args: str) -> None:
    """Run a dbt command in the mounted dbt project, failing on non-zero exit."""
    command = [
        "dbt",
        *args,
        "--project-dir",
        DBT_DIR,
        "--profiles-dir",
        DBT_DIR,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"dbt {' '.join(args)} failed with exit code {result.returncode}")


@dag(
    dag_id=DAG_ID,
    # Same schedule hour as ingest_raw so ExternalTaskSensor can match runs by
    # logical date without an execution_date_fn mapping.
    schedule="0 0 * * *",
    start_date=START_DATE,
    catchup=True,
    max_active_runs=1,
    default_args=DEFAULT_TASK_ARGS,
    doc_md=__doc__,
    tags=["dbt", "governance", "warehouse"],
)
def transform_and_govern() -> None:
    """Define the transform/govern task graph."""

    wait_for_ingest = ExternalTaskSensor(
        task_id="wait_for_ingest",
        external_dag_id="ingest_raw",
        external_task_id="close_run",
        mode="reschedule",
        timeout=4 * 60 * 60,
        poke_interval=60,
    )

    @task(task_id="dbt_deps")
    def dbt_deps() -> None:
        """Resolve dbt package dependencies."""
        _dbt("deps")

    @task(task_id="dbt_build")
    def dbt_build() -> None:
        """Run dbt build; any test failure raises and fails the DAG."""
        _dbt("build")

    @task(task_id="emit_lineage")
    def emit_lineage(**context: Any) -> dict[str, Any]:
        """Parse manifest/catalog and emit lineage artifacts under the ingest run id."""
        from agentic_warehouse_ops.governance.catalog import build_lineage
        from agentic_warehouse_ops.ingestion.registry import lookup_run

        ingest_run_id = lookup_run(
            _engine(), dag_id="ingest_raw", logical_date=context["logical_date"]
        )
        run_id = ingest_run_id or context["dag_run"].run_id
        manifest_path = Path(DBT_DIR) / "target" / "manifest.json"
        catalog_path = Path(DBT_DIR) / "target" / "catalog.json"
        lineage = build_lineage(
            manifest_path=manifest_path,
            catalog_path=catalog_path if catalog_path.exists() else None,
            run_id=run_id,
            artifacts_dir=ARTIFACTS_DIR,
        )
        return {
            "run_id": run_id,
            "manifest_hash": lineage["manifest_hash"],
            "lineage_path": lineage["artifacts"]["lineage"],
        }

    @task(task_id="apply_grants")
    def apply_grants_task(**context: Any) -> None:
        """Apply role grants generated from the manifest's governance meta."""
        from agentic_warehouse_ops.governance.grants import apply_grants

        manifest = json.loads(Path(DBT_DIR, "target", "manifest.json").read_text())
        dialect = "snowflake" if os.environ.get("WAREHOUSE_PROFILE") == "snowflake" else "duckdb"
        apply_grants(_engine(), manifest, dialect=dialect)

    @task(task_id="stamp_manifest_hash")
    def stamp_manifest_hash_task(lineage: dict[str, Any]) -> None:
        """Stamp the dbt manifest hash onto the ingest pipeline_runs row."""
        from agentic_warehouse_ops.ingestion.registry import stamp_manifest_hash

        stamp_manifest_hash(
            _engine(), run_id=lineage["run_id"], dbt_manifest_hash=lineage["manifest_hash"]
        )

    @task(task_id="snapshot_manifest")
    def snapshot_manifest_task(**context: Any) -> str:
        """Snapshot target/manifest.json and register it in meta.manifest_registry."""
        from agentic_warehouse_ops.common.reproducibility import snapshot_manifest

        digest = snapshot_manifest(
            Path(DBT_DIR, "target", "manifest.json"),
            _engine(),
            artifacts_dir="/opt/airflow/artifacts/manifests",
        )
        print(f"snapshot manifest {digest}")
        return digest

    @task(task_id="embed_tickets")
    def embed_tickets_task() -> None:
        """Embed support-ticket text into the vector index (resumable, hash-keyed)."""
        from agentic_warehouse_ops.agent.index import embed_support_tickets, get_embedder

        result = embed_support_tickets(_engine(), embedder=get_embedder())
        print(result)

    @task(task_id="embed_catalog")
    def embed_catalog_task() -> None:
        """Embed model/column descriptions from the dbt manifest."""
        from agentic_warehouse_ops.agent.index import embed_catalog, get_embedder

        manifest = json.loads(Path(DBT_DIR, "target", "manifest.json").read_text())
        result = embed_catalog(_engine(), manifest, embedder=get_embedder())
        print(result)

    lineage = emit_lineage()
    wait_for_ingest >> dbt_deps() >> dbt_build() >> lineage
    (
        lineage
        >> apply_grants_task()
        >> stamp_manifest_hash_task(lineage)
        >> [
            snapshot_manifest_task(),
            embed_tickets_task(),
            embed_catalog_task(),
        ]
    )


dag = transform_and_govern()
