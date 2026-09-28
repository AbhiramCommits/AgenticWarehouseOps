"""Build a local eval warehouse without S3 or Airflow (CI path).

Reads the generated ``data/seeds`` parquet partitions, loads them into a
DuckDB warehouse through the exact same validation/load code as the S3 path,
embeds tickets and catalog descriptions with the deterministic embedder, and
leaves the warehouse ready for ``dbt build`` and ``awo eval``.

Run as: ``python -m agentic_warehouse_ops.evals.warehouse``
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

from agentic_warehouse_ops.agent.embedder import FakeEmbedder
from agentic_warehouse_ops.agent.index import embed_catalog, embed_support_tickets
from agentic_warehouse_ops.common.warehouse import DuckDBEngine, WarehouseEngine
from agentic_warehouse_ops.ingestion.registry import migrate_registry
from agentic_warehouse_ops.ingestion.s3_to_warehouse import load_partition_from_files
from agentic_warehouse_ops.ingestion.schemas import TABLE_SCHEMAS

RUN_ID = "eval-local"


def build_eval_warehouse(
    seeds_dir: str | Path = "data/seeds",
    duckdb_path: str | None = None,
    engine: WarehouseEngine | None = None,
    manifest: dict | None = None,
) -> WarehouseEngine:
    """Load all ``data/seeds`` partitions into a DuckDB warehouse.

    Args:
        seeds_dir: Directory of partitioned parquet files (as written by
            ``generate-sources``).
        duckdb_path: DuckDB file path; defaults to ``DUCKDB_PATH`` or
            ``data/warehouse.duckdb``.
        engine: Optional pre-built engine (mutually exclusive with
            ``duckdb_path``).
        manifest: Optional parsed dbt manifest for catalog embedding; when
            omitted, ``dbt/target/manifest.json`` is loaded.

    Returns:
        The warehouse engine with the raw schema populated.
    """
    resolved_path = duckdb_path or os.environ.get("DUCKDB_PATH", "data/warehouse.duckdb")
    resolved = engine or DuckDBEngine(resolved_path)
    migrate_registry(resolved)
    seeds = Path(seeds_dir)
    if not seeds.is_dir():
        raise FileNotFoundError(
            f"{seeds} missing; run `make seed` first (pinned window, defect_rate=0)"
        )
    for table in TABLE_SCHEMAS:
        partition_dirs = sorted(seeds.joinpath(table).glob("dt=*"))
        if not partition_dirs:
            continue
        for partition_dir in partition_dirs:
            partition_dt = date.fromisoformat(partition_dir.name.split("=")[1])
            files = [
                (str(path.relative_to(seeds)), path.read_bytes())
                for path in sorted(partition_dir.glob("*.parquet"))
            ]
            result = load_partition_from_files(table, partition_dt, resolved, RUN_ID, files)
            print(
                f"{table}/dt={partition_dt}: {result.rows_loaded} loaded,"
                f" {result.rows_rejected} rejected"
            )
    embedder = FakeEmbedder()
    tickets = embed_support_tickets(resolved, embedder=embedder)
    catalog = embed_catalog(resolved, manifest or _load_manifest(), embedder=embedder)
    print(f"embeddings: tickets {tickets.embedded} new, catalog {catalog.embedded} new")
    return resolved


def _load_manifest() -> dict:
    """Load the dbt manifest for catalog embedding (json import kept local)."""
    import json

    manifest_path = Path("dbt/target/manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError("dbt/target/manifest.json missing; run `dbt parse` first")
    return json.loads(manifest_path.read_text())


if __name__ == "__main__":
    try:
        build_eval_warehouse()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
