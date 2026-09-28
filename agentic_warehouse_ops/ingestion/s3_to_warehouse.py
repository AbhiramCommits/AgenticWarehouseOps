"""Load S3 parquet partitions into the warehouse raw schema, idempotently.

``load_partition`` reads every parquet file under an S3 prefix
``<table>/dt=<date>/``, validates each row against the table's pydantic schema,
and writes valid rows to ``raw.<table>`` with the load columns ``_ingested_at``,
``_source_key``, ``_run_id`` and ``_file_hash``. Rows failing validation are
routed to ``raw.<table>_rejects`` with a ``reject_reason`` instead of aborting
the load. Reloading the same (table, dt, run) never duplicates rows: the
partition is deleted and re-inserted inside a single transaction.
"""

from __future__ import annotations

import hashlib
import io
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
from botocore.client import BaseClient
from pydantic import BaseModel, ValidationError

from agentic_warehouse_ops.common.s3 import get_s3_client
from agentic_warehouse_ops.common.settings import Settings
from agentic_warehouse_ops.common.warehouse import WarehouseEngine
from agentic_warehouse_ops.ingestion.schemas import SOURCE_TABLES, TABLE_SCHEMAS

LOAD_COLUMNS: tuple[str, ...] = ("_ingested_at", "_source_key", "_run_id", "_file_hash")


@dataclass(frozen=True)
class LoadResult:
    """Outcome of a single partition load."""

    table: str
    dt: date
    rows_loaded: int
    rows_rejected: int
    source_keys: list[str]
    duration_ms: int


def _utcnow() -> datetime:
    """Return the current UTC time as a naive datetime for TIMESTAMP columns."""
    return datetime.now(UTC).replace(tzinfo=None)


def _insert_sql(table: str, model: type[BaseModel], rejects: bool = False) -> str:
    """Build the INSERT statement for ``raw.<table>`` (or its rejects twin)."""
    target = f"{table}_rejects" if rejects else table
    columns = list(model.model_fields) + ["dt", *LOAD_COLUMNS]
    if rejects:
        columns.append("reject_reason")
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO raw.{target} ({', '.join(columns)}) VALUES ({placeholders})"


_VALID_INSERT_SQL = {table: _insert_sql(table, model) for table, model in TABLE_SCHEMAS.items()}
_REJECT_INSERT_SQL = {
    table: _insert_sql(table, model, rejects=True) for table, model in TABLE_SCHEMAS.items()
}


def _clean_value(value: Any) -> Any:
    """Convert a pandas scalar into a plain Python value, mapping nulls to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stringify(value: Any) -> Any:
    """Best-effort string form of a value for rejects tables (None stays None)."""
    if value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def _format_errors(exc: ValidationError) -> str:
    """Render the first validation errors as a compact, truncated string."""
    parts = []
    for error in exc.errors()[:10]:
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)[:2000]


def list_partitions(
    bucket: str,
    table: str,
    dt: date,
    s3_client: BaseClient | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Return the S3 keys for every parquet file in ``<table>/dt=<date>/``.

    Args:
        bucket: S3 bucket holding the landing files.
        table: Source table name.
        dt: Partition date.
        s3_client: Optional boto3 S3 client; defaults to one built from settings.
        settings: Optional settings; defaults to reading the environment.

    Returns:
        Sorted S3 object keys ending in ``.parquet`` under the partition prefix.
    """
    resolved = settings or Settings()
    client = s3_client or get_s3_client(resolved)
    prefix = f"{table}/dt={dt.isoformat()}/"
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                keys.append(key)
    return sorted(keys)


def load_partition(
    table: str,
    dt: date,
    engine: WarehouseEngine,
    run_id: str,
    *,
    bucket: str | None = None,
    s3_client: BaseClient | None = None,
    settings: Settings | None = None,
) -> LoadResult:
    """Load one (table, dt) partition from S3 into ``raw.<table>``.

    Idempotent: existing rows for the partition (valid and rejected) are
    deleted and re-inserted inside one transaction, so re-running the same
    ``(table, dt, run_id)`` never duplicates rows.

    Args:
        table: Source table name; must be a key of ``TABLE_SCHEMAS``.
        dt: Partition date to load.
        engine: Warehouse engine (tables must already exist via
            :func:`~agentic_warehouse_ops.ingestion.registry.migrate_registry`).
        run_id: Pipeline run id stamped into ``_run_id``.
        bucket: Optional S3 bucket; defaults to ``Settings.raw_bucket``.
        s3_client: Optional boto3 S3 client; defaults to one built from settings.
        settings: Optional settings; defaults to reading the environment.

    Returns:
        A :class:`LoadResult` summarising rows loaded, rows rejected, and the
        source keys read.

    Raises:
        KeyError: If ``table`` has no registered schema.
        Exception: On load failure; the transaction is rolled back so the
            previous partition contents are preserved.
    """
    started = time.monotonic()
    resolved = settings or Settings()
    client = s3_client or get_s3_client(resolved)
    bucket_name = bucket or resolved.raw_bucket
    keys = list_partitions(bucket_name, table, dt, s3_client=client, settings=resolved)
    if not keys:
        return LoadResult(table, dt, 0, 0, [], int((time.monotonic() - started) * 1000))
    files: list[tuple[str, bytes]] = []
    for key in keys:
        files.append((key, client.get_object(Bucket=bucket_name, Key=key)["Body"].read()))
    return load_partition_from_files(table, dt, engine, run_id, files)


def load_partition_from_files(
    table: str,
    dt: date,
    engine: WarehouseEngine,
    run_id: str,
    files: Sequence[tuple[str, bytes]],
) -> LoadResult:
    """Load one (table, dt) partition from in-memory parquet files.

    Identical semantics to :func:`load_partition` (idempotent delete-then-
    insert, validation, rejects) but reading ``(source_key, parquet_bytes)``
    tuples instead of S3 — used by CI and the eval harness, which have no
    object store.

    Args:
        table: Source table name; must be a key of ``TABLE_SCHEMAS``.
        dt: Partition date to load.
        engine: Warehouse engine (raw tables must exist via
            :func:`~agentic_warehouse_ops.ingestion.registry.migrate_registry`).
        run_id: Pipeline run id stamped into ``_run_id``.
        files: ``(source_key, parquet_bytes)`` pairs for every partition file.

    Returns:
        A :class:`LoadResult` summarising rows loaded, rows rejected, and the
        source keys read.

    Raises:
        KeyError: If ``table`` has no registered schema.
        Exception: On load failure; the transaction is rolled back.
    """
    started = time.monotonic()
    if not files:
        return LoadResult(table, dt, 0, 0, [], 0)
    model = TABLE_SCHEMAS[table]
    fields = list(model.model_fields)
    ingested_at = _utcnow()
    valid_rows: list[tuple[Any, ...]] = []
    reject_rows: list[tuple[Any, ...]] = []

    engine.begin()
    try:
        engine.execute(f"DELETE FROM raw.{table} WHERE dt = ?", [dt])
        engine.execute(f"DELETE FROM raw.{table}_rejects WHERE dt = ?", [dt])
        for key, body in files:
            file_hash = hashlib.sha256(body).hexdigest()
            frame = pd.read_parquet(io.BytesIO(body))
            for record in frame.to_dict("records"):
                cleaned: dict[str, Any] = {
                    str(name): _clean_value(value) for name, value in record.items()
                }
                try:
                    instance = model(**cleaned)
                except ValidationError as exc:
                    reject_rows.append(
                        (
                            *(_stringify(cleaned.get(name)) for name in fields),
                            dt,
                            ingested_at,
                            key,
                            run_id,
                            file_hash,
                            _format_errors(exc),
                        )
                    )
                else:
                    valid_rows.append(
                        (
                            *(instance.model_dump()[name] for name in fields),
                            dt,
                            ingested_at,
                            key,
                            run_id,
                            file_hash,
                        )
                    )
        if valid_rows:
            engine.executemany(_VALID_INSERT_SQL[table], valid_rows)
        if reject_rows:
            engine.executemany(_REJECT_INSERT_SQL[table], reject_rows)
        engine.commit()
    except Exception:
        engine.rollback()
        raise
    keys = [key for key, _ in files]
    return LoadResult(
        table, dt, len(valid_rows), len(reject_rows), keys, int((time.monotonic() - started) * 1000)
    )


__all__ = [
    "LOAD_COLUMNS",
    "LoadResult",
    "SOURCE_TABLES",
    "list_partitions",
    "load_partition",
    "load_partition_from_files",
]
