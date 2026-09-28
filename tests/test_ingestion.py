"""Tests for the ingestion layer: S3 -> warehouse loads, registry, quality gate, DAGs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from moto import mock_aws

from agentic_warehouse_ops.common.s3 import get_s3_client
from agentic_warehouse_ops.common.settings import Settings
from agentic_warehouse_ops.common.warehouse import DuckDBEngine, WarehouseEngine
from agentic_warehouse_ops.ingestion.generate_sources import generate_dataset
from agentic_warehouse_ops.ingestion.quality import (
    QualityGateError,
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
    list_partitions,
    load_partition,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DT = date(2026, 3, 1)
BUCKET = "raw-landing"
ORDERS_KEY = f"orders/dt={DT.isoformat()}/part-0.parquet"


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[WarehouseEngine]:
    """A fresh DuckDB engine in a temp directory."""
    yield DuckDBEngine(str(tmp_path / "warehouse.duckdb"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings wired for the moto-mocked S3 and a temp DuckDB file."""
    return Settings(
        duckdb_path=str(tmp_path / "warehouse.duckdb"),
        s3_endpoint_url=None,
        raw_bucket=BUCKET,
        s3_access_key="test",
        s3_secret_key="test",
    )


@pytest.fixture
def s3_client(settings: Settings) -> Iterator[Any]:
    """A boto3 S3 client backed by moto with the raw-landing bucket created."""
    with mock_aws():
        client = get_s3_client(settings)
        client.create_bucket(Bucket=BUCKET)
        yield client


def _generate_seeds(tmp_path: Path) -> Path:
    """Generate one day of deterministic synthetic sources into tmp_path."""
    seeds_dir = tmp_path / "seeds"
    generate_dataset(
        out_dir=seeds_dir,
        seed=3,
        days=1,
        orders_per_day=50,
        defect_rate=0.0,
        start_date=DT,
    )
    return seeds_dir


def _upload_seeds(client: Any, seeds_dir: Path) -> None:
    """Upload every generated parquet file to the mocked S3 bucket."""
    for path in sorted(seeds_dir.rglob("*.parquet")):
        key = path.relative_to(seeds_dir).as_posix()
        client.put_object(Bucket=BUCKET, Key=key, Body=path.read_bytes())


def test_list_partitions_returns_expected_keys(
    s3_client: Any, settings: Settings, tmp_path: Path
) -> None:
    """Only parquet keys under the exact dt prefix are returned, sorted."""
    _upload_seeds(s3_client, _generate_seeds(tmp_path))
    keys = list_partitions(BUCKET, "orders", DT, s3_client=s3_client, settings=settings)
    assert keys == [ORDERS_KEY]
    assert (
        list_partitions(BUCKET, "orders", date(1999, 1, 1), s3_client=s3_client, settings=settings)
        == []
    )


def test_load_partition_writes_rows_and_load_columns(
    engine: WarehouseEngine, s3_client: Any, settings: Settings, tmp_path: Path
) -> None:
    """Valid rows land in raw.<table> with dt and all four load columns set."""
    seeds_dir = _generate_seeds(tmp_path)
    _upload_seeds(s3_client, seeds_dir)
    migrate_registry(engine)
    result = load_partition("orders", DT, engine, "run-1", s3_client=s3_client, settings=settings)

    source = pd.read_parquet(seeds_dir / "orders" / f"dt={DT.isoformat()}" / "part-0.parquet")
    assert result.rows_loaded == len(source)
    assert result.rows_rejected == 0
    assert result.source_keys == [ORDERS_KEY]

    loaded = engine.execute("SELECT * FROM raw.orders")
    assert len(loaded) == len(source)
    assert set(loaded["dt"].dt.date) == {DT}
    assert (loaded["_run_id"] == "run-1").all()
    assert (loaded["_source_key"] == ORDERS_KEY).all()
    body = (seeds_dir / "orders" / f"dt={DT.isoformat()}" / "part-0.parquet").read_bytes()
    assert (loaded["_file_hash"] == hashlib.sha256(body).hexdigest()).all()
    assert loaded["_ingested_at"].notna().all()


def test_load_partition_is_idempotent(
    engine: WarehouseEngine, s3_client: Any, settings: Settings, tmp_path: Path
) -> None:
    """Loading the same partition twice must not duplicate rows."""
    _upload_seeds(s3_client, _generate_seeds(tmp_path))
    migrate_registry(engine)
    first = load_partition("orders", DT, engine, "run-1", s3_client=s3_client, settings=settings)
    second = load_partition("orders", DT, engine, "run-1", s3_client=s3_client, settings=settings)
    assert first.rows_loaded == second.rows_loaded
    count = int(engine.execute("SELECT COUNT(*) AS n FROM raw.orders").iloc[0]["n"])
    assert count == first.rows_loaded

    # A different run id must also be idempotent (delete-then-insert).
    load_partition("orders", DT, engine, "run-2", s3_client=s3_client, settings=settings)
    count = int(engine.execute("SELECT COUNT(*) AS n FROM raw.orders").iloc[0]["n"])
    assert count == first.rows_loaded
    assert (engine.execute("SELECT DISTINCT _run_id FROM raw.orders")["_run_id"] == ["run-2"]).all()


def test_rejects_routed_not_dropped(
    engine: WarehouseEngine, s3_client: Any, settings: Settings, tmp_path: Path
) -> None:
    """Invalid rows land in raw.<table>_rejects with a reason; valid rows survive."""
    seeds_dir = _generate_seeds(tmp_path)
    partition_dir = seeds_dir / "orders" / f"dt={DT.isoformat()}"
    bad = pd.DataFrame(
        {
            "order_id": ["not-an-int", "also-bad"],
            "customer_id": [1, 2],
            "order_ts": pd.to_datetime("2026-03-01 10:00:00"),
            "status": ["pending", "pending"],
            "channel": ["web", "web"],
            "currency": ["USD", "USD"],
        }
    )
    bad.to_parquet(partition_dir / "part-1.parquet", index=False)
    _upload_seeds(s3_client, seeds_dir)
    migrate_registry(engine)

    result = load_partition("orders", DT, engine, "run-1", s3_client=s3_client, settings=settings)

    good = pd.read_parquet(partition_dir / "part-0.parquet")
    assert result.rows_loaded == len(good)
    assert result.rows_rejected == 2

    rejects = engine.execute("SELECT * FROM raw.orders_rejects")
    assert len(rejects) == 2
    assert rejects["reject_reason"].str.contains("order_id").all()

    raw = engine.execute("SELECT * FROM raw.orders")
    assert len(raw) == len(good)
    assert set(raw["order_id"]) == set(good["order_id"])


def test_registry_rows_written(engine: WarehouseEngine) -> None:
    """open_run/close_run/record_load_audit persist exactly one registry row."""
    migrate_registry(engine)
    logical_date = datetime(2026, 3, 1, 1, 0, tzinfo=UTC)

    run_id = open_run(engine, dag_id="ingest_raw", logical_date=logical_date)
    assert run_id
    assert open_run(engine, dag_id="ingest_raw", logical_date=logical_date) == run_id

    runs = engine.execute("SELECT * FROM meta.pipeline_runs")
    assert len(runs) == 1
    assert runs.iloc[0]["status"] == "running"
    assert pd.isna(runs.iloc[0]["ended_at"])

    record_load_audit(
        engine,
        run_id=run_id,
        table_name="orders",
        partition_dt=DT,
        rows_loaded=100,
        rows_rejected=2,
        source_keys=[ORDERS_KEY],
        duration_ms=42,
    )
    audits = engine.execute("SELECT * FROM meta.load_audit")
    assert len(audits) == 1
    row = audits.iloc[0]
    assert row["run_id"] == run_id
    assert row["table_name"] == "orders"
    assert row["partition_dt"].date() == DT
    assert row["rows_loaded"] == 100
    assert row["rows_rejected"] == 2
    assert row["duration_ms"] == 42
    assert json.loads(row["source_keys"]) == [ORDERS_KEY]

    close_run(engine, run_id=run_id, status="success")
    runs = engine.execute("SELECT * FROM meta.pipeline_runs")
    assert runs.iloc[0]["status"] == "success"
    assert not pd.isna(runs.iloc[0]["ended_at"])


def test_quality_gate_failure_raises() -> None:
    """Floors, reject ceilings, and empty partitions each raise; healthy loads pass."""
    good = LoadResult("orders", DT, 100, 1, ["k"], 10)
    assert (
        evaluate_quality_gate([good], row_floors={"orders": 50}, reject_ceiling=0.05)["passed"]
        is True
    )

    with pytest.raises(QualityGateError, match="floor"):
        evaluate_quality_gate(
            [LoadResult("orders", DT, 3, 0, ["k"], 10)], row_floors={"orders": 50}
        )
    with pytest.raises(QualityGateError, match="reject ceiling"):
        evaluate_quality_gate(
            [LoadResult("orders", DT, 100, 50, ["k"], 10)],
            row_floors={"orders": 1},
            reject_ceiling=0.05,
        )
    with pytest.raises(QualityGateError, match="empty partition"):
        evaluate_quality_gate([LoadResult("orders", DT, 0, 0, ["k"], 10)])

    # Partitions with no source files are exempt from every rule.
    assert (
        evaluate_quality_gate([LoadResult("products", DT, 0, 0, [], 5)])["checked_partitions"] == 0
    )


class _TaskProxy:
    """Stand-in for a TaskFlow operator when airflow is not installed."""

    def __init__(self, name: str = "task") -> None:
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> _TaskProxy:
        return _TaskProxy()

    def partial(self, *args: Any, **kwargs: Any) -> _TaskProxy:
        return _TaskProxy()

    def expand(self, *args: Any, **kwargs: Any) -> _TaskProxy:
        return _TaskProxy()

    def set_downstream(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __rshift__(self, other: Any) -> Any:
        return other


class _SensorStub:
    """Stand-in for airflow sensors (e.g. ExternalTaskSensor)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __rshift__(self, other: Any) -> Any:
        return other


def _stub_airflow_modules() -> None:
    """Install minimal airflow stand-ins so DAG files import without airflow.

    The real apache-airflow distribution cannot coexist with this project's
    pandas>=2.2 pin, so the DAG-integrity test defines the task graph through
    proxy objects instead of executing tasks.
    """
    if "airflow" in sys.modules:
        return

    def task_stub(*args: Any, **kwargs: Any) -> Any:
        def decorator(fn: Any) -> Any:
            proxy = _TaskProxy(fn.__name__)
            proxy.__doc__ = fn.__doc__
            return proxy

        return decorator

    def dag_stub(*args: Any, **kwargs: Any) -> Any:
        def decorator(fn: Any) -> Any:
            return fn

        return decorator

    airflow = ModuleType("airflow")
    decorators = ModuleType("airflow.decorators")
    utils = ModuleType("airflow.utils")
    trigger_rule = ModuleType("airflow.utils.trigger_rule")
    sensors = ModuleType("airflow.sensors")
    external_task = ModuleType("airflow.sensors.external_task")
    decorators.task = task_stub
    decorators.dag = dag_stub
    trigger_rule.TriggerRule = SimpleNamespace(ALL_DONE="all_done")
    external_task.ExternalTaskSensor = _SensorStub
    sys.modules["airflow"] = airflow
    sys.modules["airflow.decorators"] = decorators
    sys.modules["airflow.utils"] = utils
    sys.modules["airflow.utils.trigger_rule"] = trigger_rule
    sys.modules["airflow.sensors"] = sensors
    sys.modules["airflow.sensors.external_task"] = external_task


def test_all_dags_import_cleanly() -> None:
    """Every DAG file must import and define its task graph without errors."""
    _stub_airflow_modules()
    dag_files = sorted((REPO_ROOT / "dags").glob("*.py"))
    assert dag_files, "no DAG files found"
    for dag_file in dag_files:
        spec = importlib.util.spec_from_file_location(f"dag_{dag_file.stem}", dag_file)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"failed to import {dag_file.name}: {exc}")
