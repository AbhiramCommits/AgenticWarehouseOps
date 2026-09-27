"""Tests for the synthetic source-data generator."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from agentic_warehouse_ops.ingestion.generate_sources import (
    TABLES,
    app,
    generate_dataset,
)

START = date(2026, 1, 1)


def _collect_parquet(root: Path) -> dict[str, pd.DataFrame]:
    """Read every parquet file under ``root`` keyed by relative path."""
    return {
        str(path.relative_to(root)): pd.read_parquet(path)
        for path in sorted(root.rglob("*.parquet"))
    }


def _read_table(root: Path, table: str) -> pd.DataFrame:
    """Concatenate all partitions of ``table`` under ``root`` into one frame."""
    return pd.concat(
        [pd.read_parquet(path) for path in sorted((root / table).glob("dt=*/part-0.parquet"))],
        ignore_index=True,
    )


def _defect_summary(orders: pd.DataFrame, customers: pd.DataFrame) -> tuple[int, int, int]:
    """Count null, duplicate, and orphan defects in an orders frame."""
    nulls = int(orders["customer_id"].isna().sum())
    duplicates = len(orders) - int(orders["order_id"].nunique())
    orphans = len(set(orders["customer_id"].dropna()) - set(customers["customer_id"]))
    return nulls, duplicates, orphans


def test_deterministic_under_fixed_seed(tmp_path: Path) -> None:
    """Two runs with the same seed and dates must produce identical files."""
    kwargs = {
        "seed": 123,
        "days": 2,
        "orders_per_day": 100,
        "defect_rate": 0.02,
        "start_date": START,
    }
    run1, run2 = tmp_path / "run1", tmp_path / "run2"
    generate_dataset(out_dir=run1, **kwargs)
    generate_dataset(out_dir=run2, **kwargs)

    files1 = _collect_parquet(run1)
    files2 = _collect_parquet(run2)
    assert list(files1) == list(files2)
    for key in files1:
        assert_frame_equal(files1[key], files2[key])


def test_defect_rate_within_tolerance(tmp_path: Path) -> None:
    """The observed defect fraction must match the requested rate closely."""
    defect_rate = 0.05
    tables = generate_dataset(
        out_dir=tmp_path,
        seed=7,
        days=2,
        orders_per_day=500,
        defect_rate=defect_rate,
        start_date=START,
    )
    orders, customers = tables["orders"], tables["customers"]
    n = len(orders)
    nulls, duplicates, orphans = _defect_summary(orders, customers)

    assert nulls > 0, "expected some null customer_ids"
    assert duplicates > 0, "expected some duplicate order_ids"
    assert orphans > 0, "expected some orphan customer_ids"
    observed_rate = (nulls + duplicates + orphans) / n
    assert abs(observed_rate - defect_rate) <= 0.02


def test_zero_defect_rate_produces_clean_orders(tmp_path: Path) -> None:
    """With a zero defect rate, orders must be fully clean."""
    tables = generate_dataset(
        out_dir=tmp_path,
        seed=11,
        days=2,
        orders_per_day=300,
        defect_rate=0.0,
        start_date=START,
    )
    orders, customers = tables["orders"], tables["customers"]
    nulls, duplicates, orphans = _defect_summary(orders, customers)
    assert (nulls, duplicates, orphans) == (0, 0, 0)


def test_partition_paths_and_layout(tmp_path: Path) -> None:
    """Files must land at <table>/dt=YYYY-MM-DD/part-0.parquet in Hive style."""
    days = 3
    generate_dataset(
        out_dir=tmp_path,
        seed=11,
        days=days,
        orders_per_day=200,
        defect_rate=0.0,
        start_date=START,
    )

    for table in TABLES:
        assert (tmp_path / table).is_dir(), f"{table} directory missing"

    expected_dates = [START + timedelta(days=i) for i in range(days)]
    for table in ("orders", "order_items", "support_tickets"):
        dirs = sorted((tmp_path / table).glob("dt=*"))
        assert [d.name for d in dirs] == [f"dt={d.isoformat()}" for d in expected_dates]

    product_dirs = sorted((tmp_path / "products").glob("dt=*"))
    assert [d.name for d in product_dirs] == [f"dt={START.isoformat()}"]

    customer_dirs = list((tmp_path / "customers").glob("dt=*"))
    assert customer_dirs, "customers should have at least one partition"

    for table in TABLES:
        for entry in (tmp_path / table).iterdir():
            assert entry.is_dir()
            assert entry.name.startswith("dt=") and len(entry.name) == len("dt=YYYY-MM-DD")
            assert (entry / "part-0.parquet").is_file()
            frame = pd.read_parquet(entry / "part-0.parquet")
            assert "dt" not in frame.columns, "partition key must not be stored in files"


def test_defect_rate_out_of_range_raises(tmp_path: Path) -> None:
    """A defect rate outside [0, 1] must be rejected."""
    with pytest.raises(ValueError):
        generate_dataset(out_dir=tmp_path, defect_rate=1.5)


def test_cli_writes_partitions(tmp_path: Path) -> None:
    """The Typer CLI must run end to end and write partitioned output."""
    result = CliRunner().invoke(
        app,
        [
            "--out-dir",
            str(tmp_path),
            "--days",
            "1",
            "--orders-per-day",
            "50",
            "--seed",
            "9",
            "--start-date",
            "2026-03-01",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "orders" / "dt=2026-03-01" / "part-0.parquet").is_file()
    assert (tmp_path / "customers").is_dir()
    assert (tmp_path / "products" / "dt=2026-03-01" / "part-0.parquet").is_file()
