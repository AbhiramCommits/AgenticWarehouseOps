"""Tests for the governance layer: lineage, PII guardrails, grants, meta enforcement."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agentic_warehouse_ops.governance.catalog import (
    build_lineage,
    compute_manifest_hash,
    validate_meta_completeness,
)
from agentic_warehouse_ops.governance.grants import generate_grants
from agentic_warehouse_ops.governance.pii import get_pii_columns, redact

REPO_ROOT = Path(__file__).resolve().parent.parent


def _col(
    name: str, *, pii: bool = False, classification: str = "public", owner: str = "analytics"
) -> dict[str, Any]:
    """Build a manifest column dict with governance meta."""
    return {
        "name": name,
        "meta": {"pii": pii, "classification": classification, "owner": owner},
    }


def _model(
    name: str,
    columns: dict[str, dict[str, Any]],
    *,
    materialized: str = "table",
    schema: str = "marts",
    upstream: list[str] | None = None,
    config_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest model node dict."""
    node: dict[str, Any] = {
        "resource_type": "model",
        "package_name": "warehouse",
        "name": name,
        "schema": schema,
        "unique_id": f"model.warehouse.{name}",
        "config": {"materialized": materialized, **(config_extra or {})},
        "depends_on": {"nodes": upstream or []},
        "columns": columns,
    }
    return node


def _manifest(models: list[dict[str, Any]]) -> dict[str, Any]:
    return {"nodes": {node["unique_id"]: node for node in models}, "sources": {}}


@pytest.fixture
def manifest() -> dict[str, Any]:
    """A small manifest fixture with known PII and restricted columns."""
    return _manifest(
        [
            _model(
                "dim_customer",
                {
                    "customer_id": _col("customer_id", classification="internal"),
                    "full_name": _col("full_name", pii=True, classification="restricted"),
                    "email": _col("email", pii=True, classification="restricted"),
                    "phone": _col("phone", pii=True, classification="restricted"),
                    "country": _col("country"),
                },
                upstream=["model.warehouse.stg_customers"],
            ),
            _model(
                "fct_orders",
                {
                    "order_id": _col("order_id", classification="internal"),
                    "order_total": _col("order_total"),
                },
                upstream=["model.warehouse.stg_orders"],
            ),
            _model(
                "stg_support_tickets",
                {
                    "ticket_id": _col("ticket_id", classification="internal"),
                    "body_text": _col("body_text", pii=True, classification="restricted"),
                },
                materialized="view",
                schema="staging",
                upstream=["source.warehouse.raw.support_tickets"],
            ),
        ]
    )


def test_lineage_artifact_schema(tmp_path: Path, manifest: dict[str, Any]) -> None:
    """build_lineage emits the documented schema plus a Mermaid graph."""
    manifest_path = tmp_path / "manifest.json"
    catalog_path = tmp_path / "catalog.json"
    manifest_path.write_text(json.dumps(manifest))
    catalog_path.write_text(json.dumps({"nodes": {}}))
    artifacts_dir = tmp_path / "artifacts" / "lineage"

    lineage = build_lineage(
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        run_id="run-123",
        artifacts_dir=artifacts_dir,
    )

    assert lineage["run_id"] == "run-123"
    assert lineage["manifest_hash"] == compute_manifest_hash(manifest)
    assert len(lineage["models"]) == 3

    dim = next(model for model in lineage["models"] if model["name"] == "dim_customer")
    assert dim["materialized"] == "table"
    assert dim["schema"] == "marts"
    assert dim["upstream_refs"] == ["model.warehouse.stg_customers"]
    email = next(col for col in dim["columns"] if col["name"] == "email")
    assert email["pii"] is True
    assert email["classification"] == "restricted"

    tickets = next(model for model in lineage["models"] if model["name"] == "stg_support_tickets")
    assert tickets["upstream_sources"] == ["source.warehouse.raw.support_tickets"]

    json_artifact = artifacts_dir / "run-123.json"
    mmd_artifact = artifacts_dir / "run-123.mmd"
    assert json_artifact.is_file()
    assert mmd_artifact.is_file()
    mmd = mmd_artifact.read_text()
    assert "flowchart TD" in mmd
    assert "stg_customers" in mmd and "dim_customer" in mmd


def test_get_pii_columns_returns_exactly_expected(manifest: dict[str, Any]) -> None:
    """The PII set is exactly the columns tagged pii: true in the manifest."""
    assert get_pii_columns(manifest) == {
        "dim_customer.full_name",
        "dim_customer.email",
        "dim_customer.phone",
        "stg_support_tickets.body_text",
    }


def test_redact_masks_pii_and_leaves_rest_untouched(manifest: dict[str, Any]) -> None:
    """PII columns are masked; non-PII values survive verbatim."""
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "email": ["a@example.com", "b@example.com"],
            "country": ["US", "DE"],
        }
    )
    redacted = redact(frame, manifest, model_name="dim_customer")
    assert redacted["email"].tolist() == ["REDACTED", "REDACTED"]
    assert redacted["customer_id"].tolist() == [1, 2]
    assert redacted["country"].tolist() == ["US", "DE"]
    assert frame["email"].tolist() == ["a@example.com", "b@example.com"]


def test_grants_analyst_view_excludes_restricted_columns(manifest: dict[str, Any]) -> None:
    """analyst_ro views must not select restricted columns in either dialect."""
    for dialect in ("duckdb", "snowflake"):
        sql = generate_grants(manifest, dialect)
        if dialect == "snowflake":
            assert "CREATE OR REPLACE SECURE VIEW analyst_ro.dim_customer" in sql
            assert "TO ROLE analyst_ro" in sql
            assert "TO ROLE engineer_rw" in sql
            assert "TO ROLE pii_reader" in sql
        else:
            assert "CREATE OR REPLACE VIEW analyst_ro.dim_customer" in sql
        view_start = sql.index("analyst_ro.dim_customer")
        view_statement = sql[view_start : sql.index(";", view_start)]
        assert "customer_id" in view_statement
        assert "country" in view_statement
        assert "email" not in view_statement
        assert "full_name" not in view_statement
        assert "phone" not in view_statement


def test_meta_completeness_fails_on_untagged_column() -> None:
    """An untagged column is reported; fully tagged manifests are clean."""
    untagged = _manifest(
        [
            _model(
                "fct_orders",
                {
                    "order_id": {"name": "order_id"},
                    "order_total": _col("order_total"),
                },
            )
        ]
    )
    violations = validate_meta_completeness(untagged)
    assert violations == ["fct_orders.order_id: missing ['classification', 'owner', 'pii']"]

    tagged = _manifest(
        [_model("fct_orders", {"order_id": _col("order_id", classification="internal")})]
    )
    assert validate_meta_completeness(tagged) == []


def test_real_dbt_manifest_is_fully_tagged_and_pii_set_matches(
    tmp_path: Path,
) -> None:
    """dbt parse must produce a manifest with every column tagged and a known PII set."""
    env = {**os.environ, "DUCKDB_PATH": str(tmp_path / "parse.duckdb"), "DBT_TARGET": "dev"}
    dbt_bin = shutil.which("dbt")
    assert dbt_bin, "dbt executable not on PATH"
    result = subprocess.run(
        [
            dbt_bin,
            "parse",
            "--project-dir",
            str(REPO_ROOT / "dbt"),
            "--profiles-dir",
            str(REPO_ROOT / "dbt"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    manifest_path = REPO_ROOT / "dbt" / "target" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert validate_meta_completeness(manifest) == []
    assert get_pii_columns(manifest) == {
        "stg_customers.full_name",
        "stg_customers.email",
        "stg_customers.phone",
        "stg_support_tickets.body_text",
        "dim_customer.full_name",
        "dim_customer.email",
        "dim_customer.phone",
    }
