"""Generate and apply role-based grant DDL from governance meta tags.

Roles: ``analyst_ro`` sees the marts minus restricted columns through
generated secure views; ``engineer_rw`` sees all marts tables; ``pii_reader``
sees all marts tables including restricted columns. Snowflake gets real
``GRANT``/``CREATE SECURE VIEW`` DDL; DuckDB (no roles) gets the closest
equivalent: an ``analyst_ro`` schema of restricted-stripped views.
"""

from __future__ import annotations

import os
from typing import Any

from agentic_warehouse_ops.common.warehouse import WarehouseEngine

ROLES = ("analyst_ro", "engineer_rw", "pii_reader")


def _is_restricted(column: dict[str, Any]) -> bool:
    """Return True when a column's meta marks it restricted or PII."""
    meta = column.get("meta") or {}
    return meta.get("classification") == "restricted" or meta.get("pii") is True


def _mart_models(manifest: dict[str, Any], package_name: str = "warehouse") -> list[dict[str, Any]]:
    """Return the materialized-table (mart) models from a manifest."""
    marts: list[dict[str, Any]] = []
    for node in manifest.get("nodes", {}).values():
        if (
            node.get("resource_type") == "model"
            and node.get("package_name") == package_name
            and (node.get("config") or {}).get("materialized") == "table"
        ):
            marts.append(node)
    return sorted(marts, key=lambda node: node["name"])


def _column_lists(node: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Split a model's columns into (allowed, restricted) name lists."""
    allowed: list[str] = []
    restricted: list[str] = []
    for name, column in (node.get("columns") or {}).items():
        if _is_restricted(column):
            restricted.append(name)
        else:
            allowed.append(name)
    return allowed, restricted


def generate_grants(manifest: dict[str, Any], dialect: str = "snowflake") -> str:
    """Generate grant DDL for the marts in a manifest.

    Args:
        manifest: Parsed dbt ``manifest.json`` contents.
        dialect: ``"snowflake"`` (secure views + GRANTs) or ``"duckdb"``
            (restricted-stripped views; roles are informational).

    Returns:
        A ``;``-separated SQL script. The ``analyst_ro`` views never include
        restricted columns.
    """
    statements: list[str] = []
    marts = _mart_models(manifest)
    if dialect == "snowflake":
        statements.append("CREATE ROLE IF NOT EXISTS analyst_ro")
        statements.append("CREATE ROLE IF NOT EXISTS engineer_rw")
        statements.append("CREATE ROLE IF NOT EXISTS pii_reader")
        warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
        if warehouse:
            statements.append(
                "GRANT USAGE ON WAREHOUSE "
                + warehouse
                + " TO ROLE analyst_ro, engineer_rw, pii_reader"
            )
        for node in marts:
            allowed, _ = _column_lists(node)
            statements.append(
                f"CREATE OR REPLACE SECURE VIEW analyst_ro.{node['name']} AS "
                f"SELECT {', '.join(allowed)} FROM {node['schema']}.{node['name']}"
            )
            statements.append(f"GRANT SELECT ON analyst_ro.{node['name']} TO ROLE analyst_ro")
            statements.append(
                f"GRANT SELECT ON ALL TABLES IN SCHEMA {node['schema']} TO ROLE engineer_rw"
            )
            statements.append(f"GRANT SELECT ON {node['schema']}.{node['name']} TO ROLE pii_reader")
    else:
        statements.append("CREATE SCHEMA IF NOT EXISTS analyst_ro")
        statements.append(
            "-- DuckDB has no role-based access control, so analyst_ro is a schema of"
            " restricted-stripped views. engineer_rw and pii_reader are informational."
        )
        for node in marts:
            allowed, _ = _column_lists(node)
            statements.append(
                f"CREATE OR REPLACE VIEW analyst_ro.{node['name']} AS "
                f"SELECT {', '.join(allowed)} FROM {node['schema']}.{node['name']}"
            )
    return ";\n".join(statements) + ";\n"


def apply_grants(
    engine: WarehouseEngine,
    manifest: dict[str, Any],
    dialect: str = "snowflake",
) -> None:
    """Execute the grant DDL against a warehouse engine, statement by statement.

    Args:
        engine: Warehouse engine receiving the DDL.
        manifest: Parsed dbt ``manifest.json`` contents.
        dialect: ``"snowflake"`` or ``"duckdb"``; see :func:`generate_grants`.
    """
    script = generate_grants(manifest, dialect)
    for statement in script.split(";"):
        stripped = statement.strip()
        if not stripped or stripped.startswith("--"):
            continue
        engine.execute(stripped)
