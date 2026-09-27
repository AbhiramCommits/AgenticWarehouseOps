"""Data-quality gate for ingestion runs.

The gate fails the DAG run (and therefore marks it ``failed`` in the run
registry) when any loaded partition breaches a rule: fewer rows than the
configured floor, a reject rate above the configured ceiling, or a partition
that had source files but loaded zero rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentic_warehouse_ops.ingestion.s3_to_warehouse import LoadResult

DEFAULT_ROW_FLOORS: dict[str, int] = {
    "customers": 1,
    "orders": 4000,
    "order_items": 8000,
    "products": 1,
    "support_tickets": 50,
}

DEFAULT_REJECT_CEILING: float = 0.01


class QualityGateError(Exception):
    """Raised when a partition breaches a data-quality rule."""


def evaluate_quality_gate(
    loads: Sequence[LoadResult],
    row_floors: Mapping[str, int] | None = None,
    reject_ceiling: float = DEFAULT_REJECT_CEILING,
) -> dict[str, Any]:
    """Evaluate floors, reject ceilings, and emptiness over load results.

    Partitions with no source files are exempt: nothing was expected to land
    for that (table, dt), so zero rows is not a failure.

    Args:
        loads: One :class:`LoadResult` per (table, partition) loaded.
        row_floors: Minimum rows per table; merged over
            :data:`DEFAULT_ROW_FLOORS`.
        reject_ceiling: Maximum allowed reject fraction of
            ``rejected / (loaded + rejected)``.

    Returns:
        ``{"checked_partitions": n, "passed": True}`` when every rule holds.

    Raises:
        QualityGateError: On the first breach, with a descriptive message.
    """
    floors = {**DEFAULT_ROW_FLOORS, **(row_floors or {})}
    checked = 0
    for result in loads:
        if not result.source_keys:
            continue
        checked += 1
        total = result.rows_loaded + result.rows_rejected
        if result.rows_loaded == 0:
            raise QualityGateError(
                f"empty partition: {result.table}/dt={result.dt} had "
                f"{len(result.source_keys)} source file(s) but loaded 0 rows"
            )
        floor = floors.get(result.table, 0)
        if result.rows_loaded < floor:
            raise QualityGateError(
                f"row floor breach: {result.table}/dt={result.dt} loaded "
                f"{result.rows_loaded} rows, floor is {floor}"
            )
        if total and result.rows_rejected / total > reject_ceiling:
            raise QualityGateError(
                f"reject ceiling breach: {result.table}/dt={result.dt} rejected "
                f"{result.rows_rejected}/{total} rows (ceiling {reject_ceiling})"
            )
    return {"checked_partitions": checked, "passed": True}
