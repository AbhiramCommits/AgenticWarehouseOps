"""PII guardrails: the single source of truth for PII column discovery.

Everything downstream (agent guardrails, redaction, exports) must ask
:func:`get_pii_columns` whether a column is PII, rather than duplicating the
list.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

REDACTED_VALUE = "REDACTED"


def get_pii_columns(manifest: dict[str, Any], package_name: str = "warehouse") -> set[str]:
    """Return the set of PII columns as ``"<model>.<column>"``.

    Args:
        manifest: Parsed dbt ``manifest.json`` contents.
        package_name: Only models in this package are scanned.

    Returns:
        Set of ``"model.column"`` strings whose schema.yml meta has
        ``pii: true``.
    """
    pii: set[str] = set()
    for node in manifest.get("nodes", {}).values():
        if node.get("resource_type") != "model" or node.get("package_name") != package_name:
            continue
        for col_name, col in (node.get("columns") or {}).items():
            if (col.get("meta") or {}).get("pii") is True:
                pii.add(f"{node['name']}.{col_name}")
    return pii


def redact(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    model_name: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with every PII column masked.

    Args:
        df: Frame to redact.
        manifest: Parsed dbt ``manifest.json`` contents.
        model_name: When given, only columns that are PII in that model are
            masked; when omitted, a column is masked if it is PII in any model.

    Returns:
        A new frame where PII cell values are replaced with
        :data:`REDACTED_VALUE`; all other values are untouched.
    """
    pii_columns = get_pii_columns(manifest)
    redacted = df.copy()
    for column in df.columns:
        scoped = f"{model_name}.{column}" if model_name else None
        is_pii = scoped in pii_columns or (
            model_name is None and any(item.endswith(f".{column}") for item in pii_columns)
        )
        if is_pii:
            redacted[column] = REDACTED_VALUE
    return redacted
