"""Dual-backend warehouse adapters.

All warehouse access goes through :func:`get_engine`, which returns an engine
implementing the :class:`WarehouseEngine` interface. DuckDB is the default so
the entire platform runs locally for free and offline; Snowflake is selected by
setting the ``SNOWFLAKE_*`` environment variables (see ``.env.example``).
Credentials are read exclusively from the environment via
:class:`~agentic_warehouse_ops.common.settings.Settings` and are never
hardcoded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd

from agentic_warehouse_ops.common.settings import Settings

WarehouseProfile = Literal["duckdb", "snowflake"]


class WarehouseEngine(ABC):
    """Common interface implemented by every warehouse backend.

    Implementations share one parameter style: positional ``?`` placeholders,
    passed positionally as a sequence of values.
    """

    @abstractmethod
    def execute(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """Run a single parameterised statement and return the result.

        Args:
            sql: SQL statement; use ``?`` placeholders for parameters.
            params: Optional sequence of parameter values bound to the placeholders.

        Returns:
            The query result as a :class:`pandas.DataFrame`.
        """

    @abstractmethod
    def executemany(self, sql: str, seq_of_params: Sequence[Sequence[Any]]) -> None:
        """Run one parameterised statement once per parameter tuple.

        Args:
            sql: SQL statement; use ``?`` placeholders for parameters.
            seq_of_params: Sequence of parameter tuples, one per execution.
        """

    @abstractmethod
    def ping(self) -> bool:
        """Return ``True`` if the warehouse is reachable and healthy."""

    @abstractmethod
    def begin(self) -> None:
        """Start an explicit transaction."""

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""

    @abstractmethod
    def rollback(self) -> None:
        """Roll back the current transaction."""


class DuckDBEngine(WarehouseEngine):
    """DuckDB-backed engine using a local database file."""

    def __init__(self, path: str) -> None:
        """Open (creating if needed) the DuckDB database at ``path``.

        Args:
            path: Filesystem path to the DuckDB database file.
        """
        parent = Path(path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._connection: duckdb.DuckDBPyConnection = duckdb.connect(path)

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        return self._connection.execute(sql, params).df()

    def executemany(self, sql: str, seq_of_params: Sequence[Sequence[Any]]) -> None:
        self._connection.executemany(sql, seq_of_params)

    def ping(self) -> bool:
        try:
            self.execute("SELECT 1")
        except Exception:
            return False
        return True

    def begin(self) -> None:
        self._connection.begin()

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()


class SnowflakeEngine(WarehouseEngine):
    """Snowflake-backed engine configured purely from environment variables."""

    def __init__(self, settings: Settings) -> None:
        """Connect to Snowflake using the credentials held in ``settings``.

        Args:
            settings: Resolved :class:`Settings`; all ``snowflake_*`` fields
                must be set.
        """
        import snowflake.connector

        self._connection: Any = snowflake.connector.connect(
            account=settings.snowflake_account,
            user=settings.snowflake_user,
            password=settings.snowflake_password,
            role=settings.snowflake_role,
            warehouse=settings.snowflake_warehouse,
            database=settings.snowflake_database,
        )

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        cursor = self._connection.cursor()
        try:
            cursor.execute(sql, params)
            return cursor.fetch_pandas_all()
        finally:
            cursor.close()

    def executemany(self, sql: str, seq_of_params: Sequence[Sequence[Any]]) -> None:
        cursor = self._connection.cursor()
        try:
            cursor.executemany(sql, seq_of_params)
        finally:
            cursor.close()

    def ping(self) -> bool:
        try:
            self.execute("SELECT 1")
        except Exception:
            return False
        return True

    def begin(self) -> None:
        self._connection.cursor().execute("BEGIN")

    def commit(self) -> None:
        self._connection.cursor().execute("COMMIT")

    def rollback(self) -> None:
        self._connection.cursor().execute("ROLLBACK")


def get_engine(
    profile: WarehouseProfile = "duckdb",
    settings: Settings | None = None,
) -> WarehouseEngine:
    """Return a warehouse engine for the requested profile.

    Args:
        profile: ``"duckdb"`` (default, local and offline) or ``"snowflake"``
            (configured from ``SNOWFLAKE_*`` environment variables).
        settings: Optional pre-built settings; defaults to reading the
            environment.

    Returns:
        A ready-to-use :class:`WarehouseEngine` implementation.
    """
    settings = settings or Settings()
    if profile == "duckdb":
        return DuckDBEngine(settings.duckdb_path)
    return SnowflakeEngine(settings)
