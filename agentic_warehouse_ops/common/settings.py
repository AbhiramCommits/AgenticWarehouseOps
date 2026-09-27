"""Environment-driven settings for the platform.

All runtime configuration comes from environment variables or a ``.env`` file.
Credentials are never hardcoded; see ``.env.example`` at the repository root.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All platform configuration, read from the environment or ``.env``.

    Attributes:
        duckdb_path: Path to the local DuckDB database file (default profile).
        snowflake_account: Snowflake account identifier (e.g. ``xy12345.us-east-1``).
        snowflake_user: Snowflake user (``SNOWFLAKE_USER`` to avoid the shell's ``USER``).
        snowflake_password: Snowflake password.
        snowflake_role: Snowflake role to assume.
        snowflake_warehouse: Snowflake virtual warehouse.
        snowflake_database: Snowflake database.
        raw_bucket: S3 bucket that raw landing files are written to.
        s3_endpoint_url: S3-compatible endpoint; leave empty to use real AWS S3.
        s3_access_key: S3 access key (MinIO default: ``minioadmin``).
        s3_secret_key: S3 secret key (MinIO default: ``minioadmin``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    duckdb_path: str = "data/warehouse.duckdb"

    snowflake_account: str | None = None
    snowflake_user: str | None = None
    snowflake_password: str | None = None
    snowflake_role: str | None = None
    snowflake_warehouse: str | None = None
    snowflake_database: str | None = None

    raw_bucket: str = "raw-landing"
    s3_endpoint_url: str | None = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
