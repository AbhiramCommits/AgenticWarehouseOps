"""Seed the ``raw-landing`` S3 bucket with local synthetic sources.

Uploads every parquet file under ``data/seeds`` (default) into the configured
bucket, preserving relative paths (``<table>/dt=YYYY-MM-DD/part-0.parquet``).
Works against local MinIO and real AWS S3 with zero code change: the client is
built exclusively via boto3 in
:func:`~agentic_warehouse_ops.common.s3.get_s3_client`.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import typer
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from agentic_warehouse_ops.common.logging import configure_logging
from agentic_warehouse_ops.common.s3 import get_s3_client
from agentic_warehouse_ops.common.settings import Settings

app = typer.Typer(help="Upload local source data into the raw-landing S3 bucket.")


def ensure_bucket(client: BaseClient, bucket_name: str) -> None:
    """Create ``bucket_name`` if it does not already exist.

    Args:
        client: boto3 S3 client.
        bucket_name: Bucket to verify (and create if missing).

    Raises:
        ClientError: If the bucket check fails for any reason other than
            the bucket not existing.
    """
    try:
        client.head_bucket(Bucket=bucket_name)
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in {"404", "NoSuchBucket", "NotFound"}:
            client.create_bucket(Bucket=bucket_name)
            structlog.get_logger().info("created_bucket", bucket=bucket_name)
            return
        raise


@app.command()
def upload(
    seeds_dir: Path = typer.Option(
        Path("data/seeds"),
        "--seeds-dir",
        help="Local directory of partitioned parquet files to upload.",
    ),
    bucket: str | None = typer.Option(
        None, "--bucket", help="Destination bucket (defaults to RAW_BUCKET)."
    ),
) -> int:
    """Upload every parquet file under ``seeds_dir`` to S3, preserving paths.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    configure_logging()
    log = structlog.get_logger()
    settings = Settings()
    client = get_s3_client(settings)
    bucket_name = bucket or settings.raw_bucket
    ensure_bucket(client, bucket_name)
    parquet_files = sorted(seeds_dir.rglob("*.parquet"))
    if not parquet_files:
        log.warning("no_parquet_files", seeds_dir=str(seeds_dir))
        return 0
    for path in parquet_files:
        key = path.relative_to(seeds_dir).as_posix()
        client.upload_file(str(path), bucket_name, key)
        log.info("uploaded", key=key, bucket=bucket_name)
    log.info("seed_complete", files=len(parquet_files), bucket=bucket_name)
    return 0


if __name__ == "__main__":
    app()
