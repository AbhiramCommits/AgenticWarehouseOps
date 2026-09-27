"""S3 helpers built exclusively on boto3.

The same code path serves local MinIO and real AWS S3: when
``S3_ENDPOINT_URL`` is set (the local-dev default) the client is pointed at
that endpoint with explicit keys; when it is unset, boto3 falls back to the
standard AWS credential chain, so production S3 requires zero code change.
"""

from __future__ import annotations

from typing import Any

import boto3
from botocore.client import BaseClient

from agentic_warehouse_ops.common.settings import Settings


def get_s3_client(settings: Settings | None = None) -> BaseClient:
    """Build a boto3 S3 client from settings.

    Args:
        settings: Optional pre-built settings; defaults to reading the
            environment.

    Returns:
        A boto3 S3 client. With an ``S3_ENDPOINT_URL`` configured it targets
        that endpoint (e.g. local MinIO); otherwise it uses the default AWS
        credential chain for real S3.
    """
    resolved = settings or Settings()
    kwargs: dict[str, Any] = {}
    if resolved.s3_endpoint_url:
        kwargs["endpoint_url"] = resolved.s3_endpoint_url
        kwargs["aws_access_key_id"] = resolved.s3_access_key
        kwargs["aws_secret_access_key"] = resolved.s3_secret_key
        kwargs["region_name"] = "us-east-1"
    return boto3.client("s3", **kwargs)
