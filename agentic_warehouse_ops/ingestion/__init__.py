"""Ingestion: synthetic source generation and landing into object storage.

Submodules (``generate_sources``, ``seed_minio``, ``s3_to_warehouse``,
``registry``, ``quality``) are imported directly; this package deliberately
stays import-light so runtime environments (e.g. Airflow) need no CLI deps.
"""

__all__: list[str] = []
