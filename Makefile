.PHONY: install seed minio-up minio-seed airflow-up airflow-down dbt-build dbt-docs eval-warehouse eval lint typecheck test fmt

install:
	uv sync

# Pinned window and clean data so dbt build (the quality gate) and the
# committed eval baseline are reproducible. Pass --defect-rate explicitly to
# seed defect data and watch the tests catch it.
seed:
	uv run generate-sources --start-date 2026-08-28 --defect-rate 0.0

minio-up:
	docker compose -f docker/docker-compose.yml up -d

minio-seed:
	uv run seed-minio

airflow-up:
	chmod -R a+w data
	docker compose -f docker/docker-compose.yml --profile airflow up -d --build --wait

airflow-down:
	docker compose -f docker/docker-compose.yml --profile airflow down

dbt-build:
	uv run dbt build --project-dir dbt --profiles-dir dbt

dbt-docs:
	uv run dbt docs generate --project-dir dbt --profiles-dir dbt

# CI / offline path: load data/seeds straight into DuckDB (no S3/Airflow),
# build the marts, and embed with the deterministic embedder.
eval-warehouse:
	uv run dbt parse --project-dir dbt --profiles-dir dbt
	uv run python -m agentic_warehouse_ops.evals.warehouse
	uv run dbt build --project-dir dbt --profiles-dir dbt

eval:
	uv run awo eval

lint:
	uv run ruff check .

typecheck:
	uv run mypy agentic_warehouse_ops

test:
	uv run pytest tests --cov=agentic_warehouse_ops --cov-report=term-missing

fmt:
	uv run ruff format .
