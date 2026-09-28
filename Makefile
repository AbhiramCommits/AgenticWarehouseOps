.PHONY: install seed minio-up minio-seed airflow-up airflow-down dbt-build dbt-docs lint typecheck test fmt

install:
	uv sync

seed:
	uv run generate-sources

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

lint:
	uv run ruff check .

typecheck:
	uv run mypy agentic_warehouse_ops

test:
	uv run pytest tests --cov=agentic_warehouse_ops --cov-report=term-missing

fmt:
	uv run ruff format .
