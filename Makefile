.PHONY: install seed minio-up minio-seed lint typecheck test fmt

install:
	uv sync

seed:
	uv run generate-sources

minio-up:
	docker compose -f docker/docker-compose.yml up -d

minio-seed:
	uv run seed-minio

lint:
	uv run ruff check .

typecheck:
	uv run mypy agentic_warehouse_ops

test:
	uv run pytest tests --cov=agentic_warehouse_ops --cov-report=term-missing

fmt:
	uv run ruff format .
