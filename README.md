# agentic-warehouse-ops

A data platform that lands raw retail-ops files from object storage into a warehouse, transforms them with dbt, governs PII, and exposes a tool-calling LLM agent that answers questions over the governed marts. Raw parquet files land in an S3-compatible bucket (MinIO locally, AWS S3 in production — both through boto3 only), Airflow copies them into the warehouse raw schema, dbt builds staging and mart models, a governance layer applies PII classification and access policies, and a LangGraph agent with tool-calling answers questions against the governed marts. DuckDB is the default warehouse so the entire platform runs free and offline; Snowflake activates purely through environment variables.

```
               ┌─────────────┐
               │  S3 / MinIO │
               └──────┬──────┘
                      │  boto3 (landing)
               ┌──────▼──────┐
               │   Airflow   │
               └──────┬──────┘
                      │  COPY
               ┌──────▼──────┐
               │  raw schema │
               └──────┬──────┘
                      │  dbt
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────────┐       ┌──────────────────┐
│  staging models   │       │      marts       │
└─────────┬─────────┘       └────────┬─────────┘
          └────────────┬─────────────┘
                       ▼
           ┌──────────────────────┐
           │  governance manifest │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │   LangGraph agent    │
           │  (tool-calling LLM)  │
           └──────────────────────┘
```

## Quickstart

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
make install      # uv sync (creates .venv, locks dependencies)
make seed         # generate synthetic sources under data/seeds/
make minio-up     # start local MinIO + create the raw-landing bucket
make minio-seed   # upload data/seeds/ into the raw-landing bucket
make test         # pytest with coverage
make lint && make typecheck
```

## Layout

- `agentic_warehouse_ops/` — Python package: `ingestion/`, `governance/`, `agent/`, `evals/`, `common/`
- `dags/` — Airflow DAGs
- `dbt/` — dbt project (`warehouse`) with staging and marts
- `data/seeds/` — generated raw files (gitignored)
- `docker/` — local MinIO stack
- `docs/` — project documentation
