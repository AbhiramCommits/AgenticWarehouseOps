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

## Airflow

```bash
make airflow-up   # webserver + scheduler + postgres (LocalExecutor); MinIO too
```

- UI: http://localhost:8080 (override with `AIRFLOW_WEBSERVER_PORT`) — login `airflow` / `airflow`
- `dags/`, `agentic_warehouse_ops/`, `dbt/` and `data/` are mounted into the containers, so code changes are picked up live
- DuckDB allows a single writer per database file, so the local dev stack runs tasks serially (`AIRFLOW__CORE__PARALLELISM=1`, override via `AIRFLOW_PARALLELISM` if you point `WAREHOUSE_PROFILE` at Snowflake)
- The `ingest_raw` DAG is daily with `catchup=True` and `max_active_runs=1`; on first start it backfills from its `start_date` (aligned with the default seed window — update `START_DATE` in `dags/ingest_raw.py` if you regenerate seeds for a different window). To backfill explicitly:

```bash
docker compose -f docker/docker-compose.yml --profile airflow exec airflow-scheduler \
  airflow dags backfill -s 2026-08-28 -e 2026-09-27 ingest_raw
```

Every DAG run writes exactly one `meta.pipeline_runs` row and one `meta.load_audit` row per table; loads are idempotent (delete-then-insert per partition in one transaction) and rows failing schema validation land in `raw.<table>_rejects` with a `reject_reason`. The `data_quality_gate` fails the run on row-floor breaches, reject-ceiling breaches, or empty partitions.

## Layout

- `agentic_warehouse_ops/` — Python package: `ingestion/`, `governance/`, `agent/`, `evals/`, `common/`
- `dags/` — Airflow DAGs
- `dbt/` — dbt project (`warehouse`) with staging and marts
- `data/seeds/` — generated raw files (gitignored)
- `docker/` — local MinIO stack
- `docs/` — project documentation
