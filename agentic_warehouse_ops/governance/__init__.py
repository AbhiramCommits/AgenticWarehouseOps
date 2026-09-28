"""PII governance: catalog/lineage artifacts, PII guardrails, and role grants.

The dbt schema.yml files are the single source of truth for column metadata
(``pii``, ``classification``, ``owner``). This package turns that metadata
into machine-readable artifacts that the agent's guardrails and the warehouse
grants both consume.
"""

__all__: list[str] = []
