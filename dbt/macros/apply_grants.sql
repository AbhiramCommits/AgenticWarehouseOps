{% macro _graph_nodes() %}
    {% if graph is mapping %}
        {# Parse/compile time: the graph is an empty dict placeholder. #}
        {{ return(graph.get('nodes', {})) }}
    {% endif %}
    {{ return(graph.nodes) }}
{% endmacro %}

{% macro mart_nodes() %}
    {% set marts = [] %}
    {% for node in _graph_nodes().values() %}
        {% if node.resource_type == 'model'
            and node.package_name == 'warehouse'
            and node.config.materialized == 'table' %}
            {% do marts.append(node) %}
        {% endif %}
    {% endfor %}
    {{ return(marts) }}
{% endmacro %}

{% macro restricted_columns(node) %}
    {% set cols = [] %}
    {% for name, column in node.columns.items() %}
        {% set meta = column.meta or {} %}
        {% if meta.get('classification') == 'restricted' or meta.get('pii') == true %}
            {% do cols.append(name) %}
        {% endif %}
    {% endfor %}
    {{ return(cols) }}
{% endmacro %}

{% macro apply_governance_grants() %}
    {% set marts = mart_nodes() %}
    {% if marts | length == 0 %}
        {{ return('select 1') }}
    {% endif %}
    {% if target.type == 'snowflake' %}
        {% do run_query('CREATE ROLE IF NOT EXISTS analyst_ro') %}
        {% do run_query('CREATE ROLE IF NOT EXISTS engineer_rw') %}
        {% do run_query('CREATE ROLE IF NOT EXISTS pii_reader') %}
        {% for node in marts %}
            {% set restricted = restricted_columns(node) %}
            {% set allowed = [] %}
            {% for name, column in node.columns.items() %}
                {% if name not in restricted %}
                    {% do allowed.append(name) %}
                {% endif %}
            {% endfor %}
            {% set view_sql = 'CREATE OR REPLACE SECURE VIEW analyst_ro.' ~ node.name
                ~ ' AS SELECT ' ~ allowed | join(', ')
                ~ ' FROM ' ~ node.schema ~ '.' ~ node.name %}
            {% do run_query(view_sql) %}
            {% do run_query('GRANT SELECT ON analyst_ro.' ~ node.name ~ ' TO ROLE analyst_ro') %}
        {% endfor %}
        {% for node in marts %}
            {% do run_query('GRANT SELECT ON ALL TABLES IN SCHEMA ' ~ node.schema ~ ' TO ROLE engineer_rw') %}
        {% endfor %}
        {% do run_query('GRANT SELECT ON ALL TABLES IN SCHEMA ' ~ marts[0].schema ~ ' TO ROLE pii_reader') %}
    {% else %}
        {% do run_query('CREATE SCHEMA IF NOT EXISTS analyst_ro') %}
        {% for node in marts %}
            {% set restricted = restricted_columns(node) %}
            {% set allowed = [] %}
            {% for name, column in node.columns.items() %}
                {% if name not in restricted %}
                    {% do allowed.append(name) %}
                {% endif %}
            {% endfor %}
            {% set view_sql = 'CREATE OR REPLACE VIEW analyst_ro.' ~ node.name
                ~ ' AS SELECT ' ~ allowed | join(', ')
                ~ ' FROM ' ~ node.schema ~ '.' ~ node.name %}
            {% do run_query(view_sql) %}
        {% endfor %}
    {% endif %}
    {{ return('select 1') }}
{% endmacro %}
