{% test governance_meta(model) %}
    {% set missing = [] %}
    {% if graph is not mapping %}
        {% set node = graph.nodes.get(model.unique_id) %}
        {% if node %}
            {% for col_name, col in node.columns.items() %}
                {% set meta = col.meta or {} %}
                {% if meta.get('pii') is none or meta.get('classification') is none or meta.get('owner') is none %}
                    {% do missing.append(col_name) %}
                {% endif %}
            {% endfor %}
        {% endif %}
    {% endif %}
    {% if missing | length == 0 %}
        select 1 as violation where 1 = 0
    {% else %}
        {% for col in missing %}
            select '{{ model.name }}.{{ col }} lacks governance meta' as violation
            {% if not loop.last %} union all {% endif %}
        {% endfor %}
    {% endif %}
{% endtest %}
