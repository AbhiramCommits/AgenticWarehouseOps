with item_totals as (
    select
        order_id,
        sum(line_total) as items_total
    from {{ ref('fct_order_items') }}
    group by 1
)

select o.order_id
from {{ ref('fct_orders') }} o
left join item_totals i on o.order_id = i.order_id
where abs(coalesce(i.items_total, 0) - o.order_total) > 0.01
