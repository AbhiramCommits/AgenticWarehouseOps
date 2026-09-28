select
    i.order_item_id,
    i.order_id,
    i.sku,
    i.quantity,
    i.unit_price,
    (i.quantity * i.unit_price)::numeric(14, 2) as line_total,
    o.order_ts,
    o.channel,
    o.currency,
    o.customer_id,
    p.category
from {{ ref('stg_order_items') }} i
join {{ ref('fct_orders') }} o on i.order_id = o.order_id
left join {{ ref('dim_product') }} p on i.sku = p.sku
