with orders as (
    select *
    from {{ ref('stg_orders') }}
    qualify row_number() over (partition by order_id order by order_ts desc) = 1
),

order_metrics as (
    select
        order_id,
        sum(quantity * unit_price) as order_total,
        count(*) as item_count
    from {{ ref('stg_order_items') }}
    group by 1
)

select
    o.order_id,
    o.customer_id,
    o.order_ts,
    o.status,
    o.channel,
    o.currency,
    coalesce(m.order_total, 0)::numeric(14, 2) as order_total,
    coalesce(m.item_count, 0) as item_count,
    case
        when o.status in ('pending', 'shipped')
            then datediff('day', o.order_ts, current_date)
        else null
    end as fulfilment_latency_days
from orders o
left join order_metrics m on o.order_id = m.order_id
