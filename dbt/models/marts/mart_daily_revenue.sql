select
    date_trunc('day', order_ts)::date as revenue_date,
    channel,
    category,
    sum(line_total)::numeric(16, 2) as revenue,
    count(distinct order_id) as orders,
    sum(quantity) as units_sold
from {{ ref('fct_order_items') }}
group by 1, 2, 3
