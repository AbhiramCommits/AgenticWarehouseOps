select *
from {{ ref('fct_orders') }}
where order_ts > current_timestamp
