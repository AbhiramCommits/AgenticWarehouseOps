select
    order_id::bigint as order_id,
    customer_id::bigint as customer_id,
    order_ts::timestamp as order_ts,
    status as status,
    channel as channel,
    currency as currency
from {{ source('raw', 'orders') }}
