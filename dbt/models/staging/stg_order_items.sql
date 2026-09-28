select
    order_item_id::bigint as order_item_id,
    order_id::bigint as order_id,
    sku as sku,
    quantity::int as quantity,
    unit_price::numeric(12, 2) as unit_price
from {{ source('raw', 'order_items') }}
