select
    sku as sku,
    trim(product_name) as product_name,
    category as category,
    supplier as supplier,
    list_price::numeric(12, 2) as list_price
from {{ source('raw', 'products') }}
