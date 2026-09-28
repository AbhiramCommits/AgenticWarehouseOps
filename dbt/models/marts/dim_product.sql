select
    sku,
    product_name,
    category,
    supplier,
    list_price
from {{ ref('stg_products') }}
qualify row_number() over (partition by sku order by list_price desc) = 1
