select
    customer_id,
    full_name,
    email,
    phone,
    signup_date,
    country,
    segment
from {{ ref('stg_customers') }}
qualify row_number() over (partition by customer_id order by signup_date desc) = 1
