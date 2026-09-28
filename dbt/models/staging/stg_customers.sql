select
    customer_id::bigint as customer_id,
    trim(full_name) as full_name,
    lower(email) as email,
    phone as phone,
    signup_date::date as signup_date,
    country as country,
    segment as segment
from {{ source('raw', 'customers') }}
