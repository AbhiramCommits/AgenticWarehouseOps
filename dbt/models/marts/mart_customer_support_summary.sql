select
    customer_id,
    count(*) as ticket_count,
    min(created_ts) as first_ticket_ts,
    max(created_ts) as last_ticket_ts,
    count(case when created_ts >= current_date - 30 then 1 end) as open_ticket_count
from {{ ref('stg_support_tickets') }}
group by 1
