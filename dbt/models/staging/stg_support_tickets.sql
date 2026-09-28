select
    ticket_id::bigint as ticket_id,
    customer_id::bigint as customer_id,
    created_ts::timestamp as created_ts,
    channel as channel,
    subject as subject,
    body_text as body_text
from {{ source('raw', 'support_tickets') }}
