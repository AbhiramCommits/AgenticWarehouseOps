"""Pydantic row schemas for every raw source table.

Each model defines the exact business columns a partition file must expose;
rows that fail to parse are routed to ``raw.<table>_rejects`` by the loader.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class _RowModel(BaseModel):
    """Base for raw row schemas: forbid unknown columns to catch schema drift."""

    model_config = ConfigDict(extra="forbid")


class CustomerRow(_RowModel):
    """Schema for raw customers rows; deliberately carries PII columns."""

    customer_id: int
    full_name: str
    email: str
    phone: str
    signup_date: datetime
    country: str
    segment: str


class OrderRow(_RowModel):
    """Schema for raw orders rows; ``customer_id`` may be null (defect tolerance)."""

    order_id: int
    customer_id: int | None
    order_ts: datetime
    status: str
    channel: str
    currency: str


class OrderItemRow(_RowModel):
    """Schema for raw order_items rows."""

    order_item_id: int
    order_id: int
    sku: str
    quantity: int
    unit_price: float


class ProductRow(_RowModel):
    """Schema for raw products rows."""

    sku: str
    product_name: str
    category: str
    supplier: str
    list_price: float


class SupportTicketRow(_RowModel):
    """Schema for raw support_tickets rows; ``body_text`` is free text."""

    ticket_id: int
    customer_id: int
    created_ts: datetime
    channel: str
    subject: str
    body_text: str


TABLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "customers": CustomerRow,
    "orders": OrderRow,
    "order_items": OrderItemRow,
    "products": ProductRow,
    "support_tickets": SupportTicketRow,
}

SOURCE_TABLES: tuple[str, ...] = tuple(TABLE_SCHEMAS)
