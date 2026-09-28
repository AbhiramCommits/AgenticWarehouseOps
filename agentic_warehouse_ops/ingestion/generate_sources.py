"""Synthetic retail-ops source-data generator.

Generates five tables (``customers``, ``orders``, ``order_items``,
``products``, ``support_tickets``) and writes them as Hive-partitioned parquet
files under ``<out_dir>/<table>/dt=YYYY-MM-DD/part-0.parquet``.

Output is fully deterministic for a fixed ``--seed`` and ``--start-date``.
A controlled fraction of orders (``--defect-rate``) carries data-quality
defects -- null ``customer_id``, duplicate ``order_id``, or orphan
``customer_id`` -- so downstream dbt tests have something real to catch.
The ``customers`` table deliberately contains PII columns (``full_name``,
``email``, ``phone``) for the governance layer, and ``support_tickets.body_text``
is free text for later embedding.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import typer
from numpy.random import Generator

from agentic_warehouse_ops.common.logging import configure_logging

app = typer.Typer(help="Generate synthetic retail-ops sources as partitioned parquet files.")

TABLES: tuple[str, ...] = ("customers", "orders", "order_items", "products", "support_tickets")

FIRST_NAMES: tuple[str, ...] = (
    "Ava",
    "Liam",
    "Noah",
    "Emma",
    "Olivia",
    "Mason",
    "Sophia",
    "Lucas",
    "Isabella",
    "Ethan",
    "Mia",
    "Logan",
    "Amelia",
    "James",
    "Harper",
    "Benjamin",
    "Ella",
    "Henry",
    "Chloe",
    "Jack",
    "Grace",
    "Leo",
    "Zoe",
    "Owen",
    "Lily",
    "Finn",
    "Nora",
    "Kai",
    "Ruby",
    "Max",
)
LAST_NAMES: tuple[str, ...] = (
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Martinez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Harris",
    "Clark",
    "Lewis",
    "Robinson",
    "Walker",
    "Hall",
    "Young",
    "King",
    "Wright",
    "Lopez",
)
COUNTRIES: dict[str, float] = {
    "US": 0.45,
    "UK": 0.15,
    "DE": 0.10,
    "FR": 0.10,
    "CA": 0.10,
    "AU": 0.05,
    "JP": 0.05,
}
COUNTRY_CODES: dict[str, str] = {
    "US": "1",
    "UK": "44",
    "DE": "49",
    "FR": "33",
    "CA": "1",
    "AU": "61",
    "JP": "81",
}
EMAIL_DOMAINS: tuple[str, ...] = ("example.com", "mail.net", "sample.org")
SEGMENTS: tuple[str, ...] = ("premium", "standard", "new")
ORDER_STATUSES: tuple[str, ...] = ("pending", "shipped", "delivered", "cancelled", "returned")
ORDER_CHANNELS: tuple[str, ...] = ("web", "mobile", "store", "partner")
CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP")
PRODUCT_ADJECTIVES: tuple[str, ...] = (
    "Classic",
    "Deluxe",
    "Compact",
    "Premium",
    "Essential",
    "Pro",
    "Eco",
    "Urban",
    "Aurora",
    "Nimbus",
    "Vivid",
    "Quiet",
    "Swift",
    "Sturdy",
    "Bright",
)
PRODUCT_NOUNS: tuple[str, ...] = (
    "Kettle",
    "Desk Lamp",
    "Backpack",
    "Headphones",
    "Blender",
    "Chair",
    "Speaker",
    "Watch",
    "Jacket",
    "Mug",
    "Monitor",
    "Tent",
    "Sneakers",
    "Keyboard",
    "Vacuum",
    "Toaster",
    "Yoga Mat",
    "Drill",
    "Bicycle",
    "Notebook",
)
CATEGORIES: tuple[str, ...] = (
    "electronics",
    "home",
    "apparel",
    "beauty",
    "sports",
    "toys",
    "grocery",
    "automotive",
    "office",
    "garden",
)
SUPPLIERS: tuple[str, ...] = (
    "Acme Supply Co.",
    "GlobalTrade Ltd",
    "Nordic Imports",
    "Pacific Wholesale",
    "Summit Goods",
    "Vertex Trading",
    "BlueBridge Ltd",
    "Eagle Distributors",
    "PrimeSource GmbH",
    "Cascade Partners",
)
TICKET_CHANNELS: tuple[str, ...] = ("web", "email", "phone", "chat")
TICKET_SUBJECTS: tuple[str, ...] = (
    "Delivery delay",
    "Wrong item received",
    "Refund request",
    "Missing package",
    "Damaged item",
    "Account login issue",
    "Promo code not working",
    "Order status question",
    "Change delivery address",
    "Subscription cancellation",
)
TICKET_BODY_TEMPLATES: tuple[str, ...] = (
    "Hi, I placed order {oid} and the {sku} still hasn't arrived. "
    "The tracking page shows it stuck in transit. Can you look into this?",
    "I received my order {oid} today but the {sku} was damaged in the box. "
    "Please advise on a replacement or a refund.",
    "The {sku} in order {oid} is the wrong colour. I need a return label as soon as possible.",
    "My order {oid} was delivered to the wrong address. Please help me recover the package.",
    "I was charged twice for order {oid}. Can someone refund the duplicate charge?",
    "I want to cancel order {oid} before it ships. Please confirm the cancellation.",
    "I can't log into my account since yesterday. Password reset emails never arrive.",
    "The promo code I received doesn't apply at checkout. It should have worked on order {oid}.",
    "How long does standard shipping usually take to {country}?",
    "The {sku} I bought last month is now cheaper. Can I get a price adjustment?",
)


def _table_rngs(seed: int, *names: str) -> dict[str, Generator]:
    """Derive one independent, deterministic NumPy RNG per table name from ``seed``.

    Args:
        seed: Master seed; identical seeds produce identical RNG streams.
        names: One name per RNG to derive.

    Returns:
        A mapping of name to a freshly seeded :class:`numpy.random.Generator`.
    """
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children, strict=True)}


def generate_customers(
    rng: Generator,
    n: int,
    end_date: date,
    window_days: int = 30,
) -> pd.DataFrame:
    """Generate ``n`` customers; ``full_name``/``email``/``phone`` are PII.

    Signup dates spread over ``window_days`` before (and including)
    ``end_date``, so a daily ingest backfill covering the same window loads
    the complete customer dimension.

    Args:
        rng: Seeded RNG for this table.
        n: Number of customers to generate.
        end_date: Latest possible signup date.
        window_days: Number of days signups spread over.

    Returns:
        A DataFrame with columns ``customer_id``, ``full_name``, ``email``,
        ``phone``, ``signup_date``, ``country``, ``segment``.
    """
    first = rng.choice(FIRST_NAMES, size=n)
    last = rng.choice(LAST_NAMES, size=n)
    countries = rng.choice(tuple(COUNTRIES), size=n, p=list(COUNTRIES.values()))
    segments = rng.choice(SEGMENTS, size=n, p=[0.25, 0.55, 0.20])
    signup_days_ago = rng.integers(0, window_days, size=n)
    signup_date = pd.to_datetime(end_date) - pd.to_timedelta(signup_days_ago, unit="D")
    domains = rng.choice(EMAIL_DOMAINS, size=n)
    emails: list[str] = []
    phones: list[str] = []
    for first_name, last_name, country, domain in zip(first, last, countries, domains, strict=True):
        emails.append(f"{first_name.lower()}.{last_name.lower()}{rng.integers(0, 1000)}@{domain}")
        phones.append(f"+{COUNTRY_CODES[country]}{rng.integers(2_000_000_000, 9_999_999_999)}")
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "full_name": [
                f"{first_name} {last_name}"
                for first_name, last_name in zip(first, last, strict=True)
            ],
            "email": emails,
            "phone": phones,
            "signup_date": signup_date,
            "country": countries,
            "segment": segments,
        }
    )


def generate_products(rng: Generator, n: int = 500) -> pd.DataFrame:
    """Generate a static product catalogue of ``n`` SKUs.

    Args:
        rng: Seeded RNG for this table.
        n: Number of products to generate.

    Returns:
        A DataFrame with columns ``sku``, ``product_name``, ``category``,
        ``supplier``, ``list_price``.
    """
    adjectives = rng.choice(PRODUCT_ADJECTIVES, size=n)
    nouns = rng.choice(PRODUCT_NOUNS, size=n)
    return pd.DataFrame(
        {
            "sku": np.array([f"SKU-{i:05d}" for i in range(1, n + 1)]),
            "product_name": [f"{a} {b}" for a, b in zip(adjectives, nouns, strict=True)],
            "category": rng.choice(CATEGORIES, size=n),
            "supplier": rng.choice(SUPPLIERS, size=n),
            "list_price": np.round(rng.uniform(5.0, 500.0, size=n), 2),
        }
    )


def _inject_defects(
    orders: pd.DataFrame,
    rng: Generator,
    defect_rate: float,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Inject controlled data-quality defects into ``orders`` in place.

    A fraction ``defect_rate`` of rows each receives exactly one defect,
    chosen uniformly: a null ``customer_id``, a duplicated ``order_id``, or an
    orphan ``customer_id`` referencing no known customer.

    Args:
        orders: Orders frame to mutate (modified in place).
        rng: Seeded RNG.
        defect_rate: Fraction of rows to defect; must be in ``[0, 1]``.
        customers: Customer frame used to derive out-of-range orphan ids.

    Returns:
        The same (mutated) orders frame.
    """
    n = len(orders)
    n_defects = int(round(defect_rate * n))
    if n_defects == 0:
        return orders
    positions = rng.choice(n, size=n_defects, replace=False)
    kinds = rng.integers(0, 3, size=n_defects)
    orders["customer_id"] = orders["customer_id"].astype("Int64")
    max_customer_id = int(customers["customer_id"].max())
    for position, kind in zip(positions, kinds, strict=True):
        if kind == 0:
            orders.loc[position, "customer_id"] = pd.NA
        elif kind == 1:
            other = (position + 1 + int(rng.integers(1, n))) % n
            orders.loc[position, "order_id"] = orders.loc[other, "order_id"]
        else:
            orders.loc[position, "customer_id"] = int(
                max_customer_id + 1 + rng.integers(0, 1_000_000)
            )
    return orders


def generate_orders(
    rng: Generator,
    customers: pd.DataFrame,
    start_date: date,
    days: int,
    orders_per_day: int,
    defect_rate: float,
) -> pd.DataFrame:
    """Generate one partition per day of orders and inject controlled defects.

    Args:
        rng: Seeded RNG for this table.
        customers: Customer frame orders reference.
        start_date: First day of the order window.
        days: Number of days to generate.
        orders_per_day: Approximate order volume per day (±50 jitter).
        defect_rate: Fraction of orders to defect.

    Returns:
        A DataFrame with columns ``order_id``, ``customer_id``, ``order_ts``,
        ``status``, ``channel``, ``currency`` plus a partition key ``dt``.
    """
    customer_ids = customers["customer_id"].to_numpy()
    segments = customers["segment"].to_numpy()
    weights = np.where(segments == "premium", 4.0, np.where(segments == "standard", 2.0, 1.0))
    weights = weights / weights.sum()
    frames: list[pd.DataFrame] = []
    next_order_id = 1
    for day_idx in range(days):
        count = max(1, int(rng.integers(orders_per_day - 50, orders_per_day + 51)))
        day = start_date + timedelta(days=day_idx)
        order_ts = pd.to_datetime(day) + pd.to_timedelta(rng.random(count) * 86399, unit="s")
        frames.append(
            pd.DataFrame(
                {
                    "order_id": np.arange(next_order_id, next_order_id + count),
                    "customer_id": rng.choice(customer_ids, size=count, p=weights),
                    "order_ts": order_ts,
                    "status": rng.choice(
                        ORDER_STATUSES, size=count, p=[0.10, 0.25, 0.55, 0.05, 0.05]
                    ),
                    "channel": rng.choice(ORDER_CHANNELS, size=count, p=[0.45, 0.30, 0.15, 0.10]),
                    "currency": rng.choice(CURRENCIES, size=count, p=[0.80, 0.12, 0.08]),
                }
            )
        )
        next_order_id += count
    orders = pd.concat(frames, ignore_index=True)
    orders["dt"] = orders["order_ts"].dt.normalize()
    return _inject_defects(orders, rng, defect_rate, customers)


def generate_order_items(
    rng: Generator,
    orders: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Generate 1-5 line items per order, priced against the product catalogue.

    Args:
        rng: Seeded RNG for this table.
        orders: Orders frame; line items inherit the parent order's ``dt``.
        products: Product catalogue frame.

    Returns:
        A DataFrame with columns ``order_item_id``, ``order_id``, ``sku``,
        ``quantity``, ``unit_price`` plus a partition key ``dt``.
    """
    counts = np.clip(1 + rng.poisson(1.2, size=len(orders)), 1, 5)
    total = int(counts.sum())
    order_ids = np.repeat(orders["order_id"].to_numpy(), counts)
    order_dts = np.repeat(orders["dt"].to_numpy(), counts)
    skus = rng.choice(products["sku"].to_numpy(), size=total)
    quantities = rng.integers(1, 6, size=total)
    sku_to_price = dict(zip(products["sku"], products["list_price"], strict=True))
    list_prices = np.array([sku_to_price[sku] for sku in skus])
    unit_price = np.round(list_prices * rng.uniform(0.75, 0.95, size=total), 2)
    return pd.DataFrame(
        {
            "order_item_id": np.arange(1, total + 1),
            "order_id": order_ids,
            "sku": skus,
            "quantity": quantities,
            "unit_price": unit_price,
            "dt": order_dts,
        }
    )


def generate_support_tickets(
    rng: Generator,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    start_date: date,
    days: int,
    tickets_per_day: int,
) -> pd.DataFrame:
    """Generate support tickets whose ``body_text`` is free text for embedding.

    Args:
        rng: Seeded RNG for this table.
        customers: Customer frame ticket authors are drawn from.
        orders: Orders frame referenced inside ticket bodies.
        products: Product catalogue referenced inside ticket bodies.
        start_date: First day of the ticket window.
        days: Number of days to generate.
        tickets_per_day: Mean daily ticket volume (Poisson-sampled).

    Returns:
        A DataFrame with columns ``ticket_id``, ``customer_id``,
        ``created_ts``, ``channel``, ``subject``, ``body_text`` plus a
        partition key ``dt``.
    """
    customer_ids = customers["customer_id"].to_numpy()
    order_ids = orders["order_id"].to_numpy()
    skus = products["sku"].to_numpy()
    countries = tuple(COUNTRIES)
    frames: list[pd.DataFrame] = []
    next_ticket_id = 1
    for day_idx in range(days):
        day = start_date + timedelta(days=day_idx)
        count = max(5, int(rng.poisson(tickets_per_day)))
        created_ts = pd.to_datetime(day) + pd.to_timedelta(rng.random(count) * 86399, unit="s")
        oids = rng.choice(order_ids, size=count)
        body_skus = rng.choice(skus, size=count)
        body_countries = rng.choice(countries, size=count)
        subjects = rng.choice(TICKET_SUBJECTS, size=count)
        templates = rng.choice(TICKET_BODY_TEMPLATES, size=count)
        bodies = [
            template.format(oid=int(oid), sku=sku, country=country)
            for template, oid, sku, country in zip(
                templates, oids, body_skus, body_countries, strict=True
            )
        ]
        frames.append(
            pd.DataFrame(
                {
                    "ticket_id": np.arange(next_ticket_id, next_ticket_id + count),
                    "customer_id": rng.choice(customer_ids, size=count),
                    "created_ts": created_ts,
                    "channel": rng.choice(TICKET_CHANNELS, size=count, p=[0.40, 0.25, 0.20, 0.15]),
                    "subject": subjects,
                    "body_text": bodies,
                }
            )
        )
        next_ticket_id += count
    tickets = pd.concat(frames, ignore_index=True)
    tickets["dt"] = tickets["created_ts"].dt.normalize()
    return tickets


def _write_partitioned(frame: pd.DataFrame, table: str, out_dir: Path) -> None:
    """Write ``frame`` as one parquet partition per unique date in its ``dt`` column.

    Files land at ``<out_dir>/<table>/dt=YYYY-MM-DD/part-0.parquet`` in Hive
    style; the ``dt`` column itself is dropped from the files.

    Args:
        frame: Frame carrying a ``dt`` partition-key column.
        table: Table name used as the partition directory root.
        out_dir: Root directory for all generated data.
    """
    for dt_value, group in frame.groupby(frame["dt"].dt.date):
        partition_dir = out_dir / table / f"dt={dt_value.isoformat()}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["dt"]).to_parquet(partition_dir / "part-0.parquet", index=False)


def generate_dataset(
    out_dir: Path,
    seed: int = 42,
    days: int = 30,
    orders_per_day: int = 5000,
    defect_rate: float = 0.01,
    start_date: date | None = None,
    n_customers: int | None = None,
    n_products: int = 500,
) -> dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset and write partitioned parquet files.

    Args:
        out_dir: Root directory for partitioned parquet output.
        seed: RNG seed; identical seeds and dates produce identical data.
        days: Number of days of history to generate.
        orders_per_day: Approximate number of orders per day.
        defect_rate: Fraction of orders carrying a controlled defect.
        start_date: First day of data; defaults to ``today - days``.
        n_customers: Number of customers; defaults to ``days * orders_per_day // 8``.
        n_products: Number of products in the catalogue.

    Returns:
        The generated DataFrames keyed by table name.

    Raises:
        ValueError: If ``defect_rate`` is outside ``[0, 1]``.
    """
    if not 0.0 <= defect_rate <= 1.0:
        raise ValueError(f"defect_rate must be between 0 and 1, got {defect_rate}")
    start = start_date or (date.today() - timedelta(days=days))
    customers_n = n_customers or max(500, (days * orders_per_day) // 8)
    rngs = _table_rngs(seed, *TABLES)

    customers = generate_customers(
        rngs["customers"],
        customers_n,
        end_date=start + timedelta(days=days - 1),
        window_days=days,
    )
    customers["dt"] = pd.to_datetime(customers["signup_date"]).dt.normalize()

    products = generate_products(rngs["products"], n_products)
    products["dt"] = pd.to_datetime(start)

    orders = generate_orders(rngs["orders"], customers, start, days, orders_per_day, defect_rate)
    order_items = generate_order_items(rngs["order_items"], orders, products)
    tickets = generate_support_tickets(
        rngs["support_tickets"],
        customers,
        orders,
        products,
        start,
        days,
        tickets_per_day=max(5, orders_per_day // 50),
    )

    tables = {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "support_tickets": tickets,
    }
    for table, frame in tables.items():
        _write_partitioned(frame, table, out_dir)
    return tables


@app.command()
def generate(
    out_dir: Path = typer.Option(
        Path("data/seeds"),
        "--out-dir",
        help="Root directory for partitioned parquet output.",
    ),
    seed: int = typer.Option(
        42, "--seed", help="RNG seed; identical seeds produce identical data."
    ),
    days: int = typer.Option(30, "--days", help="Number of days of history to generate."),
    orders_per_day: int = typer.Option(
        5000, "--orders-per-day", help="Approximate number of orders per day."
    ),
    defect_rate: float = typer.Option(
        0.01,
        "--defect-rate",
        min=0.0,
        max=1.0,
        help="Fraction of orders carrying a controlled data-quality defect.",
    ),
    start_date: datetime | None = typer.Option(
        None,
        "--start-date",
        formats=["%Y-%m-%d"],
        help="First day of data (defaults to today minus --days).",
    ),
) -> None:
    """Generate synthetic retail-ops sources and write them under --out-dir."""
    configure_logging()
    log = structlog.get_logger()
    tables = generate_dataset(
        out_dir=out_dir,
        seed=seed,
        days=days,
        orders_per_day=orders_per_day,
        defect_rate=defect_rate,
        start_date=start_date.date() if start_date else None,
    )
    for table, frame in tables.items():
        log.info("generated", table=table, rows=len(frame))


if __name__ == "__main__":
    app()
