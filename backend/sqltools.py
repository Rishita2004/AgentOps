"""The only door from the agents to PostgreSQL: named, parameterised, read-only queries."""

import datetime as dt
import decimal
import time

from sqlalchemy import text
from sqlalchemy.engine import Engine

QUERIES: dict[str, str] = {
    "stock_positions": """
        SELECT p.id, p.name, p.price::float AS price, p.reference_price::float AS reference_price, p.cost::float AS cost, p.lead_days,
               i.on_hand,
               COALESCE(SUM(o.qty) FILTER (WHERE o.status = 'open'), 0)::int AS on_order,
               MIN(o.eta) FILTER (WHERE o.status = 'open') AS next_eta,
               COALESCE(bool_or(o.expedited) FILTER (WHERE o.status = 'open'), false) AS expedited
        FROM products p
        JOIN inventory i ON i.product_id = p.id
        LEFT JOIN purchase_orders o ON o.product_id = p.id
        WHERE p.id = ANY(:ids)
        GROUP BY p.id, i.on_hand
        ORDER BY p.id""",
    "sales_history": """
        SELECT day, units, avg_price::float AS avg_price, stockout
        FROM sales_daily
        WHERE product_id = :pid AND day > (SELECT today FROM sim_state WHERE id = 1) - CAST(:days AS int)
        ORDER BY day""",
    "sales_summary": """
        SELECT product_id,
               count(*)::int AS days,
               sum(units)::int AS total_units,
               round(avg(units) FILTER (WHERE NOT stockout)::numeric, 2)::float AS avg_units_in_stock,
               min(avg_price)::float AS min_price,
               max(avg_price)::float AS max_price,
               count(*) FILTER (WHERE stockout)::int AS stockout_days
        FROM sales_daily
        WHERE product_id = ANY(:ids) AND day > (SELECT today FROM sim_state WHERE id = 1) - CAST(:days AS int)
        GROUP BY product_id
        ORDER BY product_id""",
    "open_orders": """
        SELECT product_id, qty, eta, expedited
        FROM purchase_orders
        WHERE status = 'open' AND product_id = ANY(:ids)
        ORDER BY product_id, eta""",
    "market_day": "SELECT today FROM sim_state WHERE id = 1",
}
MAX_ROWS = 2000


def jsonable(value):
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class SqlSession:
    """Runs approved queries on the read-only engine and reports every one to the recorder."""

    def __init__(self, engine: Engine, on_query=None):
        self.engine = engine
        self.on_query = on_query
        self.count = 0

    def run(self, name: str, **params) -> list[dict]:
        if name not in QUERIES:
            raise ValueError("Query is not allowlisted")
        started = time.perf_counter()
        with self.engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text("SET LOCAL statement_timeout = '4s'"))
            rows = [dict(row) for row in conn.execute(text(QUERIES[name]), params).mappings().fetchmany(MAX_ROWS)]
            conn.rollback()
        self.count += 1
        if self.on_query:
            shown = {key: (value if not isinstance(value, list) else f"{len(value)} ids") for key, value in params.items()}
            self.on_query(name, " ".join(QUERIES[name].split()), jsonable(shown), len(rows), round((time.perf_counter() - started) * 1000, 1))
        return rows
