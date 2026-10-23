"""PostgreSQL schema, seeding and the market simulator that turns agent decisions into measurable outcomes."""

import datetime as dt
import logging
import math
import os
from contextlib import contextmanager

import numpy as np
import psycopg
from psycopg import sql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS products(
  id text PRIMARY KEY, name text NOT NULL, category text NOT NULL,
  price numeric(12,2) NOT NULL CHECK (price > 0), reference_price numeric(12,2) NOT NULL CHECK (reference_price > 0),
  cost numeric(12,2) NOT NULL CHECK (cost > 0), lead_days int NOT NULL CHECK (lead_days BETWEEN 1 AND 60),
  image text NOT NULL DEFAULT '', updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS inventory(
  product_id text PRIMARY KEY REFERENCES products(id), on_hand int NOT NULL CHECK (on_hand >= 0), updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS purchase_orders(
  id bigserial PRIMARY KEY, product_id text NOT NULL REFERENCES products(id), qty int NOT NULL CHECK (qty > 0),
  placed_on date NOT NULL, eta date NOT NULL, status text NOT NULL DEFAULT 'open' CHECK (status IN ('open','received','cancelled')),
  expedited boolean NOT NULL DEFAULT false, source text NOT NULL DEFAULT 'operator');
CREATE INDEX IF NOT EXISTS po_open ON purchase_orders(product_id) WHERE status = 'open';
CREATE TABLE IF NOT EXISTS sales_daily(
  product_id text NOT NULL REFERENCES products(id), day date NOT NULL, units int NOT NULL CHECK (units >= 0),
  avg_price numeric(12,2) NOT NULL, stockout boolean NOT NULL, PRIMARY KEY (product_id, day));
CREATE TABLE IF NOT EXISTS sim_state(id int PRIMARY KEY CHECK (id = 1), today date NOT NULL);
CREATE TABLE IF NOT EXISTS sim_truth(product_id text PRIMARY KEY REFERENCES products(id), true_elasticity double precision NOT NULL, base_units double precision NOT NULL);
CREATE TABLE IF NOT EXISTS settings(key text PRIMARY KEY, value jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS agent_runs(
  id uuid PRIMARY KEY, scenario text NOT NULL, goal text NOT NULL, status text NOT NULL, sim_day date NOT NULL,
  execution_mode text NOT NULL DEFAULT 'review', started_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz, error text);
CREATE TABLE IF NOT EXISTS agent_events(
  id bigserial PRIMARY KEY, run_id uuid NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE, seq int NOT NULL,
  agent text NOT NULL, kind text NOT NULL, title text NOT NULL, detail jsonb NOT NULL DEFAULT '{}', ts timestamptz NOT NULL DEFAULT now(),
  UNIQUE (run_id, seq));
CREATE TABLE IF NOT EXISTS agent_metrics(
  run_id uuid NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE, agent text NOT NULL, provider text NOT NULL,
  model_calls int NOT NULL, tool_calls int NOT NULL, sql_queries int NOT NULL, duration_ms double precision NOT NULL,
  completed_locally int NOT NULL DEFAULT 0, rationale text NOT NULL DEFAULT '', PRIMARY KEY (run_id, agent));
CREATE TABLE IF NOT EXISTS forecasts(
  run_id uuid NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE, product_id text NOT NULL, target_day date NOT NULL,
  mean double precision NOT NULL, lower double precision NOT NULL, upper double precision NOT NULL, baseline double precision NOT NULL,
  PRIMARY KEY (run_id, product_id, target_day));
CREATE TABLE IF NOT EXISTS forecast_quality(
  run_id uuid NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE, product_id text NOT NULL,
  wape double precision NOT NULL, baseline_wape double precision NOT NULL, days_used int NOT NULL, PRIMARY KEY (run_id, product_id));
CREATE TABLE IF NOT EXISTS stockout_predictions(
  run_id uuid NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE, product_id text NOT NULL, made_on date NOT NULL, window_days int NOT NULL,
  p_stockout double precision NOT NULL, p_before_resupply double precision NOT NULL, expected_day double precision NOT NULL,
  days_cover double precision NOT NULL, on_hand int NOT NULL, incoming int NOT NULL, PRIMARY KEY (run_id, product_id));
CREATE TABLE IF NOT EXISTS elasticity_estimates(
  run_id uuid NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE, product_id text NOT NULL, elasticity double precision, std_err double precision,
  n_obs int NOT NULL, r2 double precision, usable boolean NOT NULL, reason text NOT NULL, curve jsonb NOT NULL DEFAULT '[]', PRIMARY KEY (run_id, product_id));
CREATE TABLE IF NOT EXISTS decisions(
  id uuid PRIMARY KEY, run_id uuid NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE, product_id text NOT NULL REFERENCES products(id),
  agent text NOT NULL, kind text NOT NULL CHECK (kind IN ('reorder','price','expedite')), before_value double precision, after_value double precision,
  quantity int, reason text NOT NULL, evidence jsonb NOT NULL DEFAULT '{}', needs_approval boolean NOT NULL,
  status text NOT NULL CHECK (status IN ('pending','applied','rejected','stale')), predicted_units double precision,
  created_at timestamptz NOT NULL DEFAULT now(), decided_at timestamptz, applied_on date);
"""

READ_ONLY_TABLES = ("products", "inventory", "purchase_orders", "sales_daily", "sim_state")

# id, name, category, list price (INR), base units/day at list price, true elasticity (simulator ground truth), lead days
CATALOG = [
    ("FK-1001", "Apple iPhone 13 Pro", "Smartphones", 89900, 4.0, -2.3, 7, "smartphones/iphone-13-pro"),
    ("FK-1002", "Samsung Galaxy S10", "Smartphones", 31999, 9.0, -1.7, 7, "smartphones/samsung-galaxy-s10"),
    ("FK-1003", "Apple AirPods", "Audio", 10999, 22.0, -2.0, 5, "mobile-accessories/apple-airpods"),
    ("FK-1004", "Samsung Galaxy S8", "Smartphones", 18999, 8.0, -1.4, 7, "smartphones/samsung-galaxy-s8"),
    ("FK-1005", "ASUS Zenbook Pro Duo", "Laptops", 149990, 2.5, -1.2, 10, "laptops/asus-zenbook-pro-dual-screen-laptop"),
    ("FK-1006", "Apple MacBook Pro 14", "Laptops", 159900, 3.0, -1.1, 10, "laptops/apple-macbook-pro-14-inch-space-grey"),
    ("FK-1007", "Apple Watch Series 4", "Wearables", 19900, 10.0, -1.9, 6, "mobile-accessories/apple-watch-series-4-gold"),
    ("FK-1008", "Apple iPhone X", "Smartphones", 24999, 12.0, -1.5, 7, "smartphones/iphone-x"),
]
WEEKDAY = [0.92, 0.95, 1.0, 1.05, 1.15, 1.22, 0.78]
HISTORY_DAYS = 120
OPERATOR_REORDER_DAYS = 9.0
OPERATOR_ORDER_DAYS = 16


def _normalize_pg_url(url: str) -> str:
    """Convert Render/Heroku postgres:// URLs to SQLAlchemy+psycopg format."""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    if ".onrender.com" in url and "sslmode" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


class Database:
    def __init__(self, url: str, ro_url: str | None = None):
        self.url = url
        self.engine: Engine = create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True)
        self.ro_engine: Engine | None = create_engine(ro_url, pool_size=3, max_overflow=2, pool_pre_ping=True) if ro_url else None

    @classmethod
    def from_env(cls) -> "Database":
        url = os.getenv("DATABASE_URL", "")
        url = _normalize_pg_url(url)
        if not url.startswith("postgresql"):
            raise RuntimeError("DATABASE_URL must point to PostgreSQL (postgresql+psycopg://...)")
        ro_url = os.getenv("AGENT_DB_URL") or None
        # Auto-derive AGENT_DB_URL from DATABASE_URL + AGENT_DB_PASSWORD when not set
        agent_pw = os.getenv("AGENT_DB_PASSWORD", "")
        if (not ro_url or ro_url == "placeholder") and agent_pw and "://" in url:
            from urllib.parse import urlparse, urlunparse, quote
            parsed = urlparse(url)
            ro_url = urlunparse(parsed._replace(netloc=f"agent_ro:{quote(agent_pw)}@{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")))
        if ro_url:
            ro_url = _normalize_pg_url(ro_url)
        return cls(url, ro_url)

    @property
    def psycopg_url(self) -> str:
        return self.url.replace("postgresql+psycopg://", "postgresql://", 1)

    def init_schema(self, agent_password: str | None = None) -> None:
        role = "agent_ro"
        with psycopg.connect(self.psycopg_url, autocommit=True) as conn:
            conn.execute(SCHEMA)
            if agent_password:
                try:
                    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
                    if exists:
                        conn.execute(sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(agent_password)))
                    else:
                        conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(agent_password)))
                    db_name = conn.execute("SELECT current_database()").fetchone()[0]
                    conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(db_name), sql.Identifier(role)))
                    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
                    conn.execute(sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {}").format(sql.Identifier(role)))
                    for table in READ_ONLY_TABLES:
                        conn.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(sql.Identifier(table), sql.Identifier(role)))
                except psycopg.errors.InsufficientPrivilege:
                    logger.warning("Cannot create/alter agent_ro role (managed DB). Agents will use the main database connection.")
                    self.ro_engine = None

    @contextmanager
    def begin(self):
        with self.engine.begin() as conn:
            yield conn

    def setting(self, key: str, default):
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT value FROM settings WHERE key = :k"), {"k": key}).fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value) -> None:
        import json
        with self.begin() as conn:
            conn.execute(text("INSERT INTO settings(key, value) VALUES (:k, CAST(:v AS jsonb)) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"), {"k": key, "v": json.dumps(value)})

    def today(self) -> dt.date:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT today FROM sim_state WHERE id = 1")).scalar_one()


def _rate(base: float, elasticity: float, price: float, reference: float, day: dt.date, index: int) -> float:
    return base * (price / reference) ** elasticity * WEEKDAY[day.weekday()] * (1 + 0.0015 * index)


def _price_path(rng: np.random.Generator, reference: float, days: int) -> list[float]:
    path, multiplier = [], 1.0
    while len(path) < days:
        multiplier = float(np.clip(rng.choice([0.9, 0.94, 0.97, 1.0, 1.03, 1.06, 1.09]), 0.88, 1.1))
        path += [round(reference * multiplier, 2)] * int(rng.integers(8, 15))
    path = path[:days]
    path[-1] = round(reference, 2)
    return path


def _simulate(conn, first_day: dt.date, days: int, rng: np.random.Generator, price_paths: dict | None = None, first_index: int = 0) -> None:
    products = conn.execute(text("SELECT p.id, p.price, p.reference_price, p.cost, p.lead_days, i.on_hand, t.true_elasticity, t.base_units FROM products p JOIN inventory i ON i.product_id = p.id JOIN sim_truth t ON t.product_id = p.id")).mappings().all()
    open_orders = [dict(row) for row in conn.execute(text("SELECT id, product_id, qty, eta, expedited FROM purchase_orders WHERE status = 'open'")).mappings()]
    state = {p["id"]: {"on_hand": int(p["on_hand"]), "price": float(p["price"]), "p": p} for p in products}
    sales, new_orders = [], []
    for step in range(days):
        day = first_day + dt.timedelta(days=step)
        for sku, s in state.items():
            p = s["p"]
            for order in [o for o in open_orders if o["product_id"] == sku and o["eta"] <= day and not o.get("done")]:
                s["on_hand"] += order["qty"]
                order["done"] = True
            price = price_paths[sku][step] if price_paths else s["price"]
            demand = int(rng.poisson(_rate(p["base_units"], p["true_elasticity"], price, float(p["reference_price"]), day, first_index + step)))
            sold = min(demand, s["on_hand"])
            sales.append({"p": sku, "d": day, "u": sold, "pr": round(price, 2), "s": demand > s["on_hand"]})
            s["on_hand"] -= sold
            incoming = sum(o["qty"] for o in open_orders if o["product_id"] == sku and not o.get("done"))
            if s["on_hand"] + incoming < OPERATOR_REORDER_DAYS * p["base_units"]:
                qty = max(1, math.ceil(p["base_units"] * OPERATOR_ORDER_DAYS))
                order = {"product_id": sku, "qty": qty, "eta": day + dt.timedelta(days=int(p["lead_days"])), "placed": day, "new": True, "expedited": False}
                open_orders.append(order)
                new_orders.append(order)
    conn.execute(text("INSERT INTO sales_daily(product_id, day, units, avg_price, stockout) VALUES (:p, :d, :u, :pr, :s) ON CONFLICT DO NOTHING"), sales)
    for order in [o for o in open_orders if o.get("done") and not o.get("new")]:
        conn.execute(text("UPDATE purchase_orders SET status = 'received' WHERE id = :i"), {"i": order["id"]})
    for order in new_orders:
        conn.execute(text("INSERT INTO purchase_orders(product_id, qty, placed_on, eta, status, source) VALUES (:p, :q, :pl, :e, :s, 'operator')"), {"p": order["product_id"], "q": order["qty"], "pl": order["placed"], "e": order["eta"], "s": "received" if order.get("done") else "open"})
    for sku, s in state.items():
        conn.execute(text("UPDATE inventory SET on_hand = :h, updated_at = now() WHERE product_id = :p"), {"h": s["on_hand"], "p": sku})
    conn.execute(text("UPDATE sim_state SET today = :d WHERE id = 1"), {"d": first_day + dt.timedelta(days=days - 1)})


def seed(db: Database, seed_value: int = 11, today: dt.date | None = None) -> bool:
    """Populate catalog and 120 days of synthetic sales. Returns False if data already exists."""
    today = today or dt.date.today()
    rng = np.random.default_rng(seed_value)
    with db.begin() as conn:
        if conn.execute(text("SELECT count(*) FROM products")).scalar_one():
            return False
        first = today - dt.timedelta(days=HISTORY_DAYS - 1)
        paths = {}
        for sku, name, category, price, base, elasticity, lead, image in CATALOG:
            paths[sku] = _price_path(rng, float(price), HISTORY_DAYS)
            conn.execute(text("INSERT INTO products(id, name, category, price, reference_price, cost, lead_days, image) VALUES (:i, :n, :c, :p, :p, :cost, :l, :img)"),
                         {"i": sku, "n": name, "c": category, "p": price, "cost": round(price * 0.7, 2), "l": lead, "img": f"https://cdn.dummyjson.com/product-images/{image}/thumbnail.webp"})
            conn.execute(text("INSERT INTO inventory(product_id, on_hand) VALUES (:i, :h)"), {"i": sku, "h": int(base * 14)})
            conn.execute(text("INSERT INTO sim_truth(product_id, true_elasticity, base_units) VALUES (:i, :e, :b)"), {"i": sku, "e": elasticity, "b": base})
        conn.execute(text("INSERT INTO sim_state(id, today) VALUES (1, :d)"), {"d": today})
        _simulate(conn, first, HISTORY_DAYS, rng, paths)
    return True


def advance(db: Database, days: int, seed_value: int | None = None) -> dt.date:
    """Move the simulated market forward using current prices, stock and purchase orders."""
    days = max(1, min(int(days), 30))
    with db.begin() as conn:
        conn.execute(text("LOCK TABLE sim_state IN EXCLUSIVE MODE"))
        today = conn.execute(text("SELECT today FROM sim_state WHERE id = 1")).scalar_one()
        index = (today - conn.execute(text("SELECT min(day) FROM sales_daily")).scalar_one()).days + 1
        rng = np.random.default_rng(seed_value if seed_value is not None else int(today.toordinal()) * 7919)
        _simulate(conn, today + dt.timedelta(days=1), days, rng, None, index)
        return conn.execute(text("SELECT today FROM sim_state WHERE id = 1")).scalar_one()
