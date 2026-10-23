"""Run lifecycle and decision execution against PostgreSQL."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from sqlalchemy import text

from backend import agents
from backend.db import Database
from backend.guardrails import sanitize_untrusted

logger = logging.getLogger(__name__)
GOALS = {"baseline": "Keep best sellers in stock and protect margin", "sale_surge": "Get ready for a big sale without running out of stock or losing margin"}
DECISION_COLUMNS = "id, run_id, product_id, agent, kind, before_value, after_value, quantity, reason, evidence, needs_approval, status, predicted_units, created_at, decided_at, applied_on"


def decide(db: Database, decision_id: str, approve: bool, automatic: bool = False) -> dict:
    with db.begin() as conn:
        row = conn.execute(text("SELECT d.id, d.product_id, d.kind, d.before_value, d.after_value, d.quantity, d.needs_approval, d.status, p.price::float AS price, p.lead_days "
                                "FROM decisions d JOIN products p ON p.id = d.product_id WHERE d.id = :i FOR UPDATE OF d, p"), {"i": decision_id}).mappings().first()
        if row is None:
            raise KeyError("Decision not found")
        if row["status"] == "pending" and not (automatic and row["needs_approval"]):
            today = conn.execute(text("SELECT today FROM sim_state WHERE id = 1")).scalar_one()
            status = "rejected"
            if approve:
                status = "stale"
                if row["kind"] == "price":
                    if abs(row["price"] - row["before_value"]) < 0.005:
                        conn.execute(text("UPDATE products SET price = :p, updated_at = now() WHERE id = :i"), {"p": row["after_value"], "i": row["product_id"]})
                        status = "applied"
                elif row["kind"] == "reorder":
                    open_qty = conn.execute(text("SELECT COALESCE(SUM(qty), 0) FROM purchase_orders WHERE product_id = :i AND status = 'open'"), {"i": row["product_id"]}).scalar_one()
                    if int(open_qty) == int(row["before_value"]):
                        conn.execute(text("INSERT INTO purchase_orders(product_id, qty, placed_on, eta, source) VALUES (:i, :q, :t, :t + :l, 'agent')"), {"i": row["product_id"], "q": row["quantity"], "t": today, "l": row["lead_days"]})
                        status = "applied"
                elif row["kind"] == "expedite":
                    moved = conn.execute(text("UPDATE purchase_orders SET expedited = true, eta = LEAST(eta, :t + GREATEST(1, :l / 2)) WHERE product_id = :i AND status = 'open' AND NOT expedited"),
                                         {"i": row["product_id"], "t": today, "l": row["lead_days"]}).rowcount
                    if moved:
                        status = "applied"
            conn.execute(text("UPDATE decisions SET status = :s, decided_at = now(), applied_on = CASE WHEN :s = 'applied' THEN CAST(:t AS date) END WHERE id = :i"), {"s": status, "t": today, "i": decision_id})
        return dict(conn.execute(text(f"SELECT {DECISION_COLUMNS} FROM decisions WHERE id = :i"), {"i": decision_id}).mappings().one())


class Runner:
    def __init__(self, db: Database):
        self.db = db
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-run")

    def start(self, goal: str | None, scenario: str, product_ids: list[str], resume_id: str | None = None) -> str:
        clean = GOALS[scenario]
        if goal and goal.strip():
            text_goal, _, unsafe = sanitize_untrusted(goal, 300)
            if unsafe:
                raise PermissionError("That goal can't include instructions to change rules or reveal information")
            clean = text_goal or clean
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("A run is already in progress")
        try:
            with self.db.begin() as conn:
                known = {row[0] for row in conn.execute(text("SELECT id FROM products"))}
                if not product_ids:
                    product_ids = sorted(known)[:16]
                if not product_ids or len(product_ids) > 16 or not set(product_ids) <= known:
                    raise ValueError("Choose between 1 and 16 products from the catalog")
                mode = "autopilot" if self.db.setting("auto_approve", False) else "review"
                if resume_id:
                    row = conn.execute(text("SELECT scenario, goal, status FROM agent_runs WHERE id = :i FOR UPDATE"), {"i": resume_id}).mappings().first()
                    if not row or row["status"] not in ("failed", "interrupted"):
                        raise ValueError("Only a failed or interrupted run can be resumed")
                    conn.execute(text("UPDATE agent_runs SET status = 'running', error = NULL WHERE id = :i"), {"i": resume_id})
                    run_id, scenario, clean = resume_id, row["scenario"], row["goal"]
                    ids = [r[0] for r in conn.execute(text("SELECT DISTINCT product_id FROM (SELECT product_id FROM forecasts WHERE run_id = :i UNION SELECT product_id FROM decisions WHERE run_id = :i) x"), {"i": run_id})] or product_ids
                else:
                    run_id, ids = str(uuid4()), product_ids
                    conn.execute(text("INSERT INTO agent_runs(id, scenario, goal, status, sim_day, execution_mode) VALUES (:i, :s, :g, 'running', (SELECT today FROM sim_state WHERE id = 1), :m)"), {"i": run_id, "s": scenario, "g": clean, "m": mode})
        except Exception:
            self.lock.release()
            raise
        self.executor.submit(self._work, run_id, clean, scenario, ids, bool(resume_id))
        return run_id

    def _work(self, run_id: str, goal: str, scenario: str, ids: list[str], resume: bool):
        try:
            agents.execute(self.db, run_id, goal, scenario, ids, resume)
            with self.db.begin() as conn:
                conn.execute(text("UPDATE agent_runs SET status = 'completed', finished_at = now() WHERE id = :i"), {"i": run_id})
                mode = conn.execute(text("SELECT execution_mode FROM agent_runs WHERE id = :i"), {"i": run_id}).scalar_one()
                routine = [row[0] for row in conn.execute(text("SELECT id FROM decisions WHERE run_id = :i AND status = 'pending' AND NOT needs_approval"), {"i": run_id})] if mode == "autopilot" else []
            for identifier in routine:
                decide(self.db, str(identifier), True, automatic=True)
        except Exception:
            logger.exception("Agent run failed")
            with self.db.begin() as conn:
                conn.execute(text("UPDATE agent_runs SET status = 'failed', error = 'Run stopped. It can be retried from its last checkpoint.', finished_at = now() WHERE id = :i"), {"i": run_id})
        finally:
            self.lock.release()

    def close(self):
        self.executor.shutdown(wait=True)
