"""Agent performance, scored in SQL by comparing what agents predicted with what the market later did."""

from sqlalchemy import text

from backend.db import Database

FORECAST_ACCURACY = """
WITH scored AS (
  SELECT f.run_id, f.product_id, f.mean, f.lower, f.upper, f.baseline, s.units
  FROM forecasts f
  JOIN agent_runs r ON r.id = f.run_id AND r.scenario = 'baseline' AND r.status = 'completed'
  JOIN sales_daily s ON s.product_id = f.product_id AND s.day = f.target_day
  WHERE NOT s.stockout)
SELECT product_id,
       count(*)::int AS n,
       (sum(abs(mean - units)) / NULLIF(sum(units), 0))::float AS wape,
       (sum(abs(baseline - units)) / NULLIF(sum(units), 0))::float AS baseline_wape,
       avg(mean - units)::float AS bias,
       avg((units BETWEEN lower AND upper)::int)::float AS coverage
FROM scored GROUP BY ROLLUP (product_id) ORDER BY product_id NULLS FIRST"""

STOCKOUT_SCORE = """
WITH latest AS (
  SELECT DISTINCT ON (sp.product_id, sp.made_on) sp.*
  FROM stockout_predictions sp
  JOIN agent_runs r ON r.id = sp.run_id AND r.scenario = 'baseline' AND r.status = 'completed'
  ORDER BY sp.product_id, sp.made_on, r.started_at DESC),
due AS (
  SELECT * FROM latest WHERE made_on + 14 <= (SELECT today FROM sim_state WHERE id = 1)),
outcome AS (
  SELECT d.run_id, d.product_id, d.p_stockout AS p,
         EXISTS (SELECT 1 FROM sales_daily s WHERE s.product_id = d.product_id AND s.stockout AND s.day > d.made_on AND s.day <= d.made_on + 14) AS hit,
         EXISTS (SELECT 1 FROM decisions x WHERE x.run_id = d.run_id AND x.product_id = d.product_id AND x.kind IN ('reorder','expedite') AND x.status = 'applied') AS acted
  FROM due d)
SELECT count(*)::int AS n,
       avg(power(p - hit::int, 2))::float AS brier,
       avg(((p > 0.5) = hit)::int)::float AS accuracy,
       count(*) FILTER (WHERE p > 0.5 AND hit)::int AS tp,
       count(*) FILTER (WHERE p > 0.5 AND NOT hit)::int AS fp,
       count(*) FILTER (WHERE p <= 0.5 AND hit)::int AS fn,
       count(*) FILTER (WHERE p <= 0.5 AND NOT hit)::int AS tn,
       count(*) FILTER (WHERE acted)::int AS acted
FROM outcome"""

CALIBRATION = """
WITH latest AS (
  SELECT DISTINCT ON (sp.product_id, sp.made_on) sp.*
  FROM stockout_predictions sp
  JOIN agent_runs r ON r.id = sp.run_id AND r.scenario = 'baseline' AND r.status = 'completed'
  ORDER BY sp.product_id, sp.made_on, r.started_at DESC),
due AS (SELECT * FROM latest WHERE made_on + 14 <= (SELECT today FROM sim_state WHERE id = 1))
SELECT width_bucket(LEAST(d.p_stockout, 0.9999), 0, 1, 5) AS bucket,
       count(*)::int AS n,
       avg(d.p_stockout)::float AS predicted,
       avg((EXISTS (SELECT 1 FROM sales_daily s WHERE s.product_id = d.product_id AND s.stockout AND s.day > d.made_on AND s.day <= d.made_on + 14))::int)::float AS observed
FROM due d GROUP BY 1 ORDER BY 1"""

ELASTICITY = """
SELECT e.product_id, p.name, e.elasticity, e.std_err, e.n_obs, e.r2, e.usable, e.reason, t.true_elasticity,
       abs(e.elasticity - t.true_elasticity) AS abs_error,
       (abs(e.elasticity - t.true_elasticity) <= 1.96 * e.std_err) AS covered, e.curve
FROM elasticity_estimates e
JOIN products p ON p.id = e.product_id
JOIN sim_truth t ON t.product_id = e.product_id
WHERE e.run_id = (SELECT id FROM agent_runs WHERE status = 'completed' ORDER BY started_at DESC LIMIT 1) AND e.elasticity IS NOT NULL
ORDER BY e.product_id"""

PRICE_OUTCOMES = """
SELECT d.product_id, p.name, d.applied_on, d.predicted_units::float AS predicted,
       (SELECT avg(units) FROM sales_daily s WHERE s.product_id = d.product_id AND NOT s.stockout AND s.day > d.applied_on AND s.day <= d.applied_on + 7)::float AS actual,
       (SELECT count(*) FROM sales_daily s WHERE s.product_id = d.product_id AND s.day > d.applied_on AND s.day <= d.applied_on + 7)::int AS days_observed
FROM decisions d JOIN products p ON p.id = d.product_id
WHERE d.kind = 'price' AND d.status = 'applied' AND d.applied_on IS NOT NULL
ORDER BY d.applied_on DESC LIMIT 20"""

AGENT_STATS = """
SELECT m.agent, count(*)::int AS runs, avg(m.duration_ms)::float AS avg_ms, avg(m.tool_calls)::float AS avg_tools, avg(m.sql_queries)::float AS avg_sql,
       count(*) FILTER (WHERE m.provider = 'Gemini')::int AS model_runs, sum(m.completed_locally)::int AS fell_back
FROM agent_metrics m JOIN agent_runs r ON r.id = m.run_id WHERE r.status = 'completed' GROUP BY m.agent ORDER BY m.agent"""

DECISION_FUNNEL = "SELECT kind, status, count(*)::int AS n FROM decisions GROUP BY kind, status ORDER BY kind, status"
GUARDRAILS = "SELECT count(*)::int FROM agent_events WHERE kind = 'guardrail'"
WINDOW = "SELECT (SELECT today FROM sim_state WHERE id = 1) AS today, (SELECT count(*) FROM agent_runs WHERE scenario = 'sale_surge' AND status = 'completed')::int AS excluded"


def performance(db: Database) -> dict:
    with db.engine.connect() as conn:
        rows = lambda q: [dict(r) for r in conn.execute(text(q)).mappings()]
        forecast = rows(FORECAST_ACCURACY)
        overall = next((r for r in forecast if r["product_id"] is None), None)
        stock = dict(conn.execute(text(STOCKOUT_SCORE)).mappings().one())
        meta = dict(conn.execute(text(WINDOW)).mappings().one())
        return {
            "market_day": meta["today"].isoformat(),
            "what_if_runs_excluded": meta["excluded"],
            "forecast": {"overall": overall, "by_product": [r for r in forecast if r["product_id"] is not None]},
            "stockout": {**stock, "calibration": rows(CALIBRATION)},
            "elasticity": rows(ELASTICITY),
            "price_outcomes": [r for r in rows(PRICE_OUTCOMES) if r["days_observed"] == 7],
            "agents": rows(AGENT_STATS),
            "decisions": rows(DECISION_FUNNEL),
            "guardrail_events": conn.execute(text(GUARDRAILS)).scalar_one(),
        }
