# Agent Operations

Three autonomous agents (inventory optimizer, dynamic pricer, demand forecaster), orchestrated as a **LangGraph state graph**, that query **PostgreSQL**, run **ML models**, and propose **automated decisions**, with real-time visibility into what they do and how accurate they turn out to be.

## What runs

```
supervisor -> demand_forecaster -> [inventory_optimizer || dynamic_pricer] -> policy_gate -> risk_critic -> (risk_review) -> persist
```

| Agent | PostgreSQL it reads | Model it runs | Decision it proposes |
| --- | --- | --- | --- |
| Demand forecaster | `sales_history`, `sales_summary` | Ridge on trend, weekday and log price; stock-out days are censored | 14-day forecast with 95% band, scored against a naive baseline |
| Inventory optimizer | `stock_positions`, `open_orders` | Monte Carlo simulation of demand vs stock and incoming orders | Purchase order, or expedite an existing one |
| Dynamic pricer | `sales_history`, `sales_summary` | Poisson regression estimating price elasticity with a standard error | Price change, only where the estimate is reliable |

Each agent is a bounded tool-use loop (at most 4 model turns, 10 tool calls). With `GEMINI_API_KEY`, Gemini chooses and calls the tools and writes a one-sentence note. Without it, or if it fails or is rejected, a **local policy** completes the step. Each agent's provider and any fallback are recorded and shown in the UI.

## How decisions stay safe

- **Read-only database role.** Agents query through `agent_ro`, which has `SELECT` on five tables only. It cannot write, drop, or read ground truth.
- **Allowlisted, parameterised SQL.** The model can only call named queries; it never writes SQL. Every query's exact text, row count and duration is stored and shown live.
- **Hard limits re-checked independently:** price change <= 5%, gross margin >= 25%, reorder <= 2,000 units. A gate re-validates every proposal regardless of what an agent said.
- **Material decisions are approval-only** (price move > 3%, purchase > INR 75,000, any expedite), even with auto-approve on.
- **Hallucination guard.** Agent notes may only contain numbers that appear in their own tool output. Notes with invented numbers, links, personal data or injected instructions are rejected, and selections without supporting evidence are removed.
- **PII and injection.** Operator text is sanitised before any model sees it; goals that try to override rules are refused.
- **Transactional decisions.** Applying a decision locks its rows, checks the value is unchanged (otherwise it is marked stale) and is idempotent. Concurrent approvals are serialised by a row lock so a decision applies once.
- **Access control.** Signed-in operators only, HttpOnly SameSite=Strict cookie, login rate limit, Origin checks, host allowlist, security headers, API docs disabled, generic error messages, hourly run budget.

## Agent performance (what is tracked)

Predictions are stored and later scored **in SQL** against what the market actually did:

- **Demand forecasts:** WAPE vs a naive last-week baseline, bias, and 95% interval coverage.
- **Stockout predictions:** Brier score, confusion matrix and a calibration chart over a fixed 14-day horizon.
- **Price elasticity:** estimate with a confidence range, compared with the simulator's true value, plus predicted vs actual units after each approved price change.
- **Agents:** runs, latency, SQL per run, how many ran on Gemini and how many fell back.

What-if runs (for example a +35% sale) are excluded from accuracy, since that scenario never happens in the market.

## Run it

Requires Python 3.11+, Node 22+ and PostgreSQL 14+.

```sh
python -m pip install -r backend/requirements.txt
cp .env.example .env        # fill in DATABASE_URL, SELLER_ACCESS_CODE, SESSION_SECRET, GEMINI_API_KEY
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
cd frontend && npm ci && npm run dev
```

On first start the app creates the schema, the read-only `agent_ro` role (set `AGENT_DB_PASSWORD` and `AGENT_DB_URL`) and 120 days of simulated sales.

## Honest limits

- **The market is simulated.** Sales, prices, orders and the true elasticities come from a simulator, so accuracy figures describe the models on that data, not on a real store. Elasticity is only identifiable because the simulated history contains price variation; real data needs the same.
- **Stockout calls run high.** They assume no new orders arrive in the next 14 days, but the simulated operator keeps reordering. The calibration chart shows the gap; do not read the raw probabilities as frequencies.
- **Decisions change this database only.** Nothing is sent to Flipkart or any marketplace, and there is no live Flipkart data connection.
- **Single worker, no migrations.** One API process, `create_all`-style schema bootstrap, no distributed job queue. Graph checkpoints are stored in PostgreSQL.
- **Unverified here:** no automated tests are included, Docker/compose files were not run, and auth is a shared access code, not per-user identity.
