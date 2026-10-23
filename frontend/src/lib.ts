export type Product = { id: string; name: string; category: string; price: number; reference_price: number; cost: number; lead_days: number; image: string; on_hand: number; on_order: number; sold_7d: number };
export type RunSummary = { id: string; scenario: string; goal: string; status: string; sim_day: string; execution_mode: string; started_at: string; finished_at: string | null; error: string | null; decisions: number };
export type Overview = { products: Product[]; runs: RunSummary[]; pending: number; market_day: string; auto_approve: boolean; gemini: boolean };
export type AgentEvent = { seq: number; agent: string; kind: string; title: string; detail: Record<string, unknown>; ts: string };
export type Metric = { agent: string; provider: string; model_calls: number; tool_calls: number; sql_queries: number; duration_ms: number; completed_locally: number; rationale: string };
export type Decision = { id: string; product_id: string; product_name: string; agent: string; kind: "reorder" | "price" | "expedite"; before_value: number | null; after_value: number | null; quantity: number | null; reason: string; evidence: Record<string, unknown>; needs_approval: boolean; status: string };
export type Forecast = { product_id: string; target_day: string; mean: number; lower: number; upper: number; baseline: number };
export type Stockout = { product_id: string; made_on: string; window_days: number; p_stockout: number; p_before_resupply: number; expected_day: number; days_cover: number; on_hand: number; incoming: number };
export type ElasticityRow = { product_id: string; elasticity: number | null; std_err: number | null; n_obs: number; r2: number | null; usable: boolean; reason: string; curve: { price: number; demand: number; lower: number; upper: number }[] };
export type RunDetail = { run: RunSummary; metrics: Metric[]; decisions: Decision[]; forecasts: Forecast[]; stockout: Stockout[]; elasticity: ElasticityRow[]; quality: { product_id: string; wape: number; baseline_wape: number; days_used: number }[] };
export type Performance = {
  market_day: string;
  what_if_runs_excluded: number;
  forecast: { overall: { n: number; wape: number; baseline_wape: number; bias: number; coverage: number } | null; by_product: { product_id: string; n: number; wape: number; baseline_wape: number; bias: number; coverage: number }[] };
  stockout: { n: number; brier: number | null; accuracy: number | null; tp: number; fp: number; fn: number; tn: number; acted: number; calibration: { bucket: number; n: number; predicted: number; observed: number }[] };
  elasticity: { product_id: string; name: string; elasticity: number; std_err: number; n_obs: number; r2: number; usable: boolean; reason: string; true_elasticity: number; abs_error: number; covered: boolean; curve: { price: number; demand: number; lower: number; upper: number }[] }[];
  price_outcomes: { product_id: string; name: string; applied_on: string; predicted: number; actual: number | null }[];
  agents: { agent: string; runs: number; avg_ms: number; avg_tools: number; avg_sql: number; model_runs: number; fell_back: number }[];
  decisions: { kind: string; status: string; n: number }[];
  guardrail_events: number;
};

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function api<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, { method, credentials: "same-origin", headers: body === undefined ? undefined : { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(60000) });
  } catch {
    throw new ApiError("Can't reach the server. Check that it is running.", 0);
  }
  if (!response.ok) {
    const detail = await response.json().then((value) => value?.detail).catch(() => undefined);
    throw new ApiError(typeof detail === "string" ? detail : "Something went wrong.", response.status);
  }
  return response.json();
}

const _inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
export const rupees = (value: number) => _inr.format(value);
export const pct = (value: number | null | undefined, digits = 0) => (value === null || value === undefined ? "n/a" : `${(value * 100).toFixed(digits)}%`);
export const AGENTS: Record<string, string> = { supervisor: "Supervisor", demand_forecaster: "Demand forecaster", inventory_optimizer: "Inventory optimizer", dynamic_pricer: "Dynamic pricer", policy_gate: "Policy gate", risk_critic: "Risk critic", risk_review: "Risk review" };
