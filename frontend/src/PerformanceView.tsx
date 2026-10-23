import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, Boxes, CalendarPlus, CheckCircle2, LoaderCircle, TrendingUp } from "lucide-react";
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError, pct, rupees } from "./lib";
import type { Forecast, Overview, Performance, RunDetail, Stockout } from "./lib";

const BLUE = "#2563eb";
const ORANGE = "#ea580c";
const GRAY = "#94a3b8";

function Card({ title, sub, children, wide }: { title: string; sub?: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <section className={`panel card ${wide ? "wide" : ""}`}>
      <header className="card-head"><div><h3>{title}</h3>{sub && <p>{sub}</p>}</div></header>
      {children}
    </section>
  );
}

function Stat({ label, value, note, good }: { label: string; value: string; note?: string; good?: boolean | null }) {
  return <div className={`kpi ${good === true ? "good" : good === false ? "bad" : ""}`}><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>;
}

export default function PerformanceView({ overview, reload }: { overview: Overview; reload: () => Promise<void> }) {
  const [perf, setPerf] = useState<Performance | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [product, setProduct] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const latestId = overview.runs.find((run) => run.status === "completed")?.id;

  const guard = useCallback((reason: unknown) => {
    setError(reason instanceof ApiError ? reason.message : "Something went wrong.");
  }, []);

  useEffect(() => {
    let live = true;
    api<Performance>("/performance").then((value) => live && setPerf(value)).catch(guard);
    if (latestId) api<RunDetail>(`/runs/${latestId}`).then((value) => live && setDetail(value)).catch(guard);
    return () => { live = false; };
  }, [latestId, overview.market_day, refresh, guard]);

  const advance = async (days: number) => {
    setBusy(true); setError("");
    try { await api("/market/advance", "POST", { days }); await reload(); setRefresh((value) => value + 1); } catch (reason) { guard(reason); } finally { setBusy(false); }
  };

  const ids = detail?.forecasts.map((item) => item.product_id).filter((id, index, all) => all.indexOf(id) === index) ?? [];
  const chosen = ids.includes(product) ? product : ids[0] ?? "";
  const nameOf = (id: string) => overview.products.find((item) => item.id === id)?.name ?? id;
  const series = useMemo(() => (detail?.forecasts.filter((item: Forecast) => item.product_id === chosen) ?? []).map((item, index) => ({ day: `D${index + 1}`, mean: item.mean, baseline: item.baseline, range: [item.lower, item.upper] })), [detail, chosen]);
  const risks = useMemo(() => [...(detail?.stockout ?? [])].sort((a: Stockout, b: Stockout) => b.p_before_resupply - a.p_before_resupply).map((item) => ({ name: nameOf(item.product_id), risk: Math.round(item.p_before_resupply * 100), cover: item.days_cover })), [detail, overview]);
  const elasticity = perf?.elasticity.find((item) => item.product_id === chosen) ?? perf?.elasticity[0];

  if (!perf) return <div className="loading"><LoaderCircle className="spin" size={26} /></div>;
  const f = perf.forecast.overall;
  const s = perf.stockout;
  const covered = perf.elasticity.filter((item) => item.covered).length;
  const calibration = useMemo(() => s.calibration.map((row) => ({ predicted: Math.round(row.predicted * 100), observed: Math.round(row.observed * 100), n: row.n })), [s]);

  return (
    <>
      <div className="bar">
        <div><h1>Agent performance</h1><p>Every prediction an agent made is scored against what the market actually did later. Market day: <b>{perf.market_day}</b>.</p></div>
        <div className="bar-actions"><span className="hint">Predictions need time to be proven:</span><button className="btn ghost" disabled={busy} onClick={() => void advance(7)}>{busy ? <LoaderCircle className="spin" size={14} /> : <CalendarPlus size={14} />} +7 days</button><button className="btn" disabled={busy} onClick={() => void advance(14)}>+14 days</button></div>
      </div>
      {error && <div className="alert" role="alert"><span>{error}</span></div>}
      {perf.what_if_runs_excluded > 0 && <p className="note">{perf.what_if_runs_excluded} what-if (sale) run{perf.what_if_runs_excluded > 1 ? "s are" : " is"} excluded from accuracy, because a hypothetical +35% sale never happened in the market.</p>}

      <div className="kpis">
        <Stat label="Demand forecast error (WAPE)" value={f ? pct(f.wape, 1) : "Not yet scored"} note={f ? `Naive baseline ${pct(f.baseline_wape, 1)} · ${f.n} predictions scored` : "Advance the market to score forecasts"} good={f ? f.wape < f.baseline_wape : null} />
        <Stat label="Forecast interval coverage" value={f ? pct(f.coverage) : "n/a"} note="Share of actual days inside the 95% band" good={f ? f.coverage >= 0.85 : null} />
        <Stat label="Stockout calls (Brier score)" value={s.brier === null ? "Not yet scored" : s.brier.toFixed(3)} note={s.n ? `${s.n} predictions scored over 14 days · lower is better` : "Advance the market to score stockout calls"} good={s.brier === null ? null : s.brier < 0.25} />
        <Stat label="Price elasticity within 95% error" value={perf.elasticity.length ? `${covered} of ${perf.elasticity.length}` : "n/a"} note="Estimated vs the simulator's true value" good={perf.elasticity.length ? covered >= perf.elasticity.length * 0.75 : null} />
      </div>

      <div className="grid2">
        <Card title="Demand forecast" sub="Next 14 days, with the 95% band and a naive last-week baseline" wide>
          <div className="picker"><select aria-label="Product" value={chosen} onChange={(event) => setProduct(event.target.value)}>{ids.map((id) => <option key={id} value={id}>{nameOf(id)}</option>)}</select></div>
          {series.length ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={series} margin={{ left: -10, right: 12, top: 10 }}><CartesianGrid stroke="#eef2f7" vertical={false} /><XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} /><YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} /><Tooltip /><Legend /><Area dataKey="range" stroke="none" fill={BLUE} fillOpacity={0.12} name="95% band" /><Line dataKey="baseline" stroke={GRAY} strokeDasharray="5 4" dot={false} name="Naive baseline" /><Line dataKey="mean" stroke={BLUE} strokeWidth={2.5} dot={false} name="Agent forecast" /></ComposedChart></ResponsiveContainer></div> : <p className="muted pad">Run the agents to see a forecast.</p>}
        </Card>

        <Card title="Stockout prediction" sub="Chance each product sells out within 14 days if nothing new is ordered">
          {risks.length ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={risks} layout="vertical" margin={{ left: 20, right: 20 }}><CartesianGrid stroke="#eef2f7" horizontal={false} /><XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={{ fontSize: 11 }} /><YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11 }} /><Tooltip formatter={(value) => `${value}%`} /><ReferenceLine x={50} stroke={ORANGE} strokeDasharray="4 4" /><Bar dataKey="risk" fill={ORANGE} radius={[0, 4, 4, 0]} name="Risk" /></BarChart></ResponsiveContainer></div> : <p className="muted pad">No predictions yet.</p>}
        </Card>

        <Card title="Were the stockout calls right?" sub="Predicted chance vs how often stockouts really happened">
          {s.n > 0 ? (<>
            <div className="confusion"><div className="tp"><b>{s.tp}</b><span>Warned, sold out</span></div><div className="fp"><b>{s.fp}</b><span>Warned, did not</span></div><div className="fn"><b>{s.fn}</b><span>Missed a stockout</span></div><div className="tn"><b>{s.tn}</b><span>Correctly calm</span></div></div>
            {calibration.length > 0 && <div className="chart short"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={calibration} margin={{ left: -10, right: 12 }}><CartesianGrid stroke="#eef2f7" /><XAxis dataKey="predicted" type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} name="Predicted" /><YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} /><Tooltip /><Line data={[{ predicted: 0, observed: 0 }, { predicted: 100, observed: 100 }]} dataKey="observed" stroke={GRAY} strokeDasharray="4 4" dot={false} name="Perfect" /><Scatter dataKey="observed" fill={BLUE} name="Observed" /></ComposedChart></ResponsiveContainer></div>}
            <p className="muted pad small">{s.acted} of {s.n} flagged products had a restock applied. Calls assume no new orders, so they run higher than reality when an operator keeps reordering. Compare the dots to the dashed line to see how far.</p>
          </>) : <p className="muted pad">Advance the market to let these predictions play out.</p>}
        </Card>

        <Card title="Price elasticity curve" sub="How demand responds to price, with the estimate's 95% range" wide>
          {elasticity ? (<>
            <div className="picker"><select aria-label="Elasticity product" value={elasticity.product_id} onChange={(event) => setProduct(event.target.value)}>{perf.elasticity.map((item) => <option key={item.product_id} value={item.product_id}>{item.name}</option>)}</select></div>
            <div className="elas-facts"><Stat label="Estimated elasticity" value={`${elasticity.elasticity.toFixed(2)} ± ${(1.96 * elasticity.std_err).toFixed(2)}`} note={`${elasticity.n_obs} sales days · ${elasticity.usable ? "reliable enough to act on" : "too uncertain to act on"}`} good={elasticity.usable} /><Stat label="Simulator's true value" value={elasticity.true_elasticity.toFixed(2)} note={elasticity.covered ? "Inside the estimate's range" : "Outside the estimate's range"} good={elasticity.covered} /></div>
            {elasticity.curve.length ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={elasticity.curve.map((row) => ({ price: row.price, demand: row.demand, range: [row.lower, row.upper] }))} margin={{ left: -10, right: 16, top: 10 }}><CartesianGrid stroke="#eef2f7" vertical={false} /><XAxis dataKey="price" tickFormatter={(value) => rupees(Number(value))} tick={{ fontSize: 10 }} minTickGap={30} /><YAxis tick={{ fontSize: 11 }} /><Tooltip labelFormatter={(value) => rupees(Number(value))} /><Area dataKey="range" stroke="none" fill={ORANGE} fillOpacity={0.12} name="95% range" /><Line dataKey="demand" stroke={ORANGE} strokeWidth={2.5} dot={false} name="Units per day" /></ComposedChart></ResponsiveContainer></div> : <p className="muted pad">{elasticity.reason}</p>}
          </>) : <p className="muted pad">Run the agents to estimate elasticity.</p>}
        </Card>

        <Card title="Elasticity accuracy" sub="Estimated vs true, per product" >
          {perf.elasticity.length ? <ul className="rows">{perf.elasticity.map((row) => <li key={row.product_id}><span>{row.name}</span><b>{row.elasticity.toFixed(2)}</b><small>true {row.true_elasticity.toFixed(2)}</small>{row.covered ? <CheckCircle2 size={14} className="ok" /> : <span className="warn">outside</span>}</li>)}</ul> : <p className="muted pad">n/a</p>}
        </Card>

        <Card title="Applied price changes: predicted vs actual" sub="Seven days after each approved price change">
          {perf.price_outcomes.length ? <ul className="rows">{perf.price_outcomes.map((row, index) => <li key={index}><span>{row.name}</span><small>predicted {row.predicted?.toFixed(1)}/day</small><b>actual {row.actual?.toFixed(1)}/day</b></li>)}</ul> : <p className="muted pad">Approve a price change, then advance the market 7 days.</p>}
        </Card>

        <Card title="Agents" sub="How the three agents actually executed">
          <ul className="rows agents">{perf.agents.map((row) => <li key={row.agent}><span>{row.agent.replace("_", " ")}</span><small>{row.runs} runs · {(row.avg_ms / 1000).toFixed(1)}s avg · {row.avg_sql.toFixed(1)} SQL/run</small><b>{row.model_runs} on Gemini{row.fell_back ? ` · ${row.fell_back} fell back` : ""}</b></li>)}</ul>
          <p className="muted pad small"><Activity size={12} /> {perf.guardrail_events} guardrail interventions recorded (rejected notes, removed selections, blocked inputs).</p>
        </Card>

        <Card title="Decision funnel" sub="What happened to the agents' proposals">
          {perf.decisions.length ? <ul className="rows">{perf.decisions.map((row, index) => <li key={index}><span>{row.kind}</span><small>{row.status}</small><b>{row.n}</b></li>)}</ul> : <p className="muted pad">No decisions yet.</p>}
          <p className="muted pad small"><Boxes size={12} /> <TrendingUp size={12} /> Approve or reject on the Live run tab.</p>
        </Card>
      </div>
    </>
  );
}
