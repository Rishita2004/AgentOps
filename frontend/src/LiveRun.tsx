import React, { useCallback, useEffect, useRef, useState } from "react";
import { Activity, AlertTriangle, Boxes, Check, CheckCircle2, Circle, Code2, Database, GitBranch, LoaderCircle, PackageCheck, Play, RefreshCw, ShieldCheck, Sparkles, TrendingUp, X, Zap } from "lucide-react";
import { api, ApiError, AGENTS, rupees } from "./lib";
import type { AgentEvent, Decision, Overview, RunDetail } from "./lib";

type NodeId = "supervisor" | "demand_forecaster" | "inventory_optimizer" | "dynamic_pricer" | "policy_gate" | "risk_critic" | "risk_review";

function stateOf(node: NodeId, events: AgentEvent[]): "idle" | "working" | "done" {
  const mine = events.filter((event) => event.agent === node);
  if (!mine.length) return "idle";
  if (node === "supervisor") return events.some((event) => event.kind === "plan") ? "done" : "working";
  if (node === "demand_forecaster" || node === "inventory_optimizer" || node === "dynamic_pricer") return mine.some((event) => event.kind === "agent_completed") ? "done" : "working";
  return "done";
}

function Node({ id, icon: Icon, sub, events, picked, onPick }: { id: NodeId; icon: typeof Activity; sub: string; events: AgentEvent[]; picked: string; onPick: (id: string) => void }) {
  const state = stateOf(id, events);
  const sql = events.filter((event) => event.agent === id && event.kind === "sql").length;
  const tools = events.filter((event) => event.agent === id && event.kind === "tool_call").length;
  return (
    <button className={`node ${state} ${picked === id ? "picked" : ""}`} onClick={() => onPick(id)} aria-pressed={picked === id}>
      <span className="node-icon"><Icon size={17} /></span>
      <span className="node-text"><b>{AGENTS[id]}</b><small>{sub}</small></span>
      <span className="node-state">{state === "done" ? <CheckCircle2 size={16} /> : state === "working" ? <LoaderCircle className="spin" size={16} /> : <Circle size={14} />}</span>
      {(sql > 0 || tools > 0) && <span className="node-counts"><Database size={11} /> {sql} SQL <Code2 size={11} /> {tools} tools</span>}
    </button>
  );
}

const EventRow = React.memo(function EventRow({ event }: { event: AgentEvent }) {
  const detail = event.detail as Record<string, unknown>;
  const label = event.kind === "sql" ? "SQL" : event.kind === "tool_call" ? "TOOL" : event.kind === "tool_result" ? "RESULT" : event.kind === "guardrail" ? "GUARD" : event.kind === "fallback" ? "FALLBACK" : event.kind.replace("_", " ").toUpperCase();
  return (
    <li className={`ev ${event.kind}`}>
      <span className="ev-tag">{label}</span>
      <div>
        <div className="ev-head"><b>{AGENTS[event.agent] ?? event.agent}</b><span>{event.title}</span><time>{new Date(event.ts).toLocaleTimeString("en-IN")}</time></div>
        {event.kind === "sql" && <pre className="ev-sql">{String(detail.query)}</pre>}
        {event.kind === "sql" && <small className="ev-meta">{String(detail.rows)} rows · {String(detail.duration_ms)} ms · read-only role</small>}
        {event.kind !== "sql" && Object.keys(detail).length > 0 && <details><summary>Details</summary><pre>{JSON.stringify(detail, null, 2)}</pre></details>}
      </div>
    </li>
  );
});

function DecisionCard({ decision, busy, onDecide }: { decision: Decision; busy: boolean; onDecide: (approve: boolean) => void }) {
  const Icon = decision.kind === "price" ? TrendingUp : decision.kind === "reorder" ? PackageCheck : Zap;
  const text = decision.kind === "price" ? `${rupees(decision.before_value ?? 0)} to ${rupees(decision.after_value ?? 0)}` : decision.kind === "reorder" ? `Order ${decision.quantity} units` : `Expedite ${decision.quantity} units`;
  return (
    <article className={`dec ${decision.kind}`}>
      <div className="dec-top"><Icon size={16} /><b>{decision.kind === "price" ? "Price change" : decision.kind === "reorder" ? "Purchase order" : "Expedite"}</b><span className={`pill ${decision.status}`}>{decision.status === "pending" ? (decision.needs_approval ? "Needs approval" : "Ready") : decision.status}</span></div>
      <h4>{decision.product_name}</h4>
      <p className="dec-value">{text}</p>
      <p className="dec-reason">{decision.reason}</p>
      <small className="dec-by">{decision.agent}</small>
      {decision.status === "pending" ? (
        <div className="dec-actions"><button className="btn ghost" disabled={busy} onClick={() => onDecide(false)}><X size={14} /> Reject</button><button className="btn" disabled={busy} onClick={() => onDecide(true)}>{busy ? <LoaderCircle className="spin" size={14} /> : <Check size={14} />} Approve</button></div>
      ) : (
        <small className="dec-final"><CheckCircle2 size={13} /> {decision.status === "applied" ? "Applied to PostgreSQL" : decision.status === "rejected" ? "Rejected, no changes" : "Out of date, not applied"}</small>
      )}
    </article>
  );
}

export default function LiveRun({ overview, reload }: { overview: Overview; reload: () => Promise<void> }) {
  const [runId, setRunId] = useState<string | null>(overview.runs[0]?.id ?? null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [scenario, setScenario] = useState<"baseline" | "sale_surge">("baseline");
  const [picked, setPicked] = useState("all");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [stream, setStream] = useState("");
  const logRef = useRef<HTMLOListElement>(null);
  const run = overview.runs.find((item) => item.id === runId) ?? overview.runs[0];
  const running = run?.status === "running";
  const runStatus = run?.status;

  const guard = useCallback((reason: unknown) => {
    setError(reason instanceof ApiError ? reason.message : "Something went wrong.");
  }, []);

  const loadDetail = useCallback((id: string) => api<RunDetail>(`/runs/${id}`).then(setDetail).catch(guard), [guard]);

  useEffect(() => {
    if (!runId) return;
    let source: EventSource | null = null;
    let cancelled = false;
    void api<AgentEvent[]>(`/runs/${runId}/events/all`).then((all) => {
      if (cancelled) return;
      setEvents(all);
      void loadDetail(runId);
      if (run?.status !== "running") return;
      source = new EventSource(`/api/runs/${runId}/events`);
      setStream("live");
      const seen = new Set(all.map((item) => item.seq));
      source.addEventListener("agent", (message) => {
        const next = JSON.parse((message as MessageEvent).data) as AgentEvent;
        if (seen.has(next.seq)) return;
        seen.add(next.seq);
        setEvents((current) => [...current, next]);
      });
      source.addEventListener("done", () => { source?.close(); setStream(""); void reload(); void loadDetail(runId); });
      source.onerror = () => setStream("reconnecting");
    }).catch(guard);
    return () => { cancelled = true; source?.close(); setStream(""); };
  }, [runId, runStatus, guard, loadDetail, reload]);

  useEffect(() => { logRef.current?.lastElementChild?.scrollIntoView({ block: "nearest" }); }, [events.length]);

  const start = async () => {
    setBusy("run"); setError("");
    try {
      const out = await api<{ id: string }>("/runs", "POST", { scenario });
      await reload();
      setEvents([]); setDetail(null); setRunId(out.id);
    } catch (reason) { guard(reason); } finally { setBusy(null); }
  };

  const decide = async (decision: Decision, approve: boolean) => {
    setBusy(decision.id);
    try { await api(`/decisions/${decision.id}`, "POST", { approve }); await reload(); if (runId) await loadDetail(runId); } catch (reason) { guard(reason); } finally { setBusy(null); }
  };

  const shown = picked === "all" ? events : events.filter((event) => event.agent === picked);
  const pending = detail?.decisions.filter((item) => item.status === "pending") ?? [];

  return (
    <>
      <div className="bar">
        <div><h1>Live run</h1><p>Watch the three agents work: every SQL query, tool call and decision, as it happens.</p></div>
        <div className="bar-actions">
          <select aria-label="Scenario" value={scenario} onChange={(event) => setScenario(event.target.value as "baseline" | "sale_surge")} disabled={running}><option value="baseline">Normal trading</option><option value="sale_surge">What-if: big sale +35%</option></select>
          <button className="btn big" disabled={running || busy === "run"} onClick={() => void start()}>{running || busy === "run" ? <LoaderCircle className="spin" size={16} /> : <Play size={15} fill="currentColor" />} {running ? "Agents running..." : "Run agents"}</button>
        </div>
      </div>
      {error && <div className="alert" role="alert"><span>{error}</span><button className="icon" aria-label="Dismiss" onClick={() => setError("")}><X size={15} /></button></div>}
      {run?.status === "failed" || run?.status === "interrupted" ? <div className="alert"><span>{run.error ?? "This run was interrupted."}</span><button className="btn ghost" onClick={() => void api(`/runs/${run.id}/resume`, "POST").then(reload).catch(guard)}><RefreshCw size={14} /> Resume from checkpoint</button></div> : null}
      {!run ? <div className="empty"><Sparkles size={30} /><h3>No runs yet</h3><p>Press Run agents to start the first run.</p></div> : (
        <div className="live">
          <section className="panel graph">
            <header><span><GitBranch size={14} /> State graph</span><span className={`pill ${running ? "live" : run.status}`}>{running ? (stream === "reconnecting" ? "Reconnecting" : "Live") : run.status}</span></header>
            <div className="graph-body">
              <Node id="supervisor" icon={GitBranch} sub="Scope, plan, delegate" events={events} picked={picked} onPick={setPicked} />
              <i className="wire" />
              <Node id="demand_forecaster" icon={Activity} sub="Forecast 14 days of demand" events={events} picked={picked} onPick={setPicked} />
              <i className="wire split" />
              <div className="pair">
                <Node id="inventory_optimizer" icon={Boxes} sub="Stockout risk and restock" events={events} picked={picked} onPick={setPicked} />
                <Node id="dynamic_pricer" icon={TrendingUp} sub="Elasticity and price" events={events} picked={picked} onPick={setPicked} />
              </div>
              <i className="wire join" />
              <Node id="policy_gate" icon={ShieldCheck} sub="Re-check hard limits" events={events} picked={picked} onPick={setPicked} />
              <i className="wire" />
              <Node id="risk_critic" icon={AlertTriangle} sub="Escalate material decisions" events={events} picked={picked} onPick={setPicked} />
            </div>
            <footer><button className={`chip ${picked === "all" ? "on" : ""}`} onClick={() => setPicked("all")}>Show everything</button><span>Select a node to filter the stream</span></footer>
          </section>
          <section className="panel stream">
            <header><span><Database size={14} /> Execution stream</span><span>{shown.length} events</span></header>
            {shown.length ? <ol ref={logRef} className="events">{shown.map((event) => <EventRow key={event.seq} event={event} />)}</ol> : <div className="empty small"><p>{running ? "Waiting for the first event..." : "No events for this selection."}</p></div>}
          </section>
        </div>
      )}
      {detail && (
        <section className="decisions">
          <div className="bar tight"><div><h2>Decisions from this run</h2><p>{detail.decisions.length} proposed. {pending.length} waiting for you. Each cites the evidence that produced it.</p></div></div>
          {detail.metrics.length > 0 && <div className="metrics">{detail.metrics.map((metric) => <div key={metric.agent}><b>{AGENTS[metric.agent]}</b><span className={`pill ${metric.provider === "Gemini" ? "applied" : "pending"}`}>{metric.provider === "Gemini" ? "Gemini tool agent" : "Local policy"}</span><small>{metric.model_calls} model calls · {metric.tool_calls} tool calls · {metric.sql_queries} SQL · {(metric.duration_ms / 1000).toFixed(1)}s{metric.completed_locally ? " · fell back to local policy" : ""}</small>{metric.rationale && <p>{metric.rationale}</p>}</div>)}</div>}
          {detail.decisions.length ? <div className="dec-grid">{detail.decisions.map((decision) => <DecisionCard key={decision.id} decision={decision} busy={busy === decision.id} onDecide={(approve) => void decide(decision, approve)} />)}</div> : <div className="empty small"><p>{running ? "Decisions appear when the agents finish." : "No decisions were needed in this run."}</p></div>}
        </section>
      )}
    </>
  );
}
