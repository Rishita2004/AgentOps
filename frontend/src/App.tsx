import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Activity, Database, LayoutGrid, LoaderCircle, PackageCheck, Radio, ShieldCheck, Workflow } from "lucide-react";
import LiveRun from "./LiveRun";
const PerformanceView = lazy(() => import("./PerformanceView"));
import { api, ApiError, rupees } from "./lib";
import type { Overview } from "./lib";

type Tab = "Live run" | "Performance" | "Data";

function DataView({ overview }: { overview: Overview }) {
  return (
    <>
      <div className="bar"><div><h1>Data</h1><p>The live PostgreSQL state the agents query. Prices and orders here change only when a decision is approved.</p></div></div>
      <section className="panel table">
        <table>
          <caption style={{ position: 'absolute', width: 1, height: 1, overflow: 'hidden', clip: 'rect(0,0,0,0)' }}>Product inventory and pricing data</caption>
          <thead><tr><th>Product</th><th>Price</th><th>Cost</th><th>On hand</th><th>On order</th><th>Lead time</th><th>Sold, 7 days</th></tr></thead>
          <tbody>
            {overview.products.map((product) => (
              <tr key={product.id}>
                <td><div className="cell">{product.image && <img src={product.image} alt={product.name} loading="lazy" />}<div><b>{product.name}</b><small>{product.id} · {product.category}</small></div></div></td>
                <td>{rupees(product.price)}{product.price !== product.reference_price && <small className="delta">list {rupees(product.reference_price)}</small>}</td>
                <td>{rupees(product.cost)}</td>
                <td className={product.on_hand < product.sold_7d ? "low" : ""}>{product.on_hand}</td>
                <td>{product.on_order}</td>
                <td>{product.lead_days} days</td>
                <td>{product.sold_7d}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <p className="note"><Database size={13} /> Sales history, stock and purchase orders live in PostgreSQL. The agents read them through a read-only database role and cannot write; only approved decisions do.</p>
    </>
  );
}

export default function App() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [tab, setTab] = useState<Tab>("Live run");
  const [error, setError] = useState("");

  const reload = useCallback(async () => {
    try { setOverview(await api<Overview>("/overview")); setError(""); } catch (reason) { setError(reason instanceof ApiError ? reason.message : "Something went wrong."); }
  }, []);

  const running = overview?.runs[0]?.status === "running";
  useEffect(() => {
    let live = true;
    const tick = () => api<Overview>("/overview").then((value) => { if (live) { setOverview(value); setError(""); } }).catch((reason) => {
      if (!live) return;
      setError(reason instanceof ApiError ? reason.message : "Something went wrong.");
    });
    void tick();
    const timer = window.setInterval(() => void tick(), running ? 2000 : 8000);
    return () => { live = false; window.clearInterval(timer); };
  }, [running]);

  return (
    <div className="app">
      <header className="top">
        <div className="top-in">
          <div className="brand"><Workflow size={19} /><b>Agent Operations</b><span className="env">PostgreSQL · LangGraph · {overview?.gemini ? "Gemini" : "local models"}</span></div>
          <nav aria-label="Sections" role="tablist">
            {([["Live run", Radio], ["Performance", Activity], ["Data", LayoutGrid]] as const).map(([name, Icon]) => (
              <button key={name} role="tab" className={tab === name ? "on" : ""} onClick={() => setTab(name)} aria-current={tab === name ? "page" : undefined}><Icon size={15} /> {name}{name === "Live run" && overview && overview.pending > 0 && <i>{overview.pending}</i>}</button>
            ))}
          </nav>
          <div className="top-end">
            {overview && <label className="auto" title="Routine decisions are applied automatically. Material ones always wait for you."><ShieldCheck size={14} /> Auto-approve routine<input type="checkbox" checked={overview.auto_approve} onChange={(event) => void api("/preferences", "PATCH", { auto_approve: event.target.checked }).then(reload)} aria-label="Auto-approve routine decisions" /><span /></label>}
          </div>
        </div>
      </header>
      <main className="main">
        {error && <div className="alert" role="alert"><span>{error}</span></div>}
        {!overview ? <div className="loading"><LoaderCircle className="spin" size={26} /></div> : tab === "Live run" ? <LiveRun overview={overview} reload={reload} /> : tab === "Performance" ? <Suspense fallback={<div className="loading"><LoaderCircle className="spin" size={26} /></div>}><PerformanceView overview={overview} reload={reload} /></Suspense> : <DataView overview={overview} />}
        <footer className="foot"><PackageCheck size={12} /> Demo market: sales, prices and orders are simulated. Decisions change this database only; nothing is sent to any marketplace.</footer>
      </main>
    </div>
  );
}
