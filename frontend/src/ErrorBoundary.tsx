import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props { children: ReactNode; }
interface State { error: Error | null; }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Unhandled render error:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: "2rem", maxWidth: 600, margin: "4rem auto", fontFamily: "system-ui, sans-serif" }}>
          <h1 style={{ fontSize: "1.25rem", marginBottom: "0.75rem" }}>Something went wrong</h1>
          <p style={{ color: "#64748b", marginBottom: "1rem" }}>The dashboard hit an unexpected error. Try reloading the page.</p>
          <pre style={{ background: "#f1f5f9", padding: "1rem", borderRadius: 8, fontSize: "0.8rem", overflow: "auto", maxHeight: 200 }}>{this.state.error.message}</pre>
          <button onClick={() => window.location.reload()} style={{ marginTop: "1rem", padding: "0.5rem 1rem", background: "#0f172a", color: "#fff", border: "none", borderRadius: 6, cursor: "pointer" }}>Reload page</button>
        </div>
      );
    }
    return this.props.children;
  }
}
