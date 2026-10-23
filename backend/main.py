import asyncio
import csv
import io
import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from dotenv import load_dotenv

load_dotenv()
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend import db as dbmod, performance, security
from backend.db import Database
from backend.service import Runner, decide
from backend.sqltools import jsonable

logger = logging.getLogger(__name__)
database: Database | None = None
MAX_RUNS_PER_HOUR = int(os.getenv("MAX_RUNS_PER_HOUR", "20"))


def get_db() -> Database:
    global database
    if database is None:
        database = Database.from_env()
    return database


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_format = os.getenv("LOG_FORMAT", "text")
    if log_format == "json":
        import json as _json

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                return _json.dumps({
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }, default=str)

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logging.root.handlers = [handler]
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    db = get_db()
    db.init_schema(os.getenv("AGENT_DB_PASSWORD"))
    dbmod.seed(db)
    with db.begin() as conn:
        conn.execute(text("UPDATE agent_runs SET status = 'interrupted' WHERE status = 'running'"))
    app.state.runner = Runner(db)
    if not security.seller_configured():
        logger.warning("AUTH DISABLED: SELLER_ACCESS_CODE or SESSION_SECRET not configured. All endpoints are open.")
    yield
    await asyncio.to_thread(app.state.runner.close)


DOCS = os.getenv("ENABLE_DOCS") == "1"
app = FastAPI(title="Agent operations", lifespan=lifespan, docs_url="/docs" if DOCS else None, redoc_url=None, openapi_url="/openapi.json" if DOCS else None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=security.allowed_hosts())


@app.middleware("http")
async def protect(request: Request, call_next):
    if request.method in security.UNSAFE_METHODS:
        length = request.headers.get("content-length", "")
        if "chunked" in request.headers.get("transfer-encoding", "").lower() or (length and (not length.isdigit() or int(length) > security.MAX_BODY_BYTES)):
            return JSONResponse({"detail": "Request too large"}, 413)
        origin = request.headers.get("origin")
        origins = security.allowed_origins()
        # When frontend is served from the same origin (production), allow same-origin requests
        if not origins:
            scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
            host = request.headers.get("host", "")
            origins = {f"{scheme}://{host}"} if host else set()
        if (origin is not None and origin not in origins) or (origin is None and request.cookies.get(security.COOKIE)):
            return JSONResponse({"detail": "Request not accepted"}, 403)
    response = await call_next(request)
    for name, value in security.SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    else:
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' https://cdn.dummyjson.com data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    return response


@app.exception_handler(RequestValidationError)
async def invalid(request: Request, error: RequestValidationError):
    return JSONResponse({"detail": "Please check the details and try again"}, 422)


@app.exception_handler(Exception)
async def unexpected(request: Request, error: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Something went wrong. Please try again."}, 500)


def operator(request: Request):
    pass  # Auth removed for public demo


class Login(BaseModel):
    code: str = Field(min_length=1, max_length=200)


class RunRequest(BaseModel):
    goal: str | None = Field(default=None, max_length=300)
    scenario: Literal["baseline", "sale_surge"] = "sale_surge"
    product_ids: list[str] = Field(default_factory=list, max_length=16)


class Approval(BaseModel):
    approve: bool


class Clock(BaseModel):
    days: int = Field(ge=1, le=30)


class Preference(BaseModel):
    auto_approve: bool


@app.get("/api/health")
def health():
    try:
        with get_db().engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        return JSONResponse({"status": "degraded", "db": "unreachable"}, 503)


@app.get("/api/session")
def session(request: Request):
    return {"signedIn": True, "available": True}  # Auth removed for public demo


@app.post("/api/login")
def login(body: Login, request: Request, response: Response):
    if not security.seller_configured():
        raise HTTPException(503, "Sign-in is not set up on this server")
    if not security.limiter.allow("login", request.client.host if request.client else "?", 5, 300):
        raise HTTPException(429, "Too many attempts. Try again in a few minutes.")
    if not security.check_access_code(body.code):
        raise HTTPException(401, "That access code isn't right")
    response.set_cookie(security.COOKIE, security.issue_token(), max_age=security.SESSION_SECONDS, httponly=True, samesite="strict", secure=request.url.scheme == "https" or os.getenv("COOKIE_SECURE") == "1", path="/api")
    return {"signedIn": True}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    security.revoke_token(request.cookies.get(security.COOKIE))
    response.delete_cookie(security.COOKIE, path="/api")
    return {"signedIn": False}


def _rows(db: Database, query: str, **params) -> list[dict]:
    with db.engine.connect() as conn:
        return jsonable([dict(r) for r in conn.execute(text(query), params).mappings()])


def _rows_conn(conn, query: str, **params) -> list[dict]:
    return jsonable([dict(r) for r in conn.execute(text(query), params).mappings()])


_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)


def _valid_uuid(value: str) -> str:
    if not _UUID_RE.match(value):
        raise HTTPException(400, "Invalid identifier")
    return value


@app.get("/api/overview", dependencies=[Depends(operator)])
def overview():
    db = get_db()
    with db.engine.connect() as conn:
        products = _rows_conn(conn, "SELECT p.id, p.name, p.category, p.price::float, p.reference_price::float, p.cost::float, p.lead_days, p.image, i.on_hand, "
                             "COALESCE((SELECT sum(qty) FROM purchase_orders o WHERE o.product_id = p.id AND o.status = 'open'), 0)::int AS on_order, "
                             "(SELECT COALESCE(sum(units), 0) FROM sales_daily s WHERE s.product_id = p.id AND s.day > (SELECT today FROM sim_state) - 7)::int AS sold_7d "
                             "FROM products p JOIN inventory i ON i.product_id = p.id ORDER BY p.id")
        runs = _rows_conn(conn, "SELECT id, scenario, goal, status, sim_day, execution_mode, started_at, finished_at, error, "
                         "(SELECT count(*) FROM decisions d WHERE d.run_id = r.id)::int AS decisions FROM agent_runs r ORDER BY started_at DESC LIMIT 25")
        pending = _rows_conn(conn, "SELECT count(*)::int AS n FROM decisions WHERE status = 'pending'")[0]["n"]
    return {"products": products, "runs": runs, "pending": pending, "market_day": db.today().isoformat(), "auto_approve": bool(db.setting("auto_approve", False)), "gemini": bool(os.getenv("GEMINI_API_KEY"))}


@app.get("/api/runs/{run_id}", dependencies=[Depends(operator)])
def run_detail(run_id: str):
    run_id = _valid_uuid(run_id)
    db = get_db()
    with db.engine.connect() as conn:
        head = _rows_conn(conn, "SELECT id, scenario, goal, status, sim_day, execution_mode, started_at, finished_at, error FROM agent_runs WHERE id = CAST(:i AS uuid)", i=run_id)
        if not head:
            raise HTTPException(404, "Not found")
        return {
            "run": head[0],
            "metrics": _rows_conn(conn, "SELECT agent, provider, model_calls, tool_calls, sql_queries, duration_ms::float, completed_locally, rationale FROM agent_metrics WHERE run_id = CAST(:i AS uuid) ORDER BY agent", i=run_id),
            "decisions": _rows_conn(conn, "SELECT d.id, d.product_id, p.name AS product_name, d.agent, d.kind, d.before_value, d.after_value, d.quantity, d.reason, d.evidence, d.needs_approval, d.status, d.created_at "
                                   "FROM decisions d JOIN products p ON p.id = d.product_id WHERE d.run_id = CAST(:i AS uuid) ORDER BY d.created_at, d.kind", i=run_id),
            "forecasts": _rows_conn(conn, "SELECT product_id, target_day, mean, lower, upper, baseline FROM forecasts WHERE run_id = CAST(:i AS uuid) ORDER BY product_id, target_day", i=run_id),
            "stockout": _rows_conn(conn, "SELECT product_id, made_on, window_days, p_stockout, p_before_resupply, expected_day, days_cover, on_hand, incoming FROM stockout_predictions WHERE run_id = CAST(:i AS uuid) ORDER BY p_before_resupply DESC", i=run_id),
            "elasticity": _rows_conn(conn, "SELECT product_id, elasticity, std_err, n_obs, r2, usable, reason, curve FROM elasticity_estimates WHERE run_id = CAST(:i AS uuid) ORDER BY product_id", i=run_id),
            "quality": _rows_conn(conn, "SELECT product_id, wape::float, baseline_wape::float, days_used FROM forecast_quality WHERE run_id = CAST(:i AS uuid) ORDER BY product_id", i=run_id),
        }


@app.get("/api/runs/{run_id}/events", dependencies=[Depends(operator)])
async def events(run_id: str, request: Request):
    run_id = _valid_uuid(run_id)
    db = get_db()
    try:
        after = max(0, int(request.headers.get("last-event-id", "0") or 0))
    except ValueError:
        after = 0
    if not _rows(db, "SELECT 1 AS x FROM agent_runs WHERE id = CAST(:i AS uuid)", i=run_id):
        raise HTTPException(404, "Not found")

    async def stream():
        cursor = after
        while True:
            if await request.is_disconnected():
                return
            rows = await asyncio.to_thread(_rows, db, "SELECT seq, agent, kind, title, detail, ts FROM agent_events WHERE run_id = CAST(:i AS uuid) AND seq > :s ORDER BY seq", i=run_id, s=cursor)
            for row in rows:
                cursor = row["seq"]
                yield f"id: {cursor}\nevent: agent\ndata: {json.dumps(row)}\n\n"
            status = (await asyncio.to_thread(_rows, db, "SELECT status FROM agent_runs WHERE id = CAST(:i AS uuid)", i=run_id))[0]["status"]
            if status != "running" and not rows:
                yield f"event: done\ndata: {json.dumps({'status': status})}\n\n"
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{run_id}/events/all", dependencies=[Depends(operator)])
def events_all(run_id: str):
    run_id = _valid_uuid(run_id)
    return _rows(get_db(), "SELECT seq, agent, kind, title, detail, ts FROM agent_events WHERE run_id = CAST(:i AS uuid) ORDER BY seq", i=run_id)


@app.post("/api/runs", status_code=202, dependencies=[Depends(operator)])
def start(body: RunRequest):
    if not security.limiter.allow("runs", "global", MAX_RUNS_PER_HOUR, 3600):
        raise HTTPException(429, "Hourly run limit reached")
    try:
        return {"id": app.state.runner.start(body.goal, body.scenario, body.product_ids)}
    except PermissionError as error:
        raise HTTPException(422, str(error)) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/runs/{run_id}/resume", status_code=202, dependencies=[Depends(operator)])
def resume(run_id: str):
    run_id = _valid_uuid(run_id)
    try:
        return {"id": app.state.runner.start(None, "baseline", [], resume_id=run_id)}
    except (RuntimeError, ValueError) as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/decisions/{decision_id}", dependencies=[Depends(operator)])
def decision(decision_id: str, body: Approval):
    decision_id = _valid_uuid(decision_id)
    try:
        return jsonable(decide(get_db(), decision_id, body.approve))
    except KeyError as error:
        raise HTTPException(404, "Not found") from error


@app.patch("/api/preferences", dependencies=[Depends(operator)])
def preferences(body: Preference):
    get_db().set_setting("auto_approve", body.auto_approve)
    return {"auto_approve": body.auto_approve}


@app.post("/api/market/advance", dependencies=[Depends(operator)])
def advance(body: Clock):
    if app.state.runner.lock.locked():
        raise HTTPException(409, "A run is in progress")
    return {"market_day": dbmod.advance(get_db(), body.days).isoformat()}


@app.get("/api/performance", dependencies=[Depends(operator)])
def agent_performance():
    return jsonable(performance.performance(get_db()))


@app.get("/api/decisions/export", dependencies=[Depends(operator)])
def export():
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["created_at", "agent", "product", "kind", "before", "after", "quantity", "status", "reason"])
    for row in _rows(get_db(), "SELECT d.created_at, d.agent, p.name, d.kind, d.before_value, d.after_value, d.quantity, d.status, d.reason FROM decisions d JOIN products p ON p.id = d.product_id ORDER BY d.created_at DESC LIMIT 5000"):
        writer.writerow([("'" + str(v)) if isinstance(v, str) and v[:1] in "=+-@\t\r" else v for v in row.values()])
    return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=decisions.csv"})


# ---------------------------------------------------------------------------
# Serve frontend static files in production (when frontend/dist exists)
# ---------------------------------------------------------------------------
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIR.is_dir():
    # Serve static assets (JS, CSS, images) from /assets
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIR / "assets"), name="frontend-assets")

    # SPA catch-all: any non-API route serves index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path:
            file = (_FRONTEND_DIR / full_path).resolve()
            if file.is_relative_to(_FRONTEND_DIR) and file.is_file():
                return FileResponse(file)
        return FileResponse(_FRONTEND_DIR / "index.html")
