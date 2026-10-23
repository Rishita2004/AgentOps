"""Numerical models used by the agents. Pure functions over arrays; no I/O."""

import math

import numpy as np
from sklearn.linear_model import Ridge

HORIZON = 14
MIN_MARGIN = 0.25
MAX_PRICE_MOVE = 0.05
MAX_REORDER = 2000
SERVICE_Z = 1.65


def _design(days: np.ndarray, log_price: np.ndarray, t0: int) -> np.ndarray:
    angle = 2 * np.pi * (days % 7) / 7
    return np.column_stack((np.ones(len(days)), (days - t0) / 90, np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle), log_price))


def _prepare(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    days = np.array([row["day"].toordinal() for row in rows], dtype=float)
    units = np.array([row["units"] for row in rows], dtype=float)
    price = np.array([row["avg_price"] for row in rows], dtype=float)
    ok = ~np.array([bool(row["stockout"]) for row in rows])
    return days, units, price, ok


def forecast_demand(rows: list[dict], current_price: float, reference_price: float, horizon: int = HORIZON, multiplier: float = 1.0) -> dict:
    """Ridge on trend, weekly cycle and log price. Stock-out days are censored and carry zero weight."""
    if len(rows) < 28:
        raise ValueError("At least 28 days of sales are needed")
    days, units, price, ok = _prepare(rows)
    if ok.sum() < 21 or not np.isfinite(units).all() or (units < 0).any():
        raise ValueError("Not enough uncensored sales days")
    t0 = int(days[0])
    log_price = np.log(price / reference_price)
    x = _design(days, log_price, t0)[:, 1:]
    split = len(days) - 14
    train = np.arange(len(days)) < split
    valid = ~train & ok
    check = Ridge(alpha=0.5).fit(x[train], units[train], sample_weight=ok[train].astype(float) + 1e-9)
    predicted = np.maximum(0, check.predict(x[valid]))
    actual = units[valid]
    wape = float(np.abs(actual - predicted).sum() / max(1.0, actual.sum()))
    recent = units[train & ok][-7:]
    baseline = np.full(valid.sum(), recent.mean() if len(recent) else 0.0)
    baseline_wape = float(np.abs(actual - baseline).sum() / max(1.0, actual.sum()))
    model = Ridge(alpha=0.5).fit(x, units, sample_weight=ok.astype(float) + 1e-9)
    residual = (units - model.predict(x))[ok]
    spread = max(0.5, float(residual.std()))
    future = np.arange(days[-1] + 1, days[-1] + 1 + horizon)
    fx = _design(future, np.full(horizon, math.log(current_price / reference_price)), t0)[:, 1:]
    mean = np.maximum(0, model.predict(fx)) * multiplier
    last7 = units[-7:][ok[-7:]]
    naive = float(last7.mean()) if len(last7) else float(units[ok].mean())
    return {
        "daily": [round(float(v), 2) for v in mean],
        "lower": [round(max(0.0, float(v) - 1.96 * spread * multiplier), 2) for v in mean],
        "upper": [round(float(v) + 1.96 * spread * multiplier, 2) for v in mean],
        "baseline": [round(naive, 2)] * horizon,
        "wape": round(wape, 4),
        "baseline_wape": round(baseline_wape, 4),
        "residual_std": round(spread, 3),
        "overdispersion": _dispersion(units[ok], model.predict(x)[ok]),
        "days_used": int(ok.sum()),
    }


def _dispersion(units: np.ndarray, fitted: np.ndarray) -> float:
    mu = np.maximum(fitted, 0.1)
    excess = float(((units - mu) ** 2 - mu).sum() / max(1e-9, (mu**2).sum()))
    return round(max(0.0, excess), 4)


def estimate_elasticity(rows: list[dict], reference_price: float) -> dict:
    """Poisson regression (log link) of daily units on log price, trend and weekday. Stock-out days are censored and dropped."""
    days, units, price, ok = _prepare(rows)
    days, units, price = days[ok], units[ok], price[ok]
    n = len(days)
    result = {"n_obs": int(n), "elasticity": None, "std_err": None, "r2": None, "price_cv": None, "usable": False, "reason": "Not enough sales days"}
    if n < 30:
        return result
    log_price = np.log(price / reference_price)
    cv = float(log_price.std())
    result["price_cv"] = round(cv, 4)
    if cv < 0.02:
        result["reason"] = "Price barely changed, so the effect of price can't be measured"
        return result
    x = _design(days, log_price, int(days[0]))
    beta = np.array([math.log(max(units.mean(), 0.1))] + [0.0] * (x.shape[1] - 1))
    for _ in range(40):
        mu = np.exp(np.clip(x @ beta, -20, 20))
        info = x.T @ (x * mu[:, None]) + 1e-8 * np.eye(x.shape[1])
        step = np.linalg.solve(info, x.T @ (units - mu))
        beta = beta + step
        if float(np.abs(step).max()) < 1e-9:
            break
    mu = np.exp(np.clip(x @ beta, -20, 20))
    info = x.T @ (x * mu[:, None]) + 1e-8 * np.eye(x.shape[1])
    pearson = float(((units - mu) ** 2 / np.maximum(mu, 1e-9)).sum()) / max(1, n - x.shape[1])
    cov = max(1.0, pearson) * np.linalg.inv(info)
    e, se = float(beta[-1]), float(math.sqrt(max(cov[-1, -1], 0.0)))
    null = np.full(n, units.mean())
    dev = lambda m: 2 * float(np.sum(np.where(units > 0, units * np.log(np.maximum(units, 1e-9) / m), 0.0) - (units - m)))
    result.update({"elasticity": round(e, 3), "std_err": round(se, 3), "r2": round(max(0.0, 1 - dev(mu) / dev(null)), 3) if dev(null) > 0 else 0.0})
    usable = -6.0 < e and e + 1.96 * se < -0.2
    result["usable"] = bool(usable)
    result["reason"] = "Reliable estimate" if usable else "Estimate is too uncertain to act on"
    return result


def plan_price(current: float, cost: float, base_units: float, elasticity: dict, supply_per_day: float | None = None) -> dict:
    """Profit-maximising price inside hard limits, using the estimated elasticity. Holds the price when evidence is weak."""
    floor = math.ceil(max(cost / (1 - MIN_MARGIN), current * (1 - MAX_PRICE_MOVE)) * 100 - 1e-8) / 100
    ceiling = math.floor(current * (1 + MAX_PRICE_MOVE) * 100 + 1e-8) / 100
    e, se = elasticity.get("elasticity"), elasticity.get("std_err") or 0.0
    out = {"price": current, "held": True, "predicted_units": round(base_units, 2), "curve": [], "reason": elasticity.get("reason", "")}
    if e is None:
        return out
    for point in np.linspace(current * 0.8, current * 1.2, 25):
        p = float(point)
        row = {"price": round(p, 2), "demand": round(base_units * (p / current) ** e, 2)}
        row["lower"] = round(base_units * (p / current) ** (e - 1.96 * se), 2)
        row["upper"] = round(base_units * (p / current) ** (e + 1.96 * se), 2)
        out["curve"].append(row)
    if floor > ceiling:
        out["reason"] = "Margin floor conflicts with the 5% price limit"
        return out
    if not elasticity.get("usable"):
        return out

    def profit(p: float) -> float:
        units = base_units * (p / current) ** e
        if supply_per_day is not None:
            units = min(units, supply_per_day)
        return (p - cost) * units

    chosen = round(float(max(np.linspace(floor, ceiling, 41), key=profit)), 2)
    if abs(chosen - current) < 0.01:
        out["reason"] = "Current price is already best within the limits"
        return out
    out.update({"price": chosen, "held": False, "predicted_units": round(base_units * (chosen / current) ** e, 2), "reason": "Higher profit within the 5% and 25% margin limits"})
    return out


def stockout_risk(stock: int, arrivals: list[tuple[int, int]], daily_mean: list[float], lead_days: int, overdispersion: float, draws: int = 4000, seed: int = 7) -> dict:
    """Monte Carlo over daily demand. arrivals are (day offset from 1, quantity)."""
    horizon = len(daily_mean)
    rng = np.random.default_rng(seed)
    mu = np.maximum(np.array(daily_mean, dtype=float), 0.01)
    if overdispersion > 1e-3:
        shape = 1.0 / overdispersion
        lam = rng.gamma(shape, mu / shape, size=(draws, horizon))
    else:
        lam = np.tile(mu, (draws, 1))
    demand = rng.poisson(lam)
    supply = np.full((draws, horizon), 0.0)
    for offset, qty in arrivals:
        if 1 <= offset <= horizon:
            supply[:, offset - 1:] += qty
    cumulative = np.cumsum(demand, axis=1)
    short = cumulative > (stock + supply)
    any_short = short.any(axis=1)
    first = np.where(any_short, short.argmax(axis=1) + 1, horizon + 1)
    first_arrival = min((offset for offset, _ in arrivals), default=lead_days)
    window = max(1, min(horizon, max(lead_days, 1) if not arrivals else first_arrival))
    return {
        "p_stockout": round(float(any_short.mean()), 4),
        "p_before_resupply": round(float((first <= window).mean()), 4),
        "expected_day": round(float(np.median(first)), 1),
        "days_cover": round(stock / float(mu.mean()), 1),
        "window_days": int(window),
    }


def plan_reorder(stock: int, on_order: int, daily_mean: list[float], spread: float, lead_days: int, risk: dict) -> dict:
    mu = float(np.mean(daily_mean))
    span = lead_days + HORIZON
    safety = SERVICE_Z * max(spread, math.sqrt(mu)) * math.sqrt(span)
    target = math.ceil(mu * span + safety)
    position = stock + on_order
    quantity = 0
    if risk["p_before_resupply"] > 0.10 or position < mu * lead_days + SERVICE_Z * max(spread, math.sqrt(mu)) * math.sqrt(lead_days):
        quantity = int(min(MAX_REORDER, max(0, target - position)))
    return {"reorder": quantity, "target_position": target, "expedite": bool(on_order > 0 and risk["p_before_resupply"] > 0.5)}
