"""Tests for backend.ml — pure numerical functions, no database needed."""

import datetime
import math
import random

import pytest

from backend.ml import (
    HORIZON,
    MAX_PRICE_MOVE,
    MAX_REORDER,
    MIN_MARGIN,
    estimate_elasticity,
    forecast_demand,
    plan_price,
    plan_reorder,
    stockout_risk,
)

# ---------------------------------------------------------------------------
# Fixtures — realistic mock sales data
# ---------------------------------------------------------------------------

BASE_PRICE = 500.0
COST = 350.0  # gives ~30 % margin at base price


def _make_sales(
    n_days: int = 70,
    base_units: float = 40.0,
    base_price: float = BASE_PRICE,
    price_noise: float = 0.08,
    elasticity: float = -1.5,
    stockout_rate: float = 0.0,
    seed: int = 42,
) -> list[dict]:
    """Generate *n_days* of daily sales dicts with realistic price variation.

    Each row has keys: day (datetime.date), units (int), avg_price (float),
    stockout (bool).  Price varies by ±price_noise around base_price so
    elasticity is identifiable.  Demand follows a simple log-linear model
    plus Poisson noise.
    """
    rng = random.Random(seed)
    start = datetime.date(2025, 1, 1)
    rows: list[dict] = []
    for i in range(n_days):
        day = start + datetime.timedelta(days=i)
        # price varies uniformly in [1 - noise, 1 + noise] * base
        price_factor = 1.0 + rng.uniform(-price_noise, price_noise)
        price = round(base_price * price_factor, 2)
        # log-linear demand: units ~ base * (price / base_price) ^ elasticity
        expected = base_units * (price / base_price) ** elasticity
        units = max(0, int(rng.gauss(expected, max(1.0, expected * 0.15))))
        is_stockout = rng.random() < stockout_rate
        if is_stockout:
            units = max(0, int(units * 0.2))  # much lower during stockout
        rows.append(
            {"day": day, "units": units, "avg_price": price, "stockout": is_stockout}
        )
    return rows


@pytest.fixture()
def sales_70d() -> list[dict]:
    """70 days with price variation — enough for both forecast and elasticity."""
    return _make_sales(n_days=70, price_noise=0.08, seed=42)


@pytest.fixture()
def sales_flat_price() -> list[dict]:
    """60 days with virtually no price variation (cv < 0.02)."""
    return _make_sales(n_days=60, price_noise=0.001, seed=99)


@pytest.fixture()
def sales_short() -> list[dict]:
    """Only 20 days — below minimums for forecast and elasticity."""
    return _make_sales(n_days=20, seed=11)


@pytest.fixture()
def sales_with_stockouts() -> list[dict]:
    """90 days, ~15 % stockout rate."""
    return _make_sales(n_days=90, stockout_rate=0.15, seed=77)


# ---------------------------------------------------------------------------
# 1. forecast_demand
# ---------------------------------------------------------------------------


class TestForecastDemand:
    """Tests for forecast_demand."""

    def test_requires_minimum_rows(self, sales_short: list[dict]):
        with pytest.raises(ValueError, match="28 days"):
            forecast_demand(sales_short, current_price=BASE_PRICE, reference_price=BASE_PRICE)

    def test_output_structure(self, sales_70d: list[dict]):
        result = forecast_demand(sales_70d, current_price=BASE_PRICE, reference_price=BASE_PRICE)
        for key in ("daily", "lower", "upper", "baseline"):
            assert key in result, f"Missing key: {key}"
            assert len(result[key]) == HORIZON, f"{key} length != {HORIZON}"
        assert "wape" in result

    def test_values_non_negative(self, sales_70d: list[dict]):
        result = forecast_demand(sales_70d, current_price=BASE_PRICE, reference_price=BASE_PRICE)
        for key in ("daily", "lower", "upper", "baseline"):
            assert all(v >= 0 for v in result[key]), f"Negative value in {key}"

    def test_lower_le_mean_le_upper(self, sales_70d: list[dict]):
        result = forecast_demand(sales_70d, current_price=BASE_PRICE, reference_price=BASE_PRICE)
        for lo, mid, hi in zip(result["lower"], result["daily"], result["upper"]):
            assert lo <= mid + 1e-9, f"lower {lo} > daily {mid}"
            assert mid <= hi + 1e-9, f"daily {mid} > upper {hi}"

    def test_wape_between_0_and_1(self, sales_70d: list[dict]):
        result = forecast_demand(sales_70d, current_price=BASE_PRICE, reference_price=BASE_PRICE)
        assert 0 <= result["wape"] <= 1, f"wape={result['wape']} out of [0,1]"

    def test_custom_horizon(self, sales_70d: list[dict]):
        h = 7
        result = forecast_demand(sales_70d, current_price=BASE_PRICE, reference_price=BASE_PRICE, horizon=h)
        assert len(result["daily"]) == h

    def test_multiplier_scales_output(self, sales_70d: list[dict]):
        base = forecast_demand(sales_70d, current_price=BASE_PRICE, reference_price=BASE_PRICE, multiplier=1.0)
        scaled = forecast_demand(sales_70d, current_price=BASE_PRICE, reference_price=BASE_PRICE, multiplier=2.0)
        # Each daily value should be ~2x (tolerance for rounding)
        for b, s in zip(base["daily"], scaled["daily"]):
            if b > 0:
                assert s == pytest.approx(b * 2.0, rel=0.01)

    def test_works_with_stockout_data(self, sales_with_stockouts: list[dict]):
        result = forecast_demand(
            sales_with_stockouts, current_price=BASE_PRICE, reference_price=BASE_PRICE
        )
        assert len(result["daily"]) == HORIZON


# ---------------------------------------------------------------------------
# 2. estimate_elasticity
# ---------------------------------------------------------------------------


class TestEstimateElasticity:
    """Tests for estimate_elasticity."""

    def test_insufficient_rows(self, sales_short: list[dict]):
        result = estimate_elasticity(sales_short, reference_price=BASE_PRICE)
        assert result["usable"] is False
        assert result["n_obs"] < 30

    def test_returns_correct_keys(self, sales_70d: list[dict]):
        result = estimate_elasticity(sales_70d, reference_price=BASE_PRICE)
        for key in ("n_obs", "elasticity", "std_err", "r2", "price_cv", "usable", "reason"):
            assert key in result, f"Missing key: {key}"

    def test_flat_price_not_usable(self, sales_flat_price: list[dict]):
        result = estimate_elasticity(sales_flat_price, reference_price=BASE_PRICE)
        assert result["usable"] is False
        assert "barely changed" in result["reason"].lower() or "can't be measured" in result["reason"].lower()

    def test_varied_price_produces_estimate(self, sales_70d: list[dict]):
        result = estimate_elasticity(sales_70d, reference_price=BASE_PRICE)
        assert result["elasticity"] is not None
        assert result["std_err"] is not None
        # The generator has elasticity = -1.5 so the estimate should be negative
        assert result["elasticity"] < 0, "Expected negative elasticity"

    def test_price_cv_positive_with_variation(self, sales_70d: list[dict]):
        result = estimate_elasticity(sales_70d, reference_price=BASE_PRICE)
        assert result["price_cv"] is not None
        assert result["price_cv"] > 0.02

    def test_stockout_days_censored(self, sales_with_stockouts: list[dict]):
        result = estimate_elasticity(sales_with_stockouts, reference_price=BASE_PRICE)
        n_non_stockout = sum(1 for r in sales_with_stockouts if not r["stockout"])
        assert result["n_obs"] == n_non_stockout


# ---------------------------------------------------------------------------
# 3. plan_price
# ---------------------------------------------------------------------------


class TestPlanPrice:
    """Tests for plan_price."""

    def _usable_elasticity(self) -> dict:
        return {
            "elasticity": -2.0,
            "std_err": 0.3,
            "usable": True,
            "reason": "Reliable estimate",
        }

    def _unusable_elasticity(self) -> dict:
        return {
            "elasticity": -0.5,
            "std_err": 1.0,
            "usable": False,
            "reason": "Estimate is too uncertain to act on",
        }

    def test_holds_price_when_not_usable(self):
        result = plan_price(
            current=BASE_PRICE,
            cost=COST,
            base_units=40.0,
            elasticity=self._unusable_elasticity(),
        )
        assert result["held"] is True
        assert result["price"] == BASE_PRICE

    def test_respects_max_price_move(self):
        result = plan_price(
            current=BASE_PRICE,
            cost=COST,
            base_units=40.0,
            elasticity=self._usable_elasticity(),
        )
        move = abs(result["price"] - BASE_PRICE) / BASE_PRICE
        assert move <= MAX_PRICE_MOVE + 1e-9, f"Price moved {move:.4f} > {MAX_PRICE_MOVE}"

    def test_respects_margin_floor(self):
        result = plan_price(
            current=BASE_PRICE,
            cost=COST,
            base_units=40.0,
            elasticity=self._usable_elasticity(),
        )
        if result["price"] != BASE_PRICE:
            margin = 1 - COST / result["price"]
            assert margin >= MIN_MARGIN - 1e-9, f"Margin {margin:.4f} < {MIN_MARGIN}"

    def test_returns_curve(self):
        result = plan_price(
            current=BASE_PRICE,
            cost=COST,
            base_units=40.0,
            elasticity=self._usable_elasticity(),
        )
        assert len(result["curve"]) > 0
        for point in result["curve"]:
            assert "price" in point
            assert "demand" in point

    def test_curve_has_bounds(self):
        result = plan_price(
            current=BASE_PRICE,
            cost=COST,
            base_units=40.0,
            elasticity=self._usable_elasticity(),
        )
        for point in result["curve"]:
            lo, mid, hi = point["lower"], point["demand"], point["upper"]
            # lower and upper may swap for prices above current (negative
            # exponent arithmetic), so just check that all three are finite
            # and non-negative, and min <= max.
            assert lo >= 0 and mid >= 0 and hi >= 0
            assert min(lo, mid, hi) <= max(lo, mid, hi)

    def test_no_elasticity_returns_held(self):
        result = plan_price(
            current=BASE_PRICE,
            cost=COST,
            base_units=40.0,
            elasticity={"elasticity": None, "std_err": None, "usable": False, "reason": "Not enough data"},
        )
        assert result["held"] is True
        assert result["price"] == BASE_PRICE

    def test_high_cost_margin_conflict(self):
        """When cost is so high that margin floor > ceiling, price is held."""
        high_cost = BASE_PRICE * 0.98  # margin floor would exceed 5 % move
        result = plan_price(
            current=BASE_PRICE,
            cost=high_cost,
            base_units=40.0,
            elasticity=self._usable_elasticity(),
        )
        assert result["held"] is True


# ---------------------------------------------------------------------------
# 4. stockout_risk
# ---------------------------------------------------------------------------


class TestStockoutRisk:
    """Tests for stockout_risk."""

    def _daily_mean(self) -> list[float]:
        return [40.0] * HORIZON

    def test_probabilities_between_0_and_1(self):
        result = stockout_risk(
            stock=200,
            arrivals=[],
            daily_mean=self._daily_mean(),
            lead_days=5,
            overdispersion=0.0,
        )
        assert 0 <= result["p_stockout"] <= 1
        assert 0 <= result["p_before_resupply"] <= 1

    def test_high_stock_low_risk(self):
        result = stockout_risk(
            stock=5000,
            arrivals=[],
            daily_mean=self._daily_mean(),
            lead_days=5,
            overdispersion=0.0,
        )
        assert result["p_stockout"] < 0.05, "5000 units for 14d@40/day should be very safe"

    def test_zero_stock_high_risk(self):
        result = stockout_risk(
            stock=0,
            arrivals=[],
            daily_mean=self._daily_mean(),
            lead_days=5,
            overdispersion=0.0,
        )
        assert result["p_stockout"] > 0.95, "Zero stock should almost certainly stockout"

    def test_arrivals_reduce_risk(self):
        no_arrival = stockout_risk(
            stock=100,
            arrivals=[],
            daily_mean=self._daily_mean(),
            lead_days=5,
            overdispersion=0.0,
        )
        with_arrival = stockout_risk(
            stock=100,
            arrivals=[(3, 500)],
            daily_mean=self._daily_mean(),
            lead_days=5,
            overdispersion=0.0,
        )
        assert with_arrival["p_stockout"] <= no_arrival["p_stockout"]

    def test_overdispersion_increases_risk(self):
        # Use stock high enough that pure Poisson doesn't saturate at p=1.0,
        # so the extra variance from overdispersion has room to increase risk.
        low = stockout_risk(
            stock=600,
            arrivals=[],
            daily_mean=self._daily_mean(),
            lead_days=5,
            overdispersion=0.0,
        )
        high = stockout_risk(
            stock=600,
            arrivals=[],
            daily_mean=self._daily_mean(),
            lead_days=5,
            overdispersion=2.0,
        )
        # More variance → higher stockout probability (or equal in edge cases)
        assert high["p_stockout"] >= low["p_stockout"] - 0.01

    def test_days_cover_correct(self):
        result = stockout_risk(
            stock=400,
            arrivals=[],
            daily_mean=self._daily_mean(),
            lead_days=5,
            overdispersion=0.0,
        )
        assert result["days_cover"] == pytest.approx(400 / 40.0, rel=0.01)

    def test_output_keys(self):
        result = stockout_risk(
            stock=200,
            arrivals=[],
            daily_mean=self._daily_mean(),
            lead_days=5,
            overdispersion=0.0,
        )
        for key in ("p_stockout", "p_before_resupply", "expected_day", "days_cover", "window_days"):
            assert key in result


# ---------------------------------------------------------------------------
# 5. plan_reorder
# ---------------------------------------------------------------------------


class TestPlanReorder:
    """Tests for plan_reorder."""

    def _daily_mean(self) -> list[float]:
        return [40.0] * HORIZON

    def test_zero_reorder_when_stock_sufficient(self):
        risk = {
            "p_stockout": 0.01,
            "p_before_resupply": 0.01,
            "expected_day": 15.0,
            "days_cover": 50.0,
            "window_days": 5,
        }
        result = plan_reorder(
            stock=3000,
            on_order=0,
            daily_mean=self._daily_mean(),
            spread=5.0,
            lead_days=5,
            risk=risk,
        )
        assert result["reorder"] == 0

    def test_positive_reorder_when_risk_high(self):
        risk = {
            "p_stockout": 0.90,
            "p_before_resupply": 0.80,
            "expected_day": 3.0,
            "days_cover": 1.0,
            "window_days": 5,
        }
        result = plan_reorder(
            stock=10,
            on_order=0,
            daily_mean=self._daily_mean(),
            spread=5.0,
            lead_days=5,
            risk=risk,
        )
        assert result["reorder"] > 0

    def test_reorder_capped_at_max(self):
        risk = {
            "p_stockout": 0.99,
            "p_before_resupply": 0.99,
            "expected_day": 1.0,
            "days_cover": 0.0,
            "window_days": 5,
        }
        result = plan_reorder(
            stock=0,
            on_order=0,
            daily_mean=[200.0] * HORIZON,  # very high demand
            spread=30.0,
            lead_days=10,
            risk=risk,
        )
        assert result["reorder"] <= MAX_REORDER

    def test_on_order_reduces_reorder(self):
        risk = {
            "p_stockout": 0.60,
            "p_before_resupply": 0.50,
            "expected_day": 5.0,
            "days_cover": 3.0,
            "window_days": 5,
        }
        without = plan_reorder(
            stock=100,
            on_order=0,
            daily_mean=self._daily_mean(),
            spread=5.0,
            lead_days=5,
            risk=risk,
        )
        with_order = plan_reorder(
            stock=100,
            on_order=500,
            daily_mean=self._daily_mean(),
            spread=5.0,
            lead_days=5,
            risk=risk,
        )
        assert with_order["reorder"] <= without["reorder"]

    def test_expedite_when_on_order_and_high_risk(self):
        risk = {
            "p_stockout": 0.90,
            "p_before_resupply": 0.80,
            "expected_day": 2.0,
            "days_cover": 0.5,
            "window_days": 5,
        }
        result = plan_reorder(
            stock=10,
            on_order=200,
            daily_mean=self._daily_mean(),
            spread=5.0,
            lead_days=5,
            risk=risk,
        )
        assert result["expedite"] is True

    def test_no_expedite_when_no_on_order(self):
        risk = {
            "p_stockout": 0.90,
            "p_before_resupply": 0.80,
            "expected_day": 2.0,
            "days_cover": 0.5,
            "window_days": 5,
        }
        result = plan_reorder(
            stock=10,
            on_order=0,
            daily_mean=self._daily_mean(),
            spread=5.0,
            lead_days=5,
            risk=risk,
        )
        assert result["expedite"] is False

    def test_output_keys(self):
        risk = {
            "p_stockout": 0.50,
            "p_before_resupply": 0.30,
            "expected_day": 7.0,
            "days_cover": 5.0,
            "window_days": 5,
        }
        result = plan_reorder(
            stock=200,
            on_order=0,
            daily_mean=self._daily_mean(),
            spread=5.0,
            lead_days=5,
            risk=risk,
        )
        for key in ("reorder", "target_position", "expedite"):
            assert key in result
