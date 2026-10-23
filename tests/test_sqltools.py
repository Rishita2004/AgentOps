"""Tests for backend.sqltools — jsonable helper, QUERIES registry, and SqlSession validation."""

import datetime as dt
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from backend.sqltools import QUERIES, SqlSession, jsonable


# ── jsonable ────────────────────────────────────────────────────────────────

class TestJsonable:
    def test_date_to_isoformat(self):
        assert jsonable(dt.date(2026, 1, 15)) == "2026-01-15"

    def test_datetime_to_isoformat(self):
        assert jsonable(dt.datetime(2026, 1, 15, 9, 30, 0)) == "2026-01-15T09:30:00"

    def test_decimal_to_float(self):
        assert jsonable(Decimal("19.99")) == 19.99
        assert isinstance(jsonable(Decimal("19.99")), float)

    def test_nested_dict(self):
        inp = {"price": Decimal("9.50"), "day": dt.date(2026, 3, 1)}
        out = jsonable(inp)
        assert out == {"price": 9.50, "day": "2026-03-01"}

    def test_nested_list(self):
        inp = [Decimal("1.1"), dt.date(2026, 6, 1), "hello"]
        out = jsonable(inp)
        assert out == [1.1, "2026-06-01", "hello"]

    def test_nested_dict_in_list(self):
        inp = [{"ts": dt.datetime(2026, 5, 1, 12, 0)}]
        assert jsonable(inp) == [{"ts": "2026-05-01T12:00:00"}]

    def test_tuple_converted_to_list(self):
        assert jsonable((Decimal("2"), "a")) == [2.0, "a"]

    @pytest.mark.parametrize("value", ["hello", 42, 3.14, None, True])
    def test_passthrough(self, value):
        assert jsonable(value) is value


# ── QUERIES dict ────────────────────────────────────────────────────────────

EXPECTED_QUERY_NAMES = [
    "stock_positions",
    "sales_history",
    "sales_summary",
    "open_orders",
    "market_day",
]


class TestQueries:
    @pytest.mark.parametrize("name", EXPECTED_QUERY_NAMES)
    def test_query_exists(self, name):
        assert name in QUERIES

    def test_all_values_are_nonempty_strings(self):
        for name, sql in QUERIES.items():
            assert isinstance(sql, str), f"{name} is not a string"
            assert sql.strip(), f"{name} is empty"

    def test_no_unexpected_missing_queries(self):
        """EXPECTED_QUERY_NAMES should cover every key in QUERIES."""
        assert set(QUERIES.keys()) == set(EXPECTED_QUERY_NAMES)


# ── SqlSession.run — validation only ───────────────────────────────────────

class TestSqlSessionValidation:
    def test_rejects_non_allowlisted_name(self):
        fake_engine = MagicMock()
        session = SqlSession(engine=fake_engine)
        with pytest.raises(ValueError, match="not allowlisted"):
            session.run("drop_all_tables")
        # Engine should never have been touched
        fake_engine.connect.assert_not_called()

    def test_rejects_empty_name(self):
        fake_engine = MagicMock()
        session = SqlSession(engine=fake_engine)
        with pytest.raises(ValueError, match="not allowlisted"):
            session.run("")
        fake_engine.connect.assert_not_called()
