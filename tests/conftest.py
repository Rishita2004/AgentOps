"""Shared fixtures for guardrails tests."""

import pytest

from backend.guardrails import POLICY_NUMBERS, allowed_numbers


@pytest.fixture()
def policy_numbers():
    """The hard-coded policy numbers from guardrails."""
    return POLICY_NUMBERS


@pytest.fixture()
def sample_tool_output():
    """A realistic dict of tool output keyed by product id."""
    return {
        "SKU-001": {"price": 499.0, "units_sold": 120, "stock": 340},
        "SKU-002": {"price": 299.0, "units_sold": 80, "stock": 150},
    }


@pytest.fixture()
def sample_allowed(sample_tool_output):
    """Pre-computed allowed number set from sample_tool_output."""
    return allowed_numbers(sample_tool_output)
