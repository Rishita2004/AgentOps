"""Tests for backend.security — tokens, access codes, rate limiting, origin/host parsing."""

import time

import pytest

from backend import security

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Valid env values that satisfy seller_configured()
VALID_ACCESS_CODE = "supersecret!!"  # 14 chars  (>= 12)
VALID_SESSION_SECRET = "s" * 32  # exactly 32  (>= 32)


@pytest.fixture(autouse=True)
def _clean_revoked():
    """Ensure the module-level _revoked dict is empty before and after each test."""
    security._revoked.clear()
    yield
    security._revoked.clear()


@pytest.fixture()
def _configured(monkeypatch):
    """Set env vars so that seller_configured() returns True."""
    monkeypatch.setenv("SELLER_ACCESS_CODE", VALID_ACCESS_CODE)
    monkeypatch.setenv("SESSION_SECRET", VALID_SESSION_SECRET)


# ===================================================================
# 1. issue_token + verify_token
# ===================================================================


class TestIssueAndVerifyToken:
    """Valid, expired, tampered, and empty-token scenarios."""

    def test_valid_token_verifies(self, monkeypatch, _configured):
        token = security.issue_token()
        assert security.verify_token(token) is True

    def test_expired_token_fails(self, monkeypatch, _configured):
        # Issue a token whose expiry is in the past.
        frozen = 1_000_000.0
        monkeypatch.setattr(time, "time", lambda: frozen)
        token = security.issue_token()

        # Jump forward past SESSION_SECONDS so the token is expired.
        monkeypatch.setattr(time, "time", lambda: frozen + security.SESSION_SECONDS + 1)
        assert security.verify_token(token) is False

    def test_tampered_payload_fails(self, monkeypatch, _configured):
        token = security.issue_token()
        # Flip a character in the random nonce portion to tamper the payload.
        parts = token.rsplit(".", 2)
        # parts: [expiry, nonce, signature]
        tampered_nonce = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
        tampered_token = f"{parts[0]}.{tampered_nonce}.{parts[2]}"
        assert security.verify_token(tampered_token) is False

    def test_tampered_signature_fails(self, monkeypatch, _configured):
        token = security.issue_token()
        # Replace last hex char of the signature.
        tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
        assert security.verify_token(tampered) is False

    def test_none_token_fails(self, monkeypatch, _configured):
        assert security.verify_token(None) is False

    def test_empty_string_fails(self, monkeypatch, _configured):
        assert security.verify_token("") is False


# ===================================================================
# 2. revoke_token
# ===================================================================


class TestRevokeToken:
    """A revoked token must no longer verify."""

    def test_revoked_token_no_longer_verifies(self, monkeypatch, _configured):
        token = security.issue_token()
        assert security.verify_token(token) is True

        security.revoke_token(token)
        assert security.verify_token(token) is False

    def test_revoke_none_is_noop(self):
        # Should not raise.
        security.revoke_token(None)

    def test_revoke_empty_is_noop(self):
        security.revoke_token("")


# ===================================================================
# 3. check_access_code
# ===================================================================


class TestCheckAccessCode:
    """Correct code, wrong code, and not-configured scenarios."""

    def test_correct_code_returns_true(self, monkeypatch, _configured):
        assert security.check_access_code(VALID_ACCESS_CODE) is True

    def test_wrong_code_returns_false(self, monkeypatch, _configured):
        assert security.check_access_code("wrong_code!!!") is False

    def test_not_configured_returns_false(self, monkeypatch):
        monkeypatch.delenv("SELLER_ACCESS_CODE", raising=False)
        monkeypatch.delenv("SESSION_SECRET", raising=False)
        assert security.check_access_code(VALID_ACCESS_CODE) is False

    def test_short_access_code_returns_false(self, monkeypatch):
        monkeypatch.setenv("SELLER_ACCESS_CODE", "short")  # < 12
        monkeypatch.setenv("SESSION_SECRET", VALID_SESSION_SECRET)
        assert security.check_access_code("short") is False


# ===================================================================
# 4. seller_configured
# ===================================================================


class TestSellerConfigured:
    """True only when both env vars meet their minimum lengths."""

    def test_true_when_both_set(self, monkeypatch, _configured):
        assert security.seller_configured() is True

    def test_false_when_access_code_missing(self, monkeypatch):
        monkeypatch.delenv("SELLER_ACCESS_CODE", raising=False)
        monkeypatch.setenv("SESSION_SECRET", VALID_SESSION_SECRET)
        assert security.seller_configured() is False

    def test_false_when_session_secret_missing(self, monkeypatch):
        monkeypatch.setenv("SELLER_ACCESS_CODE", VALID_ACCESS_CODE)
        monkeypatch.delenv("SESSION_SECRET", raising=False)
        assert security.seller_configured() is False

    def test_false_when_access_code_too_short(self, monkeypatch):
        monkeypatch.setenv("SELLER_ACCESS_CODE", "short")  # 5 chars
        monkeypatch.setenv("SESSION_SECRET", VALID_SESSION_SECRET)
        assert security.seller_configured() is False

    def test_false_when_session_secret_too_short(self, monkeypatch):
        monkeypatch.setenv("SELLER_ACCESS_CODE", VALID_ACCESS_CODE)
        monkeypatch.setenv("SESSION_SECRET", "s" * 31)  # 31 < 32
        assert security.seller_configured() is False

    def test_false_when_both_missing(self, monkeypatch):
        monkeypatch.delenv("SELLER_ACCESS_CODE", raising=False)
        monkeypatch.delenv("SESSION_SECRET", raising=False)
        assert security.seller_configured() is False


# ===================================================================
# 5. RateLimiter
# ===================================================================


class TestRateLimiter:
    """Allows up to limit, blocks after limit, resets after window."""

    def test_allows_up_to_limit(self):
        rl = security.RateLimiter()
        for _ in range(5):
            assert rl.allow("login", "192.168.1.1", limit=5, window=60) is True

    def test_blocks_after_limit(self):
        rl = security.RateLimiter()
        for _ in range(3):
            rl.allow("login", "10.0.0.1", limit=3, window=60)
        assert rl.allow("login", "10.0.0.1", limit=3, window=60) is False

    def test_resets_after_window(self, monkeypatch):
        rl = security.RateLimiter()
        base = 1_000_000.0
        monkeypatch.setattr(time, "monotonic", lambda: base)

        for _ in range(3):
            rl.allow("login", "10.0.0.2", limit=3, window=10)

        # Still within window — blocked.
        assert rl.allow("login", "10.0.0.2", limit=3, window=10) is False

        # Jump past the window.
        monkeypatch.setattr(time, "monotonic", lambda: base + 11)
        assert rl.allow("login", "10.0.0.2", limit=3, window=10) is True

    def test_different_buckets_independent(self):
        rl = security.RateLimiter()
        for _ in range(2):
            rl.allow("login", "user1", limit=2, window=60)
        assert rl.allow("login", "user1", limit=2, window=60) is False
        # Different bucket should still be allowed.
        assert rl.allow("api", "user1", limit=2, window=60) is True

    def test_different_users_independent(self):
        rl = security.RateLimiter()
        for _ in range(2):
            rl.allow("login", "user_a", limit=2, window=60)
        assert rl.allow("login", "user_a", limit=2, window=60) is False
        assert rl.allow("login", "user_b", limit=2, window=60) is True


# ===================================================================
# 6. allowed_origins
# ===================================================================


class TestAllowedOrigins:
    """Parses ALLOWED_ORIGINS; strips whitespace; returns a set."""

    def test_parses_comma_separated(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "http://a.com,http://b.com")
        result = security.allowed_origins()
        assert result == {"http://a.com", "http://b.com"}

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "  http://a.com , http://b.com  ")
        result = security.allowed_origins()
        assert result == {"http://a.com", "http://b.com"}

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        result = security.allowed_origins()
        assert "http://127.0.0.1:5173" in result
        assert "http://localhost:5173" in result

    def test_skips_empty_entries(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "http://a.com,,http://b.com,")
        result = security.allowed_origins()
        assert "" not in result
        assert result == {"http://a.com", "http://b.com"}


# ===================================================================
# 7. allowed_hosts
# ===================================================================


class TestAllowedHosts:
    """Parses ALLOWED_HOSTS; strips whitespace; returns a list."""

    def test_parses_comma_separated(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_HOSTS", "example.com,api.example.com")
        result = security.allowed_hosts()
        assert result == ["example.com", "api.example.com"]

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_HOSTS", " example.com , api.example.com ")
        result = security.allowed_hosts()
        assert result == ["example.com", "api.example.com"]

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("ALLOWED_HOSTS", raising=False)
        result = security.allowed_hosts()
        assert "127.0.0.1" in result
        assert "localhost" in result

    def test_skips_empty_entries(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_HOSTS", "host1,,host2,")
        result = security.allowed_hosts()
        assert "" not in result
        assert result == ["host1", "host2"]
