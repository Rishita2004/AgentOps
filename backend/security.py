import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from hashlib import sha256
from threading import Lock

COOKIE = "seller_session"
SESSION_SECONDS = 8 * 3600
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
MAX_BODY_BYTES = 1_200_000


def _secret() -> bytes:
    return os.getenv("SESSION_SECRET", "").encode()


def seller_configured() -> bool:
    return len(os.getenv("SELLER_ACCESS_CODE", "")) >= 12 and len(_secret()) >= 32


def check_access_code(candidate: str) -> bool:
    expected = os.getenv("SELLER_ACCESS_CODE", "")
    return seller_configured() and hmac.compare_digest(candidate.encode(), expected.encode())


_revoked: set[str] = set()


def issue_token() -> str:
    payload = f"{int(time.time()) + SESSION_SECONDS}.{secrets.token_urlsafe(18)}"
    return f"{payload}.{hmac.new(_secret(), payload.encode(), sha256).hexdigest()}"


def verify_token(token: str | None) -> bool:
    if not token or not seller_configured() or token in _revoked:
        return False
    payload, _, signature = token.rpartition(".")
    expected = hmac.new(_secret(), payload.encode(), sha256).hexdigest()
    if not payload or not hmac.compare_digest(signature, expected):
        return False
    try:
        return int(payload.split(".", 1)[0]) > time.time()
    except ValueError:
        return False


def revoke_token(token: str | None) -> None:
    if token:
        _revoked.add(token)
        if len(_revoked) > 5000:
            _revoked.clear()


class RateLimiter:
    def __init__(self):
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, bucket: str, who: str, limit: int, window: int) -> bool:
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 10000:
                self._hits.clear()
            hits = self._hits[(bucket, who)]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


limiter = RateLimiter()


def allowed_origins() -> set[str]:
    return {item.strip() for item in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",") if item.strip()}


def allowed_hosts() -> list[str]:
    return [item.strip() for item in os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if item.strip()]


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}
