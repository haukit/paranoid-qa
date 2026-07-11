from __future__ import annotations

import secrets
from datetime import datetime, timezone

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from paranoid_qa.config import settings


def _serializer() -> URLSafeTimedSerializer:
    if not settings.demo_secret_key:
        raise RuntimeError("PARANOID_QA_DEMO_SECRET_KEY is required when the demo gate is on")
    return URLSafeTimedSerializer(settings.demo_secret_key, salt="demo-session")


def issue_session(sid: str) -> str:
    """Return a signed, timestamped token carrying the session id."""
    return _serializer().dumps({"sid": sid})


def read_session(token: str) -> str | None:
    """Return the session id if the token is valid and unexpired, else None."""
    try:
        data = _serializer().loads(token, max_age=settings.demo_session_days * 86400)
    except (BadSignature, SignatureExpired):
        return None
    return data.get("sid")


class DemoDenied(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        """Gate refusal exception"""
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


_remaining: dict[str, int] = {}  # session id -> remaining questions
_daily: dict[str, int] = {}  # UTC date -> questions served that day


def start_session() -> str:
    """Create a session with a fresh quota; return its signed token."""
    sid = secrets.token_urlsafe(16)
    _remaining[sid] = settings.demo_questions_per_session
    return issue_session(sid)


def charge(sid: str) -> None:
    """Charge one question to the session and the global daily budget, or raise DemoDenied."""
    remaining = _remaining.get(sid)
    if remaining is None:
        raise DemoDenied(401, "Session expired; start a new session")
    if remaining <= 0:
        raise DemoDenied(429, "Session question limit reached")
    today = datetime.now(timezone.utc).date().isoformat()
    if _daily.get(today, 0) >= settings.demo_global_daily_limit:
        raise DemoDenied(429, "Global daily limit reached; try again tomorrow")

    _remaining[sid] = remaining - 1
    _daily[today] = _daily.get(today, 0) + 1


def remaining(sid: str) -> int | None:
    """Questions left for the session, or None if it has no live record."""
    return _remaining.get(sid)
