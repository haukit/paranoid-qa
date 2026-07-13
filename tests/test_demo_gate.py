import pytest
from fastapi.testclient import TestClient

from paranoid_qa.config import settings
from paranoid_qa.serving import demo
from paranoid_qa.serving.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_gate(monkeypatch):
    demo._remaining.clear()
    demo._daily.clear()
    monkeypatch.setattr(settings, "demo_require_access", True)
    monkeypatch.setattr(settings, "demo_disabled", False)
    monkeypatch.setattr(settings, "demo_invite_code", "test-code")
    monkeypatch.setattr(settings, "demo_secret_key", "test-secret")


def test_ask_without_session_is_401():
    assert client.post("/ask_json", json={"question": "hello there"}).status_code == 401


def test_bad_invite_code_is_401():
    assert client.post("/demo/session", json={"token": "wrong"}).status_code == 401


def test_session_flow_allows_ask():
    token = client.post("/demo/session", json={"token": "test-code"}).json()["session"]
    response = client.post(
        "/ask_json",
        json={"question": "hello there"},
        headers={"X-Demo-Session": token},
    )
    assert response.status_code == 200
    assert response.json()["answer"].startswith("This is a stub")


def test_session_quota_exhausts(monkeypatch):
    monkeypatch.setattr(settings, "demo_questions_per_session", 1)
    token = client.post("/demo/session", json={"token": "test-code"}).json()["session"]
    headers = {"X-Demo-Session": token}
    assert (
        client.post("/ask_json", json={"question": "one two three"}, headers=headers).status_code
        == 200
    )
    assert (
        client.post("/ask_json", json={"question": "four five six"}, headers=headers).status_code
        == 429
    )


def test_session_status_reports_remaining(monkeypatch):
    monkeypatch.setattr(settings, "demo_questions_per_session", 3)
    token = client.post("/demo/session", json={"token": "test-code"}).json()["session"]
    headers = {"X-Demo-Session": token}
    assert client.get("/demo/session", headers=headers).json()["remaining"] == 3
    client.post("/ask_json", json={"question": "one two three"}, headers=headers)
    assert client.get("/demo/session", headers=headers).json()["remaining"] == 2
