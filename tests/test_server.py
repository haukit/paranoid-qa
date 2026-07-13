from fastapi.testclient import TestClient

from paranoid_qa.serving.api import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_stub_mode():
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["mode"] == "stub"


def test_ask_json_returns_stub_payload():
    response = client.post("/ask_json", json={"question": "deployment smoke test"})
    assert response.status_code == 200
    assert response.json()["answer"].startswith("This is a stub")


def test_ask_rejects_too_short_question():
    response = client.post("/ask_json", json={"question": "hi"})
    assert response.status_code == 422
