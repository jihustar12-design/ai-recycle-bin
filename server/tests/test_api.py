import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def event(**kw):
    base = {
        "device_id": "bin-01",
        "label": "pet",
        "confidence": 0.91,
        "latency_ms": 180,
        "bin": 1,
        "needs_review": False,
        "fw_version": "0.1.0",
        "model_version": "v0-phone",
    }
    base.update(kw)
    return base


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_list_event(client):
    r = client.post("/api/events", json=event())
    assert r.status_code == 201
    rows = client.get("/api/events").json()
    assert len(rows) == 1
    assert rows[0]["label"] == "pet"
    assert rows[0]["needs_review"] is False


@pytest.mark.parametrize(
    "bad",
    [
        {"label": "glass"},          # 정의되지 않은 클래스
        {"confidence": 1.5},         # 범위 초과
        {"bin": 9},                  # 없는 칸
        {"device_id": ""},           # 빈 장치 ID
    ],
)
def test_rejects_invalid_payload(client, bad):
    assert client.post("/api/events", json=event(**bad)).status_code == 422


def test_batch_upload(client):
    r = client.post("/api/events/batch", json=[event(label="can", bin=0), event(label="paper", bin=2)])
    assert r.status_code == 201
    assert r.json()["count"] == 2
    assert len(client.get("/api/events?label=can").json()) == 1


def test_feedback_and_stats(client):
    a = client.post("/api/events", json=event(label="pet")).json()["id"]
    b = client.post("/api/events", json=event(label="can", bin=0, needs_review=True, confidence=0.5)).json()["id"]
    client.post("/api/feedback", json={"event_id": a, "correct_label": "pet"})  # 정답
    client.post("/api/feedback", json={"event_id": b, "correct_label": "paper"})  # 오분류

    s = client.get("/api/stats").json()
    assert s["total"] == 2
    assert s["by_label"]["pet"] == 1 and s["by_label"]["can"] == 1
    assert s["needs_review"] == 1
    assert s["feedback_count"] == 2
    assert s["misclassified"] == 1
    assert s["misclassification_rate"] == 0.5
    assert s["by_model_version"]["v0-phone"]["misclassification_rate"] == 0.5


def test_feedback_unknown_event(client):
    assert client.post("/api/feedback", json={"event_id": 999, "correct_label": "pet"}).status_code == 404


def test_heartbeat_latest_per_device(client):
    client.post("/api/heartbeat", json={"device_id": "bin-01", "rssi": -70, "pending_logs": 3})
    client.post("/api/heartbeat", json={"device_id": "bin-01", "rssi": -60, "pending_logs": 0})
    devs = client.get("/api/devices").json()
    assert len(devs) == 1 and devs[0]["rssi"] == -60


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "AI 분리수거함" in r.text
