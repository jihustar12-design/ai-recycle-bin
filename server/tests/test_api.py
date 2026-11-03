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


# ---------- 분류 순간의 사진 ----------
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"  # JPEG 시그니처를 가진 최소 데이터


def test_upload_and_get_image(client):
    eid = client.post("/api/events", json=event()).json()["id"]
    r = client.post(f"/api/events/{eid}/image", content=JPEG, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 201
    got = client.get(f"/api/events/{eid}/image")
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/jpeg"
    assert got.content == JPEG
    assert client.get("/api/events").json()[0]["has_image"] is True
    assert client.get("/api/stats").json()["with_image"] == 1


def test_event_without_image(client):
    eid = client.post("/api/events", json=event()).json()["id"]
    assert client.get("/api/events").json()[0]["has_image"] is False
    assert client.get(f"/api/events/{eid}/image").status_code == 404


@pytest.mark.parametrize(
    "body,ctype,status",
    [
        (JPEG, "image/png", 415),                          # 형식 헤더 오류
        (b"not a jpeg", "image/jpeg", 400),                # JPEG 아님
        (b"", "image/jpeg", 400),                          # 빈 데이터
        (b"\xff\xd8" + b"\x00" * 400_000, "image/jpeg", 413),  # 너무 큼
    ],
)
def test_rejects_bad_image(client, body, ctype, status):
    eid = client.post("/api/events", json=event()).json()["id"]
    r = client.post(f"/api/events/{eid}/image", content=body, headers={"Content-Type": ctype})
    assert r.status_code == status


def test_image_for_unknown_event(client):
    r = client.post("/api/events/999/image", content=JPEG, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 404


def test_migrates_old_db_without_image_column(tmp_path, monkeypatch):
    """이전 버전 DB(has_image 컬럼 없음)로 서버를 켜도 동작해야 한다."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        """CREATE TABLE sort_events (id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT NOT NULL, ts TEXT,
           received_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')), label TEXT NOT NULL,
           confidence REAL NOT NULL, latency_ms INTEGER NOT NULL, bin INTEGER NOT NULL,
           needs_review INTEGER NOT NULL DEFAULT 0, fw_version TEXT NOT NULL, model_version TEXT NOT NULL)"""
    )
    old.execute("INSERT INTO sort_events (device_id,label,confidence,latency_ms,bin,fw_version,model_version) "
                "VALUES ('bin-01','can',0.9,100,0,'0.1.0','v0')")
    old.commit()
    old.close()

    monkeypatch.setenv("DB_PATH", str(path))
    from app.main import app

    with TestClient(app) as c:
        rows = c.get("/api/events").json()
        assert rows[0]["has_image"] is False
        assert c.get("/api/stats").json()["total"] == 1
