"""AI 스마트 분리수거함 로그 수집 서버."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import db
from .schemas import LABELS, FeedbackIn, HeartbeatIn, SortEventIn, SortEventOut, Stats

DASHBOARD_DIR = Path(os.environ.get("DASHBOARD_DIR", Path(__file__).resolve().parents[2] / "dashboard"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="AI Recycle Bin", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------- 분류 이벤트 ----------
@app.post("/api/events", status_code=201)
def create_event(ev: SortEventIn) -> dict:
    with db.connect() as conn:
        cur = conn.execute(
            """INSERT INTO sort_events
               (device_id, ts, label, confidence, latency_ms, bin, needs_review, fw_version, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ev.device_id,
                ev.ts.isoformat() if ev.ts else None,
                ev.label,
                ev.confidence,
                ev.latency_ms,
                ev.bin,
                int(ev.needs_review),
                ev.fw_version,
                ev.model_version,
            ),
        )
        return {"id": cur.lastrowid}


@app.post("/api/events/batch", status_code=201)
def create_events_batch(events: list[SortEventIn]) -> dict:
    """장치가 Wi-Fi 복구 후 버퍼에 쌓인 로그를 한 번에 보낼 때 사용."""
    ids = [create_event(ev)["id"] for ev in events]
    return {"ids": ids, "count": len(ids)}


@app.get("/api/events", response_model=list[SortEventOut])
def list_events(
    limit: int = Query(50, ge=1, le=500),
    device_id: Optional[str] = None,
    label: Optional[str] = None,
) -> list[dict]:
    sql = "SELECT * FROM sort_events WHERE 1=1"
    args: list = []
    if device_id:
        sql += " AND device_id = ?"
        args.append(device_id)
    if label:
        sql += " AND label = ?"
        args.append(label)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with db.connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [{**dict(r), "needs_review": bool(r["needs_review"])} for r in rows]


# ---------- 오분류 신고 ----------
@app.post("/api/feedback", status_code=201)
def create_feedback(fb: FeedbackIn) -> dict:
    with db.connect() as conn:
        if not conn.execute("SELECT 1 FROM sort_events WHERE id = ?", (fb.event_id,)).fetchone():
            raise HTTPException(404, "event not found")
        cur = conn.execute(
            "INSERT INTO feedback (event_id, correct_label) VALUES (?, ?)", (fb.event_id, fb.correct_label)
        )
        return {"id": cur.lastrowid}


# ---------- 장치 상태 ----------
@app.post("/api/heartbeat", status_code=201)
def heartbeat(hb: HeartbeatIn) -> dict:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO device_status (device_id, rssi, free_heap, uptime_s, pending_logs) VALUES (?, ?, ?, ?, ?)",
            (hb.device_id, hb.rssi, hb.free_heap, hb.uptime_s, hb.pending_logs),
        )
    return {"ok": True}


@app.get("/api/devices")
def devices() -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT d.* FROM device_status d
               JOIN (SELECT device_id, MAX(id) AS mid FROM device_status GROUP BY device_id) m ON d.id = m.mid
               ORDER BY d.device_id"""
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- 통계 ----------
@app.get("/api/stats", response_model=Stats)
def stats() -> dict:
    with db.connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sort_events").fetchone()[0]
        by_label = {lb: 0 for lb in LABELS}
        for r in conn.execute("SELECT label, COUNT(*) c FROM sort_events GROUP BY label"):
            by_label[r["label"]] = r["c"]
        needs_review = conn.execute("SELECT COUNT(*) FROM sort_events WHERE needs_review = 1").fetchone()[0]
        agg = conn.execute("SELECT AVG(confidence), AVG(latency_ms) FROM sort_events").fetchone()

        # 이벤트별 가장 최근 신고만 사용
        latest_fb = """SELECT f.event_id, f.correct_label FROM feedback f
                       JOIN (SELECT event_id, MAX(id) mid FROM feedback GROUP BY event_id) m ON f.id = m.mid"""
        fb_rows = conn.execute(
            f"SELECT e.model_version, e.label, lf.correct_label FROM ({latest_fb}) lf "
            "JOIN sort_events e ON e.id = lf.event_id"
        ).fetchall()

        per_model: dict[str, dict[str, float]] = {}
        for r in conn.execute(
            "SELECT model_version, COUNT(*) c, AVG(confidence) ac FROM sort_events GROUP BY model_version"
        ):
            per_model[r["model_version"]] = {"count": r["c"], "avg_confidence": round(r["ac"], 4),
                                             "feedback": 0, "misclassified": 0}

    misclassified = 0
    for r in fb_rows:
        m = per_model[r["model_version"]]
        m["feedback"] += 1
        if r["label"] != r["correct_label"]:
            m["misclassified"] += 1
            misclassified += 1
    for m in per_model.values():
        m["misclassification_rate"] = round(m["misclassified"] / m["feedback"], 4) if m["feedback"] else 0.0

    return {
        "total": total,
        "by_label": by_label,
        "needs_review": needs_review,
        "feedback_count": len(fb_rows),
        "misclassified": misclassified,
        "misclassification_rate": round(misclassified / len(fb_rows), 4) if fb_rows else 0.0,
        "avg_confidence": round(agg[0] or 0.0, 4),
        "avg_latency_ms": round(agg[1] or 0.0, 1),
        "by_model_version": per_model,
    }


# ---------- 대시보드 ----------
@app.get("/", include_in_schema=False)
def dashboard():
    index = DASHBOARD_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "dashboard not found")
    return FileResponse(index)
