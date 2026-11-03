"""SQLite 저장소. DB_PATH 환경변수로 위치를 바꿀 수 있다 (테스트는 임시 파일 사용)."""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS sort_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT    NOT NULL,
    ts            TEXT,
    received_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    label         TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    latency_ms    INTEGER NOT NULL,
    bin           INTEGER NOT NULL,
    needs_review  INTEGER NOT NULL DEFAULT 0,
    fw_version    TEXT    NOT NULL,
    model_version TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_received ON sort_events(received_at);

CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      INTEGER NOT NULL REFERENCES sort_events(id),
    correct_label TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS device_status (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id    TEXT    NOT NULL,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    rssi         INTEGER,
    free_heap    INTEGER,
    uptime_s     INTEGER,
    pending_logs INTEGER NOT NULL DEFAULT 0
);
"""


def db_path() -> str:
    return os.environ.get("DB_PATH", str(Path(__file__).resolve().parents[1] / "data" / "smartbin.db"))


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
