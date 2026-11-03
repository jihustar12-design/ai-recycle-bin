"""ESP32 <-> 서버 메시지 규격. 펌웨어와 fake_device.py 모두 이 형식을 따른다."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

# 분류 클래스 (Edge Impulse 라벨과 동일하게 유지)
Label = Literal["can", "pet", "paper", "general", "empty"]
LABELS: tuple[str, ...] = ("can", "pet", "paper", "general", "empty")


class SortEventIn(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    ts: Optional[datetime] = None  # 장치 시각(없으면 서버 수신 시각)
    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: int = Field(ge=0)
    bin: int = Field(ge=0, le=3)
    needs_review: bool = False
    fw_version: str = "0.0.0"
    model_version: str = "unknown"


class SortEventOut(SortEventIn):
    id: int
    received_at: datetime
    has_image: bool = False


class HeartbeatIn(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    rssi: Optional[int] = None
    free_heap: Optional[int] = None
    uptime_s: Optional[int] = None
    pending_logs: int = 0  # 장치에 쌓여 있는 미전송 로그 수


class FeedbackIn(BaseModel):
    event_id: int
    correct_label: Label


class Stats(BaseModel):
    total: int
    by_label: dict[str, int]
    needs_review: int
    with_image: int
    feedback_count: int
    misclassified: int
    misclassification_rate: float
    avg_confidence: float
    avg_latency_ms: float
    by_model_version: dict[str, dict[str, float]]
