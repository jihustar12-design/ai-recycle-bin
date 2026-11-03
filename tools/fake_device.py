"""가짜 ESP32. 실제 펌웨어와 같은 형식/동작으로 서버에 로그를 보낸다.

- 투입 이벤트를 랜덤으로 만들어 POST /api/events
- --fail-rate 로 Wi-Fi 끊김을 흉내: 실패한 로그는 버퍼에 쌓았다가 복구되면 /api/events/batch 로 재전송
- 주기적으로 /api/heartbeat

예)
  python tools/fake_device.py                       # 무한 반복, 2초 간격
  python tools/fake_device.py --count 30 --interval 0 --fail-rate 0.3   # CI용
"""
import argparse
import random
import sys
import time
from datetime import datetime, timezone

import requests

LABELS = ["can", "pet", "paper", "general"]
BIN_OF = {"can": 0, "pet": 1, "paper": 2, "general": 3}
CONF_THRESHOLD = 0.70  # firmware/smart_bin/config.h 와 동일하게 유지


def make_event(device_id: str, model_version: str, rng: random.Random) -> dict:
    label = rng.choices(LABELS, weights=[3, 4, 3, 1])[0]
    conf = round(min(0.99, max(0.30, rng.gauss(0.85, 0.12))), 3)
    needs_review = conf < CONF_THRESHOLD
    return {
        "device_id": device_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "confidence": conf,
        "latency_ms": int(rng.gauss(180, 25)),
        "bin": BIN_OF["general"] if needs_review else BIN_OF[label],
        "needs_review": needs_review,
        "fw_version": "sim-0.1.0",
        "model_version": model_version,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--device-id", default="sim-bin-01")
    p.add_argument("--model-version", default="v0-phone")
    p.add_argument("--count", type=int, default=0, help="0이면 무한 반복")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--fail-rate", type=float, default=0.0, help="전송 실패를 흉내 낼 확률 (0~1)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    rng = random.Random(args.seed)
    buffer: list[dict] = []
    sent = 0
    start = time.time()
    i = 0
    while args.count == 0 or i < args.count:
        i += 1
        buffer.append(make_event(args.device_id, args.model_version, rng))

        if rng.random() < args.fail_rate:
            print(f"[{i}] Wi-Fi 끊김 흉내 → 버퍼 {len(buffer)}건 보관")
        else:
            try:
                if len(buffer) == 1:
                    requests.post(f"{args.url}/api/events", json=buffer[0], timeout=5).raise_for_status()
                else:
                    requests.post(f"{args.url}/api/events/batch", json=buffer, timeout=5).raise_for_status()
                print(f"[{i}] 전송 {len(buffer)}건 ({buffer[-1]['label']}, {buffer[-1]['confidence']})")
                sent += len(buffer)
                buffer.clear()
            except requests.RequestException as e:
                print(f"[{i}] 전송 실패: {e} → 버퍼 {len(buffer)}건 보관")

        if i % 10 == 0:
            try:
                requests.post(f"{args.url}/api/heartbeat", json={
                    "device_id": args.device_id, "rssi": rng.randint(-80, -45),
                    "free_heap": rng.randint(150_000, 200_000), "uptime_s": int(time.time() - start),
                    "pending_logs": len(buffer),
                }, timeout=5)
            except requests.RequestException:
                pass
        time.sleep(args.interval)

    # 종료 전 남은 로그 비우기
    if buffer:
        try:
            requests.post(f"{args.url}/api/events/batch", json=buffer, timeout=5).raise_for_status()
            sent += len(buffer)
            buffer.clear()
        except requests.RequestException as e:
            print(f"마지막 전송 실패: {e}")

    print(f"완료: 생성 {i}건 / 전송 {sent}건 / 미전송 {len(buffer)}건")
    return 0 if not buffer else 1


if __name__ == "__main__":
    sys.exit(main())
