"""가짜 ESP32. 실제 펌웨어와 같은 형식/동작으로 서버에 로그와 '분류 순간의 사진'을 보낸다.

- 투입 이벤트를 만들어 POST /api/events → 바로 이어서 POST /api/events/{id}/image (JPEG)
- --fail-rate 로 Wi-Fi 끊김을 흉내: 실패한 로그는 버퍼에 쌓았다가 /api/events/batch 로 재전송
  (펌웨어와 같이, 버퍼에 쌓였던 로그는 메모리 절약을 위해 사진 없이 보냄)
- --use-model: 실제 int8 모델(model/export/model_int8.tflite)로 사진을 분류 → 대시보드에 진짜 결과
  (numpy, pillow, ai-edge-litert 또는 tensorflow 필요. 학습용 model/.venv 에 모두 있음)
- 주기적으로 /api/heartbeat

예)
  python tools/fake_device.py                                   # 샘플 사진 + 무작위 결과
  python tools/fake_device.py --use-model                       # 샘플 사진 + 실제 모델 결과
  python tools/fake_device.py --images model/dataset/test --use-model --interval 3   # 시연용
  python tools/fake_device.py --count 30 --interval 0 --fail-rate 0.3               # CI용
"""
import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["can", "pet", "paper", "general"]
BIN_OF = {"can": 0, "pet": 1, "paper": 2, "general": 3}
CONF_THRESHOLD = 0.70  # firmware/smart_bin/config.h 와 동일하게 유지


class RealModel:
    """펌웨어와 같은 전처리(model/evaluate.py) + int8 모델로 사진을 분류한다."""

    def __init__(self, model_path: Path):
        sys.path.insert(0, str(ROOT / "model"))
        import numpy as np
        from evaluate import Interpreter, to_model_input
        from PIL import Image

        self.np, self.Image, self.prep = np, Image, to_model_input
        self.labels = (model_path.parent / "labels.txt").read_text().split()
        self.it = Interpreter(model_path=str(model_path))
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def classify(self, path: Path) -> tuple[str, float, int]:
        t0 = time.time()
        rgb = self.np.asarray(self.Image.open(path).convert("RGB"))
        s, z = self.inp["quantization"]
        self.it.set_tensor(self.inp["index"], self.prep(rgb, int(self.inp["shape"][1]), s, z)[None])
        self.it.invoke()
        q = self.it.get_tensor(self.out["index"])[0].astype("float32")
        os_, oz = self.out["quantization"]
        probs = (q - oz) * os_
        i = int(probs.argmax())
        return self.labels[i], round(float(probs[i]), 3), int((time.time() - t0) * 1000)


def make_event(device_id, model_version, rng, image, model) -> dict | None:
    if model and image:
        label, conf, latency = model.classify(image)
        if label == "empty":  # 펌웨어도 빈칸이면 동작·기록하지 않음
            return None
    else:
        truth = image.name.split(".")[0] if image else None
        label = truth if truth in LABELS else rng.choices(LABELS, weights=[3, 4, 3, 1])[0]
        conf = round(min(0.99, max(0.30, rng.gauss(0.85, 0.12))), 3)
        latency = int(rng.gauss(180, 25))
    needs_review = conf < CONF_THRESHOLD
    return {
        "device_id": device_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "confidence": conf,
        "latency_ms": latency,
        "bin": BIN_OF["general"] if needs_review else BIN_OF[label],
        "needs_review": needs_review,
        "fw_version": "sim-0.2.0",
        "model_version": model_version,
    }


def find_images(arg: str) -> list[Path]:
    if arg == "none":
        return []
    candidates = [Path(arg)] if arg else [ROOT / "model" / "dataset" / "test", ROOT / "tools" / "samples"]
    for d in candidates:
        files = sorted(p for p in d.glob("*.jpg") if not p.name.startswith("empty."))
        if files:
            return files
    return []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--device-id", default="sim-bin-01")
    p.add_argument("--model-version", default="")
    p.add_argument("--count", type=int, default=0, help="0이면 무한 반복")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--fail-rate", type=float, default=0.0, help="전송 실패를 흉내 낼 확률 (0~1)")
    p.add_argument("--images", default="", help="사진 폴더 (기본: model/dataset/test → tools/samples, 'none'이면 사진 없음)")
    p.add_argument("--use-model", action="store_true", help="실제 int8 모델로 분류")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    rng = random.Random(args.seed)
    images = find_images(args.images)
    model = RealModel(ROOT / "model" / "export" / "model_int8.tflite") if args.use_model else None
    model_version = args.model_version or ("v1-trashnet-160-fast" if model else "sim-random")
    print(f"사진 {len(images)}장 · 분류: {'실제 모델' if model else '무작위'} · 서버 {args.url}")

    buffer: list[dict] = []
    sent = photos = 0
    start = time.time()
    i = 0
    while args.count == 0 or i < args.count:
        i += 1
        image = rng.choice(images) if images else None
        ev = make_event(args.device_id, model_version, rng, image, model)
        if ev is None:
            print(f"[{i}] 빈칸으로 판단 → 동작 안 함")
            time.sleep(args.interval)
            continue

        if rng.random() < args.fail_rate:
            buffer.append(ev)
            print(f"[{i}] Wi-Fi 끊김 흉내 → 버퍼 {len(buffer)}건 보관")
        else:
            try:
                if buffer:  # 복구되면 쌓인 로그부터 (사진 없이)
                    requests.post(f"{args.url}/api/events/batch", json=buffer, timeout=5).raise_for_status()
                    sent += len(buffer)
                    print(f"[{i}] 버퍼 {len(buffer)}건 재전송")
                    buffer.clear()
                r = requests.post(f"{args.url}/api/events", json=ev, timeout=5)
                r.raise_for_status()
                sent += 1
                note = ""
                if image:
                    ri = requests.post(f"{args.url}/api/events/{r.json()['id']}/image", data=image.read_bytes(),
                                       headers={"Content-Type": "image/jpeg"}, timeout=5)
                    if ri.ok:
                        photos += 1
                        note = f" + 사진 {image.name}"
                print(f"[{i}] 전송 {ev['label']} {ev['confidence']}{note}")
            except requests.RequestException as e:
                buffer.append(ev)
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

    if buffer:
        try:
            requests.post(f"{args.url}/api/events/batch", json=buffer, timeout=5).raise_for_status()
            sent += len(buffer)
            buffer.clear()
        except requests.RequestException as e:
            print(f"마지막 전송 실패: {e}")

    print(f"완료: 전송 {sent}건 (사진 {photos}장) / 미전송 {len(buffer)}건")
    return 0 if not buffer else 1


if __name__ == "__main__":
    sys.exit(main())
