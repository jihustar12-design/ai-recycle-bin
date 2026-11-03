"""저장소에 올라간 int8 모델(model/export/model_int8.tflite)의 정확도를 test 세트로 직접 측정한다.

CI 에서 매번 실행되어, README 의 정확도가 '적어 둔 숫자'가 아니라 '측정된 숫자'임을 보장한다.
전처리는 펌웨어(firmware/smart_bin/preprocess.h)와 같은 알고리즘(가운데 크롭 + half-pixel 양선형)을
numpy 로 구현했다.

  python model/evaluate.py                      # 정확도 + 혼동 행렬
  python model/evaluate.py --min-accuracy 0.75  # 기준 미달이면 실패(exit 1)
  python model/evaluate.py --dump DIR           # C++ 펌웨어 테스트용 원본 픽셀/정답/예측 저장
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # 학습 환경(TensorFlow)에서도 동작
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

HERE = Path(__file__).resolve().parent


def to_model_input(rgb: np.ndarray, size: int, scale: float, zero_point: int) -> np.ndarray:
    """preprocess.h 의 toModelInput 과 같은 계산 (가운데 정사각형 크롭 → half-pixel 양선형 축소 → int8)."""
    h, w, _ = rgb.shape
    s = min(w, h)
    ox, oy = (w - s) // 2, (h - s) // 2
    img = rgb[oy : oy + s, ox : ox + s].astype(np.float32)
    ratio = s / size
    src = np.maximum((np.arange(size, dtype=np.float32) + 0.5) * ratio - 0.5, 0)
    i0 = np.minimum(src.astype(np.int32), s - 1)
    i1 = np.minimum(i0 + 1, s - 1)
    f = (src - i0)[:, None]
    rows = img[i0] + (img[i1] - img[i0]) * f[:, :, None]                 # 세로 보간
    out = rows[:, i0] + (rows[:, i1] - rows[:, i0]) * f[None, :, :]      # 가로 보간
    q = np.round(out / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "export" / "model_int8.tflite"))
    ap.add_argument("--labels", default=str(HERE / "export" / "labels.txt"))
    ap.add_argument("--data", default=str(HERE / "dataset" / "test"))
    ap.add_argument("--min-accuracy", type=float, default=0.0)
    ap.add_argument("--dump", default="", help="C++ 테스트용 데이터 저장 폴더")
    ap.add_argument("--json", default="", help="결과를 JSON 으로 저장")
    args = ap.parse_args()

    labels = Path(args.labels).read_text().split()
    it = Interpreter(model_path=args.model)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    size = int(inp["shape"][1])
    s_in, z_in = inp["quantization"]

    files = sorted(Path(args.data).glob("*.jpg"))
    if not files:
        print(f"{args.data} 에 이미지가 없습니다. 먼저 python model/prepare_dataset.py 를 실행하세요.")
        return 2

    n = len(labels)
    conf = np.zeros((n, n), dtype=int)
    dump = Path(args.dump) if args.dump else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        index = open(dump / "index.txt", "w")

    for i, f in enumerate(files):
        truth = labels.index(f.name.split(".")[0])
        rgb = np.asarray(Image.open(f).convert("RGB"))
        x = to_model_input(rgb, size, s_in, z_in)
        it.set_tensor(inp["index"], x[None])
        it.invoke()
        pred = int(np.argmax(it.get_tensor(outp["index"])[0]))
        conf[truth, pred] += 1
        if dump:
            raw = f"{i:04d}.rgb"
            (dump / raw).write_bytes(rgb.tobytes())
            index.write(f"{raw} {rgb.shape[1]} {rgb.shape[0]} {truth} {pred}\n")
    if dump:
        index.close()

    acc = float(np.trace(conf) / conf.sum())
    per = {}
    for i, lb in enumerate(labels):
        col, row = conf[:, i].sum(), conf[i].sum()
        per[lb] = {"precision": round(float(conf[i, i] / col), 4) if col else 0.0,
                   "recall": round(float(conf[i, i] / row), 4) if row else 0.0}

    head = "| 실제 \\ 예측 | " + " | ".join(labels) + " |"
    sep = "|---" * (n + 1) + "|"
    rows = [f"| **{lb}** | " + " | ".join(str(v) for v in conf[i]) + " |" for i, lb in enumerate(labels)]
    md = [
        "## 모델 정확도 (ESP32 int8 모델, TrashNet test)",
        "",
        f"**{acc * 100:.1f}%** ({int(np.trace(conf))}/{conf.sum()})  ·  "
        f"모델 {os.path.getsize(args.model) / 1024:.0f} KB  ·  입력 {size}x{size} RGB",
        "",
        head, sep, *rows,
        "",
        "| 클래스 | precision | recall |", "|---|---|---|",
        *[f"| {lb} | {v['precision'] * 100:.1f}% | {v['recall'] * 100:.1f}% |" for lb, v in per.items()],
    ]
    print("\n".join(md))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write("\n".join(md) + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"accuracy": round(acc, 4), "confusion": conf.tolist(), "per_class": per, "labels": labels},
            indent=2, ensure_ascii=False))

    if acc < args.min_accuracy:
        print(f"\n실패: 정확도 {acc * 100:.1f}% < 기준 {args.min_accuracy * 100:.0f}%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
