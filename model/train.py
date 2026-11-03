"""맥에서 직접 학습: MobileNetV2 전이학습 → ESP32용 int8 TFLite.

Edge Impulse 무료 학습 시간(월 25분) 제한 없이 정확도를 최대로 끌어올리기 위한 스크립트.

  1단계  베이스(ImageNet 사전학습) 고정, 분류기만 학습
  2단계  베이스 뒷부분을 풀어 낮은 학습률로 미세조정(fine-tuning)
  평가   test 세트로 float 모델 / int8 모델 정확도 + 혼동 행렬
  출력   model/export/model_int8.tflite, labels.txt, model_data.h, metrics.json

사용 (model/README.md 참고):
  python model/train.py
  python model/train.py --size 96          # 더 빠르고 가벼운 모델
  python model/train.py --alpha 0.5        # 더 크고 정확한 모델 (ESP32에서 느려짐)
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

HERE = Path(__file__).resolve().parent
# Edge Impulse 와 같은 알파벳순. 펌웨어는 이름으로 매핑하므로 순서가 달라도 안전.
LABELS = ["can", "empty", "general", "paper", "pet"]


def list_split(folder: Path) -> tuple[list[str], list[int]]:
    paths, ys = [], []
    for p in sorted(folder.glob("*.jpg")):
        label = p.name.split(".")[0]
        if label in LABELS:
            paths.append(str(p))
            ys.append(LABELS.index(label))
    return paths, ys


def stratified_val(paths, ys, ratio, seed):
    rng = random.Random(seed)
    by = {}
    for p, y in zip(paths, ys):
        by.setdefault(y, []).append(p)
    tr, va = ([], []), ([], [])
    for y, ps in by.items():
        rng.shuffle(ps)
        n = round(len(ps) * ratio)
        va[0].extend(ps[:n])
        va[1].extend([y] * n)
        tr[0].extend(ps[n:])
        tr[1].extend([y] * (len(ps) - n))
    return tr, va


def make_loader(size: int):
    def load(path, y):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        # ESP32 에서도 똑같이: 4:3 → 가운데 정사각형 크롭 → size×size
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        s = tf.minimum(h, w)
        img = tf.image.crop_to_bounding_box(img, (h - s) // 2, (w - s) // 2, s, s)
        img = tf.image.resize(img, (size, size), method="bilinear")
        return tf.cast(img, tf.float32), tf.one_hot(y, len(LABELS))  # 0~255 그대로
    return load


def dataset(paths, ys, size, batch, shuffle, seed=0):
    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return ds.map(make_loader(size), num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(size: int, alpha: float, weights):
    aug = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom((-0.15, 0.1)),
            keras.layers.RandomTranslation(0.08, 0.08),
            keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
            keras.layers.RandomContrast(0.2),
        ],
        name="augment",
    )
    base = keras.applications.MobileNetV2(
        input_shape=(size, size, 3), alpha=alpha, include_top=False, weights=weights
    )
    base.trainable = False

    inp = keras.Input((size, size, 3), name="image")          # 0~255 RGB
    x = aug(inp)
    x = keras.layers.Rescaling(1 / 127.5, offset=-1, name="to_mobilenet_range")(x)
    x = base(x, training=False)                               # BatchNorm 은 항상 추론 모드
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    out = keras.layers.Dense(len(LABELS), activation="softmax", name="probs")(x)
    return keras.Model(inp, out), base


def fit(model, train_ds, val_ds, epochs, lr, patience, label):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    print(f"\n=== {label}: 최대 {epochs} epoch, lr={lr} ===", flush=True)
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping("val_accuracy", patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-7),
        ],
    )


def fast_head(model, base, tr_p, tr_y, va_p, va_y, args):
    """빠른 모드: 베이스 모델로 특징을 '한 번만' 뽑아 두고, 작은 분류기만 학습한다.
    매 epoch 마다 사진 전체를 베이스에 통과시키지 않으므로 수십 배 빠르다.
    대신 미세조정을 하지 않아 정확도가 조금 낮을 수 있다."""
    feat = keras.Sequential(
        [keras.Input((args.size, args.size, 3)), keras.layers.Rescaling(1 / 127.5, offset=-1), base,
         keras.layers.GlobalAveragePooling2D()]
    )
    load = make_loader(args.size)

    def images(paths):
        return np.stack([load(tf.constant(p), 0)[0].numpy() for p in paths])

    print("\n=== 빠른 모드: 특징 추출 (원본 + 뒤집기·회전 3종) ===", flush=True)
    x_tr_img, x_va_img = images(tr_p), images(va_p)
    variants = [x_tr_img, x_tr_img[:, :, ::-1], x_tr_img[:, ::-1, :], np.rot90(x_tr_img, 1, axes=(1, 2))]
    x_tr = np.concatenate([feat.predict(v, batch_size=64, verbose=0) for v in variants])
    y_tr = np.concatenate([np.array(tr_y)] * len(variants))
    x_va = feat.predict(x_va_img, batch_size=64, verbose=0)

    head = keras.Sequential(
        [keras.Input((x_tr.shape[1],)), keras.layers.Dropout(0.25),
         keras.layers.Dense(len(LABELS), activation="softmax")]
    )
    head.compile(keras.optimizers.Adam(1e-3),
                 keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=["accuracy"])
    print("=== 빠른 모드: 분류기 학습 ===", flush=True)
    head.fit(
        x_tr, keras.utils.to_categorical(y_tr, len(LABELS)),
        validation_data=(x_va, keras.utils.to_categorical(va_y, len(LABELS))),
        epochs=200, batch_size=64, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping("val_accuracy", patience=20, restore_best_weights=True)],
    )
    # 학습된 분류기 가중치를 전체 모델(이미지 입력)에 옮겨 담는다
    model.get_layer("probs").set_weights(head.layers[-1].get_weights())


def confusion(y_true, y_pred):
    m = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def report(name, y_true, y_pred):
    m = confusion(y_true, y_pred)
    acc = float(np.trace(m) / m.sum())
    print(f"\n[{name}] test accuracy = {acc * 100:.1f}%")
    print("실제\\예측  " + " ".join(f"{lb:>8}" for lb in LABELS))
    for i, lb in enumerate(LABELS):
        print(f"{lb:>9}  " + " ".join(f"{v:>8}" for v in m[i]))
    per = {}
    for i, lb in enumerate(LABELS):
        tp = m[i, i]
        prec = tp / m[:, i].sum() if m[:, i].sum() else 0.0
        rec = tp / m[i].sum() if m[i].sum() else 0.0
        per[lb] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4)}
    return {"accuracy": round(acc, 4), "confusion": m.tolist(), "per_class": per}


def to_int8_tflite(model, rep_paths, size):
    load = make_loader(size)

    def rep():
        for p in rep_paths:
            img, _ = load(tf.constant(p), 0)
            yield [tf.expand_dims(img, 0)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def eval_tflite(blob, paths, ys, size):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = i_d["quantization"]
    load = make_loader(size)
    preds = []
    for p in paths:
        img, _ = load(tf.constant(p), 0)
        q = np.clip(np.round(img.numpy() / s_in + z_in), -128, 127).astype(np.int8)
        it.set_tensor(i_d["index"], q[None])
        it.invoke()
        preds.append(int(np.argmax(it.get_tensor(o_d["index"])[0])))
    return preds, (float(s_in), int(z_in))


def write_header(blob: bytes, path: Path, size: int, quant):
    lines = [
        "// 자동 생성: model/train.py — 직접 수정하지 마세요",
        "#pragma once",
        "#include <stdint.h>",
        f"#define MODEL_INPUT_SIZE {size}",
        f"#define MODEL_INPUT_SCALE {quant[0]:.8f}f",
        f"#define MODEL_INPUT_ZERO_POINT {quant[1]}",
        f'static const char* const MODEL_LABELS[] = {{{", ".join(chr(34) + lb + chr(34) for lb in LABELS)}}};',
        f"#define MODEL_LABEL_COUNT {len(LABELS)}",
        "alignas(16) static const unsigned char g_model[] = {",
    ]
    for i in range(0, len(blob), 16):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in blob[i : i + 16]) + ",")
    lines += ["};", f"static const unsigned int g_model_len = {len(blob)};", ""]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "dataset"))
    ap.add_argument("--out", default=str(HERE / "export"))
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=30)
    ap.add_argument("--epochs-fine", type=int, default=30)
    ap.add_argument("--fine-tune-fraction", type=float, default=0.6, help="베이스 뒤쪽 몇 %를 풀지")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' 또는 'none'(테스트용)")
    ap.add_argument("--limit", type=int, default=0, help="빠른 동작 확인용: 클래스당 장수 제한")
    ap.add_argument("--fast", action="store_true",
                    help="빠른 모드: 특징을 한 번만 뽑아 분류기만 학습 (수 분, 미세조정 없음)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    t0 = time.time()

    data = Path(args.data)
    tr_paths, tr_ys = list_split(data / "train")
    te_paths, te_ys = list_split(data / "test")
    if not tr_paths or not te_paths:
        raise SystemExit(f"{data} 에 데이터가 없습니다. 먼저 python3 model/prepare_dataset.py 를 실행하세요.")
    if args.limit:
        def cut(ps, ys):
            keep, cnt = [], {}
            for p, y in zip(ps, ys):
                if cnt.get(y, 0) < args.limit:
                    keep.append((p, y))
                    cnt[y] = cnt.get(y, 0) + 1
            return [k[0] for k in keep], [k[1] for k in keep]
        tr_paths, tr_ys = cut(tr_paths, tr_ys)
        te_paths, te_ys = cut(te_paths, te_ys)

    (tr_p, tr_y), (va_p, va_y) = stratified_val(tr_paths, tr_ys, 0.15, args.seed)
    print(f"train {len(tr_p)} / val {len(va_p)} / test {len(te_paths)}  (입력 {args.size}x{args.size}, alpha {args.alpha})")

    train_ds = dataset(tr_p, tr_y, args.size, args.batch, True, args.seed)
    val_ds = dataset(va_p, va_y, args.size, args.batch, False)
    test_ds = dataset(te_paths, te_ys, args.size, args.batch, False)

    weights = None if args.weights == "none" else args.weights
    model, base = build_model(args.size, args.alpha, weights)

    if args.fast:
        fast_head(model, base, tr_p, tr_y, va_p, va_y, args)
    else:
        # 1단계: 분류기만
        fit(model, train_ds, val_ds, args.epochs_head, 1e-3, 6, "1단계 (베이스 고정)")

        # 2단계: 베이스 뒷부분 미세조정 (BatchNorm 은 계속 고정)
        base.trainable = True
        cut_at = int(len(base.layers) * (1 - args.fine_tune_fraction))
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= cut_at and not isinstance(layer, keras.layers.BatchNormalization)
        fit(model, train_ds, val_ds, args.epochs_fine, 2e-5, 8, "2단계 (미세조정)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    y_float = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    m_float = report("float32", te_ys, y_float)

    print("\nint8 양자화 중…", flush=True)
    rng = random.Random(args.seed)
    rep_paths = rng.sample(tr_p, min(300, len(tr_p)))
    blob = to_int8_tflite(model, rep_paths, args.size)
    y_int8, quant = eval_tflite(blob, te_paths, te_ys, args.size)
    m_int8 = report("int8 (ESP32에 들어갈 모델)", te_ys, y_int8)

    (out / "model_int8.tflite").write_bytes(blob)
    (out / "labels.txt").write_text("\n".join(LABELS) + "\n")
    write_header(blob, out / "model_data.h", args.size, quant)
    fw_header = HERE.parent / "firmware" / "smart_bin" / "model_data.h"
    if fw_header.parent.exists():
        write_header(blob, fw_header, args.size, quant)  # 펌웨어가 바로 쓰도록 복사
    model.save(out / "model_float.keras")

    metrics = {
        "dataset": "TrashNet (MIT) → 5 classes",
        "input": [args.size, args.size, 3],
        "alpha": args.alpha,
        "labels": LABELS,
        "train": len(tr_p), "val": len(va_p), "test": len(te_paths),
        "float32": m_float,
        "int8": m_int8,
        "tflite_bytes": len(blob),
        "input_quant": {"scale": quant[0], "zero_point": quant[1]},
        "train_minutes": round((time.time() - t0) / 60, 1),
        "tensorflow": tf.__version__,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n모델 크기 {len(blob) / 1024:.0f} KB · 학습 {metrics['train_minutes']}분")
    print(f"저장 → {out}/ (model_int8.tflite, model_data.h, labels.txt, metrics.json)")
    print("펌웨어 모델도 갱신됨 → firmware/smart_bin/model_data.h (config.h 의 MODEL_VERSION 도 바꿔 주세요)")


if __name__ == "__main__":
    main()
