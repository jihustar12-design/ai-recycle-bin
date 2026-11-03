"""공개 데이터(TrashNet)로 Edge Impulse 학습용 데이터셋을 만든다.

TrashNet (Gary Thung & Mindy Yang, Stanford CS229, MIT License)
  https://github.com/garythung/trashnet
  6클래스 2,527장, 흰 보드 위에서 촬영 (512x384)

이 프로젝트의 5클래스로 변환:
  can     ← metal
  pet     ← plastic
  paper   ← paper + cardboard
  general ← trash + glass   (분리수거함에 유리 칸이 없으므로 일반으로 보냄)
  empty   ← TrashNet 사진의 빈 배경 부분을 잘라 합성 (아무것도 안 들어간 상태)

출력 (Edge Impulse 업로드용, 파일명 맨 앞이 라벨):
  model/dataset/train/<label>.<원본>.jpg
  model/dataset/test/<label>.<원본>.jpg
  model/dataset/manifest.csv

사용:  python3 model/prepare_dataset.py            (Pillow 필요)
"""
import argparse
import csv
import io
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageStat

URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
HERE = Path(__file__).resolve().parent

# (라벨, 원본 폴더, 최대 장수)  — 클래스별 장수를 비슷하게 맞춰 한쪽으로 치우치지 않게 한다
SOURCES = [
    ("can", "metal", 410),
    ("pet", "plastic", 410),
    ("paper", "paper", 250),
    ("paper", "cardboard", 160),
    ("general", "trash", 137),   # 전부 사용
    ("general", "glass", 273),
]
N_EMPTY = 410
OUT_SIZE = (320, 240)  # ESP32 카메라 QVGA 와 같은 4:3 비율
TEST_RATIO = 0.2


def download(cache: Path) -> Path:
    if cache.exists() and zipfile.is_zipfile(cache):
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"TrashNet 다운로드 중 (~43MB)… {URL}")
    with urllib.request.urlopen(URL, timeout=300) as r:
        data = r.read()
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise SystemExit("다운로드한 파일이 zip이 아닙니다. 네트워크를 확인하세요.")
    cache.write_bytes(data)
    return cache


def load_index(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = defaultdict(list)
    for name in zf.namelist():
        if name.startswith("__MACOSX") or not name.lower().endswith(".jpg"):
            continue
        parts = name.split("/")
        if len(parts) >= 3:
            idx[parts[-2]].append(name)
    for v in idx.values():
        v.sort()
    return idx


def open_img(zf: zipfile.ZipFile, name: str) -> Image.Image:
    return Image.open(io.BytesIO(zf.read(name))).convert("RGB")


def make_empty(zf, all_names, rng: random.Random, n: int) -> list[Image.Image]:
    """원본 사진 네 귀퉁이 중 물체가 없는(밝고 균일한) 부분을 잘라 빈 바닥 이미지를 만든다."""
    out: list[Image.Image] = []
    tries = 0
    while len(out) < n and tries < n * 30:
        tries += 1
        im = open_img(zf, rng.choice(all_names))
        w, h = im.size
        pw, ph = w // 4, h // 4
        x = rng.choice([0, w - pw])
        y = rng.choice([0, h - ph])
        patch = im.crop((x, y, x + pw, y + ph))
        st = ImageStat.Stat(patch.convert("L"))
        if st.mean[0] < 140 or st.stddev[0] > 14:  # 어둡거나 무늬가 있으면 물체 일부로 보고 버림
            continue
        img = patch.resize(OUT_SIZE, Image.BICUBIC)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.15))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.2)))
        out.append(img)
    if len(out) < n:
        print(f"경고: empty 이미지를 {len(out)}장만 만들었습니다")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE / "dataset"))
    p.add_argument("--cache", default=str(HERE / ".cache" / "trashnet-resized.zip"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "train").mkdir(parents=True)
    (out / "test").mkdir(parents=True)

    zf = zipfile.ZipFile(download(Path(args.cache)))
    idx = load_index(zf)

    rows = []
    counts: Counter = Counter()

    def save(img: Image.Image, label: str, stem: str, split: str, source: str) -> None:
        fn = f"{label}.{stem}.jpg"
        img.resize(OUT_SIZE, Image.BICUBIC).save(out / split / fn, quality=90)
        rows.append({"split": split, "label": label, "file": f"{split}/{fn}", "source": source})
        counts[(split, label)] += 1

    # 원본 사진 단위로 train/test 를 나눈다 (같은 사진이 양쪽에 들어가지 않음)
    for label, folder, cap in SOURCES:
        names = idx[folder][:]
        rng.shuffle(names)
        names = names[:cap]
        n_test = round(len(names) * TEST_RATIO)
        for i, name in enumerate(names):
            split = "test" if i < n_test else "train"
            stem = Path(name).stem  # 예: metal12
            save(open_img(zf, name), label, stem, split, f"trashnet/{folder}")

    all_names = [n for v in idx.values() for n in v]
    empties = make_empty(zf, all_names, rng, N_EMPTY)
    n_test = round(len(empties) * TEST_RATIO)
    for i, img in enumerate(empties):
        save(img, "empty", f"bg{i:04d}", "test" if i < n_test else "train", "trashnet/background-crop")

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "label", "file", "source"])
        w.writeheader()
        w.writerows(rows)

    labels = sorted({lb for _, lb in counts})
    print(f"\n{'label':<8} {'train':>6} {'test':>6}")
    for lb in labels:
        print(f"{lb:<8} {counts[('train', lb)]:>6} {counts[('test', lb)]:>6}")
    print(f"{'total':<8} {sum(v for (s, _), v in counts.items() if s == 'train'):>6} "
          f"{sum(v for (s, _), v in counts.items() if s == 'test'):>6}")
    print(f"\n완료 → {out}")


if __name__ == "__main__":
    main()
