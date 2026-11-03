# 모델: 공개 데이터(TrashNet)로 학습

## 1. 데이터셋 만들기

```bash
cd ~/Desktop/피지컬ai/ai-recycle-bin
python3 -m pip install pillow        # 처음 한 번
python3 model/prepare_dataset.py
```

결과: `model/dataset/train/` 1,640장, `model/dataset/test/` 410장 (클래스당 410장, 8:2 분할).
파일명 맨 앞이 라벨이에요 (`can.metal12.jpg`). 이 폴더는 `.gitignore` 처리되어 GitHub에 올라가지 않아요.

| 라벨 | 원본 (TrashNet) | 칸 |
|---|---|---|
| can | metal | 0 |
| pet | plastic | 1 |
| paper | paper + cardboard | 2 |
| general | trash + glass | 3 |
| empty | 사진의 빈 배경을 잘라 합성 | (동작 안 함) |

출처: TrashNet — Gary Thung, Mindy Yang (Stanford CS229, MIT License) https://github.com/garythung/trashnet

## 2. 맥에서 학습 (권장)

Edge Impulse 무료 계정은 학습 시간이 **월 25분**이라 160×160 고정확도 설정을 끝까지 돌릴 수 없어요.
그래서 맥에서 직접 학습합니다. 시간 제한이 없고 설정도 마음대로 바꿀 수 있어요.

```bash
cd ~/Desktop/피지컬ai/ai-recycle-bin
python3 -m venv model/.venv
source model/.venv/bin/activate
pip install -r model/requirements.txt     # 처음 한 번 (TensorFlow, 몇 분 걸림)
python model/train.py --fast     # 빠른 모드 (약 3~5분) — 먼저 이걸로
python model/train.py            # 전체 모드 (20~40분) — 정확도를 더 올리고 싶을 때
```

| 모드 | 방식 | 시간 | 정확도 |
|---|---|---|---|
| `--fast` | 사전학습 모델로 특징을 한 번만 뽑고 분류기만 학습 | 약 3~5분 | 조금 낮을 수 있음 |
| 기본 | 매 epoch 증강 + 베이스 미세조정 | 20~40분 | 가장 높음 |

- 1단계(베이스 고정) → 2단계(미세조정) 순서로 자동 진행, 성능이 더 안 오르면 일찍 멈춤
- 끝나면 test 세트 **정확도와 혼동 행렬**을 float32 / int8(ESP32용) 둘 다 출력
- 결과물: `model/export/`
  - `model_int8.tflite` — ESP32에 들어갈 모델
  - `model_data.h` — 펌웨어에 바로 include 하는 C 배열
  - `metrics.json` — 정확도·혼동 행렬 (발표 자료용)

옵션 예시

```bash
python model/train.py --size 96      # 더 빠른 추론 (정확도 약간 하락)
python model/train.py --alpha 0.5    # 더 정확 (모델 커지고 ESP32에서 느려짐)
```

> Edge Impulse 프로젝트(`ai-recycle-bin`)에는 데이터와 Impulse(160×160, MobileNetV2 0.35) 설정이 올라가 있어요.
> 학습 시간이 초기화되면(매월) 거기서 비교 학습을 해 볼 수도 있어요.

## 3. 정확도 확인

```bash
python model/evaluate.py                       # int8 모델로 test 410장 분류 → 정확도·혼동 행렬
make -C firmware/tests model TFLM=<esp-tflite-micro 경로> DUMP=<evaluate.py --dump 폴더>   # 펌웨어 C++ 코드로 같은 평가
```

CI(`model` 작업)가 push 할 때마다 두 가지를 모두 실행하고, 정확도가 78% 아래로 떨어지면 실패해요.
결과는 GitHub Actions 실행 화면의 Summary 에 표로 나와요.

## 4. 정확도를 최대로 끌어올리는 포인트

1. **상자 안 촬영 환경을 TrashNet과 똑같이 만들기** (가장 중요)
   - 바닥을 **흰색 무광 보드**로 (TrashNet이 흰 포스터보드 위에서 촬영됨)
   - 위/앞에서 고르게 비추는 **흰색 LED**, 그림자 최소화
   - 카메라는 물체 전체가 화면의 50~80%를 차지하는 거리
2. **학습과 추론의 전처리를 같게** — 4:3 이미지 → 가운데 정사각형 크롭 → 160×160 (train.py 와 펌웨어가 동일하게)
3. **신뢰도 0.70 미만은 일반 칸 + 확인 필요** (`config.h` CONF_THRESHOLD). 대시보드 오분류 신고로 약점 파악
4. 버전 기록: `config.h` 의 `MODEL_VERSION` 을 `v1-trashnet-160` 처럼 바꿔 대시보드에서 비교

## 5. 알려진 한계

- TrashNet은 미국 제품 위주라 한국 캔·페트 라벨과 모양이 다를 수 있음
- `plastic`에는 페트병 외 플라스틱 용기도 섞여 있음
- `general`의 2/3가 유리병이라 과자 봉지 같은 비닐류는 약할 수 있음 → 낮은 신뢰도로 일반 칸에 가는 경우가 많음
- 카메라 형식(RGB565, 색 단계가 적음)에서는 PC 측정 기준 약 3%p 낮아짐 (82.2% → 78.5%)
