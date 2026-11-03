# 모델: 공개 데이터(TrashNet) 기반 학습

## 1. 데이터셋 생성

```bash
python3 -m pip install pillow        # 최초 1회
python3 model/prepare_dataset.py
```

- 결과: `model/dataset/train/` 1,640장, `model/dataset/test/` 410장 (클래스당 410장, 8:2 분할)
- 파일명 앞부분이 라벨임 (예: `can.metal12.jpg`)
- 학습·평가 분할은 원본 사진 단위로 수행하여 동일 사진이 양쪽에 포함되지 않도록 함
- 생성된 데이터는 `.gitignore` 대상이며, 스크립트로 동일하게 재생성 가능 (seed 고정)

| 라벨 | 원본 (TrashNet) | 투입 칸 |
|---|---|---|
| can | metal | 0 |
| pet | plastic | 1 |
| paper | paper + cardboard | 2 |
| general | trash + glass | 3 |
| empty | 원본 사진의 빈 배경을 잘라 합성 | 동작 안 함 |

출처: TrashNet (Gary Thung, Mindy Yang, Stanford CS229), MIT License, https://github.com/garythung/trashnet

## 2. 학습

Edge Impulse 무료 계정은 학습 시간이 월 25분으로 제한되어 160×160 설정의 학습을 완료할 수 없었습니다.
따라서 로컬(macOS)에서 TensorFlow로 학습했습니다.

```bash
python3 -m venv model/.venv          # Python 3.10~3.12 사용 (TensorFlow 지원 버전)
source model/.venv/bin/activate
pip install -r model/requirements.txt
python model/train.py --fast         # 빠른 모드 (약 3~5분) ← 제출 모델
python model/train.py                # 전체 모드 (20~40분, 미세조정 포함)
```

| 모드 | 방식 | 소요 시간 | 비고 |
|---|---|---|---|
| `--fast` | 사전학습 모델로 특징을 1회 추출한 뒤 분류기만 학습 (뒤집기·회전 증강 포함) | 약 3~5분 | **제출 모델** |
| 기본 | 1단계 분류기 학습 → 2단계 베이스 모델 뒷부분 미세조정, 매 epoch 증강 | 20~40분 | 정확도 향상 여지 |

- 공통 설정: MobileNetV2 α0.35 (ImageNet 사전학습), 입력 160×160 RGB, label smoothing 0.05, early stopping
- 학습 종료 시 test 세트에 대한 정확도와 혼동 행렬을 float32 / int8 모델 각각 출력함
- 산출물 (`model/export/`)
  - `model_int8.tflite`: ESP32 탑재용 전정수(int8) 양자화 모델
  - `model_data.h`: 펌웨어에서 include하는 C 배열 (`firmware/smart_bin/model_data.h`에도 자동 복사)
  - `metrics.json`: 정확도, 혼동 행렬, 클래스별 precision/recall

옵션 예시

```bash
python model/train.py --size 96      # 입력 축소: 추론 속도 향상, 정확도 소폭 하락
python model/train.py --alpha 0.5    # 모델 확대: 정확도 향상, 모델 크기·추론 시간 증가
```

## 3. 정확도 검증

```bash
python model/evaluate.py --min-accuracy 0.78
make -C firmware/tests model TFLM=<esp-tflite-micro 경로> DUMP=<evaluate.py --dump 출력 폴더>
```

- `evaluate.py`: 저장소의 int8 모델로 test 410장을 분류하여 정확도와 혼동 행렬을 출력함. 전처리는 펌웨어와 동일한 알고리즘을 numpy로 구현함
- `make ... model`: 펌웨어 C++ 코드(`preprocess.h`, `model_runner.cpp`)와 TFLite Micro로 동일 평가를 수행하고, RGB565(카메라 형식) 입력 결과도 함께 측정함
- CI의 `Model accuracy` 작업이 push마다 두 검증을 모두 실행하며, 정확도가 78% 미만이면 실패 처리함. 결과는 GitHub Actions 실행 화면의 Summary에 표로 출력됨

| 측정 | 결과 |
|---|---|
| int8 모델 (Python, CI) | 81.2% |
| int8 모델 (학습 직후 TensorFlow) | 81.5% |
| float32 모델 | 84.4% |
| 펌웨어 C++ / TFLite Micro, RGB888 입력 | 82.2% |
| 펌웨어 C++ / TFLite Micro, RGB565 입력 | 78.5% |
| Python과 C++ 예측 일치율 | 97.8% |

## 4. 정확도 확보를 위한 장치 구성 지침

1. **장치 내부 촬영 환경을 TrashNet과 유사하게 구성** (가장 중요)
   - 바닥: 흰색 무광 보드 (TrashNet은 흰 포스터보드 위에서 촬영됨)
   - 조명: 위/앞에서 고르게 비추는 흰색 LED, 그림자 최소화
   - 거리: 물체가 화면의 50~80%를 차지하도록 카메라 배치
2. **학습·추론 전처리 일치**: 4:3 이미지 → 가운데 정사각형 크롭 → 160×160 (train.py, evaluate.py, 펌웨어 동일)
3. **신뢰도 기준**: 0.70 미만은 일반 칸 투입 및 `needs_review` 기록 (`config.h`의 `CONF_THRESHOLD`)
4. **버전 관리**: 재학습 시 `config.h`의 `MODEL_VERSION`을 변경하여 대시보드에서 모델별 성능 비교

## 5. 한계

- TrashNet은 해외 제품 위주로 구성되어 국내 캔·페트병의 라벨 및 형태와 차이가 있을 수 있음
- `plastic` 클래스에 페트병 외 플라스틱 용기가 포함되어 있음
- `general` 클래스의 약 2/3가 유리병이어서 비닐류 등 일반 쓰레기 인식이 약할 수 있으며, 이 경우 낮은 신뢰도로 일반 칸에 투입됨
- 카메라 형식(RGB565, 색 단계 감소)에서는 PC 측정 기준 약 3.7%p 낮게 측정됨 (82.2% → 78.5%)
