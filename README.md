# AI 스마트 분리수거함 (ai-recycle-bin)

ESP32-S3가 **기기 안에서** 쓰레기를 분류(캔/페트/종이/일반)하고, 서보모터로 해당 칸에 넣은 뒤
분류 로그를 서버 DB에 저장하는 피지컬 AI 프로젝트.

![CI](https://github.com/jihustar12-design/ai-recycle-bin/actions/workflows/ci.yml/badge.svg)

## 필수 조건 충족

| 조건 | 구현 |
|---|---|
| ESP32 모델 사용 | XIAO ESP32S3 Sense 에서 직접 학습한 MobileNetV2 int8 모델(634KB)을 TFLite Micro로 **온디바이스 추론** (Wi-Fi 없이도 분류) |
| 로그를 DB에 저장 | ESP32 → HTTP POST → FastAPI → SQLite (`sort_events`, `feedback`, `device_status`) |
| CI 구현 | GitHub Actions: 서버 테스트, E2E(가짜 장치→서버→DB), 펌웨어 로직 테스트, 펌웨어 컴파일(모델/가짜 2종), **모델 정확도 검증**, Docker 빌드 |

## 모델 성능

TrashNet(Stanford, MIT License) 2,527장을 5개 클래스로 바꿔 학습 1,640장 / **test 410장**으로 평가했어요.
ESP32에 들어가는 **int8 모델 기준**이고, 아래 수치는 CI(`model` 작업)가 매 push마다 test 세트로 직접 측정한 값이에요.
(학습 직후 TensorFlow 기준으로는 81.5% — JPEG 디코더 차이로 1장 정도 달라짐)

| 항목 | 값 |
|---|---|
| **정확도 (int8, ESP32용)** | **81.2%** (333/410, CI 측정) |
| 모델 | MobileNetV2 α0.35 전이학습, 입력 160×160 RGB, **634 KB** |
| 추론 메모리 | 445 KB (PSRAM) |
| 참고: TrashNet 원 논문 CNN | 약 75% (6개 클래스) |

혼동 행렬 (행: 실제, 열: 예측)

| 실제 \ 예측 | can | empty | general | paper | pet |
|---|---|---|---|---|---|
| **can** | 66 | 0 | 11 | 3 | 2 |
| **empty** | 0 | 77 | 0 | 5 | 0 |
| **general** | 7 | 0 | 58 | 9 | 8 |
| **paper** | 2 | 0 | 3 | 77 | 0 |
| **pet** | 2 | 0 | 21 | 4 | 55 |

- 가장 큰 약점은 **페트 → 일반(21장)**: 일반 클래스에 섞인 유리병과 투명 페트병을 헷갈림
- 신뢰도 70% 미만은 일반 칸 + `needs_review`로 기록 → 대시보드에서 오분류 신고로 확인
- **펌웨어 C++ 코드로 PC에서 같은 410장을 분류한 결과**(TFLite Micro): RGB888 82.2%, 카메라 형식 RGB565 78.5%, Python 결과와 예측 97.8% 일치
- 수치는 TrashNet 촬영 환경(흰 배경·밝은 조명) 기준. 실제 상자 안 정확도는 보드 연결 후 별도 측정 예정

학습·평가 방법: [`model/README.md`](model/README.md)

## 구조

```
[투입] → 초음파 감지 → 카메라 촬영 → 온디바이스 분류 → 판단(신뢰도 기준)
      → 서보로 칸 이동 → 로그 버퍼(오프라인 대비) → 서버 → SQLite → 대시보드
```

```
ai-recycle-bin/
├── model/
│   ├── prepare_dataset.py  # TrashNet → 5클래스 데이터셋
│   ├── train.py            # 맥에서 학습 → int8 TFLite + model_data.h
│   ├── evaluate.py         # int8 모델 정확도 측정 (CI)
│   └── export/             # model_int8.tflite, metrics.json
├── firmware/
│   ├── smart_bin/          # Arduino 스케치
│   │   ├── smart_bin.ino   # 메인 루프
│   │   ├── decision.h      # 판단 로직·로그 버퍼·JSON (하드웨어 무관, PC에서 테스트)
│   │   ├── classifier.*    # 분류기 (USE_TFLITE_MODEL 1: 실제 모델 / 0: 가짜)
│   │   ├── preprocess.h    # RGB565 → 크롭·160×160·int8 (PC에서 테스트)
│   │   ├── model_runner.*  # TFLite Micro 추론
│   │   ├── model_data.h    # 학습된 모델 (train.py 가 생성)
│   │   ├── camera.*        # 카메라 초기화
│   │   ├── config.h        # 핀·기준값
│   │   └── secrets.example.h
│   └── tests/              # 판단 로직 + 모델 추론 테스트 (g++, TFLite Micro)
├── server/                 # FastAPI + SQLite + pytest
├── dashboard/index.html    # 실시간 대시보드 (서버가 / 에서 제공)
├── tools/
│   ├── fake_device.py      # 가짜 ESP32 (사진 전송, --use-model 로 실제 모델 분류)
│   └── samples/            # 샘플 사진 13장 (TrashNet)
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## 실행 (하드웨어 없이)

```bash
# 1) 서버
cd server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000

# 2) 다른 터미널에서 가짜 장치 (샘플 사진과 함께 전송)
python tools/fake_device.py --fail-rate 0.2
#    실제 모델로 분류한 결과를 보려면 (학습 환경 model/.venv 에서)
python tools/fake_device.py --use-model --interval 3

# 3) 브라우저: http://localhost:8000  (API 문서: /docs)
```

Docker로 실행: `docker compose up --build`

## 테스트

```bash
cd server && pytest -q && ruff check . ../tools ../model   # 서버
make -C firmware/tests                                       # 펌웨어 판단 로직
python model/evaluate.py --min-accuracy 0.78                 # 모델 정확도 (model/README.md)
```

## 메시지 규격 (ESP32 → 서버)

`POST /api/events`

```json
{
  "device_id": "bin-01",
  "label": "pet",
  "confidence": 0.91,
  "latency_ms": 180,
  "bin": 1,
  "needs_review": false,
  "fw_version": "0.1.0",
  "model_version": "v1-trashnet-160-fast"
}
```

- `label`: `can` | `pet` | `paper` | `general` | `empty`
- `bin`: 0 캔, 1 페트, 2 종이, 3 일반
- `needs_review`: 신뢰도가 기준(0.70) 미만이라 일반 칸으로 보낸 경우

분류 순간의 사진: 이벤트 직후 `POST /api/events/{id}/image` (body = JPEG, `Content-Type: image/jpeg`, 최대 300KB).
대시보드의 **"방금 들어온 쓰레기"** 카드와 이벤트 표에 사진이 표시돼요. Wi-Fi가 끊겨 버퍼에 쌓였던 로그는 사진 없이 전송돼요.

그 외: `POST /api/events/batch`, `GET /api/events`, `GET /api/events/{id}/image`, `GET /api/stats`, `POST /api/feedback`, `POST /api/heartbeat`, `GET /api/devices`

## 진행 상황

- [x] 서버 · DB · 대시보드 · 가짜 장치 · CI
- [x] 펌웨어 골격 (가짜 분류기로 서보/서버 연동 가능)
- [x] 공개 데이터(TrashNet) 데이터셋 구성 → `model/README.md`
- [x] 모델 학습 (int8 81.2%) + 펌웨어 연결(TFLite Micro) + CI 정확도 검증
- [ ] (보드) 카메라·서보·센서 동작 확인
- [ ] (보드) 실제 상자 안 정확도·추론 속도 측정
- [ ] 상자 조립 · 시연 영상

## 보드 연결 (하드웨어 준비 후)

1. Arduino IDE에 ESP32 보드 매니저 추가, 보드 `XIAO_ESP32S3`, PSRAM `OPI PSRAM`
2. 라이브러리 `ESP32Servo` 설치 (TFLite Micro 는 ESP32 코어 3.x 에 포함되어 있어 설치 불필요)
3. `firmware/smart_bin/secrets.example.h` → `secrets.h` 복사 후 Wi-Fi·서버 주소 입력
4. 배선: 서보 D0, 초음파 TRIG D1 / ECHO D2 (**서보는 별도 5V 전원, GND 공통**)
5. 모델 없이 서보·서버만 먼저 확인하려면 `config.h` 의 `USE_TFLITE_MODEL` 을 0으로
