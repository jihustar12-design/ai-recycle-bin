# AI 스마트 분리수거함 (ai-recycle-bin)

ESP32-S3가 **기기 안에서** 쓰레기를 분류(캔/페트/종이/일반)하고, 서보모터로 해당 칸에 넣은 뒤
분류 로그를 서버 DB에 저장하는 피지컬 AI 프로젝트.

![CI](https://github.com/jihustar12-design/ai-recycle-bin/actions/workflows/ci.yml/badge.svg)

## 필수 조건 충족

| 조건 | 구현 |
|---|---|
| ESP32 모델 사용 | XIAO ESP32S3 Sense + Edge Impulse 이미지 분류 모델(int8)을 온디바이스 추론 |
| 로그를 DB에 저장 | ESP32 → HTTP POST → FastAPI → SQLite (`sort_events`, `feedback`, `device_status`) |
| CI 구현 | GitHub Actions: 서버 테스트, E2E, 펌웨어 로직 테스트, 펌웨어 컴파일, Docker 빌드 |

## 구조

```
[투입] → 초음파 감지 → 카메라 촬영 → 온디바이스 분류 → 판단(신뢰도 기준)
      → 서보로 칸 이동 → 로그 버퍼(오프라인 대비) → 서버 → SQLite → 대시보드
```

```
ai-recycle-bin/
├── firmware/
│   ├── smart_bin/          # Arduino 스케치
│   │   ├── smart_bin.ino   # 메인 루프
│   │   ├── decision.h      # 판단 로직·로그 버퍼·JSON (하드웨어 무관, PC에서 테스트)
│   │   ├── classifier.*    # 분류기 (지금은 가짜, 나중에 Edge Impulse)
│   │   ├── camera.*        # 카메라 초기화
│   │   ├── config.h        # 핀·기준값
│   │   └── secrets.example.h
│   └── tests/              # 판단 로직 단위 테스트 (g++)
├── server/                 # FastAPI + SQLite + pytest
├── dashboard/index.html    # 실시간 대시보드 (서버가 / 에서 제공)
├── tools/fake_device.py    # 가짜 ESP32 (하드웨어 없이 전체 흐름 테스트)
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

# 2) 다른 터미널에서 가짜 장치
python tools/fake_device.py --fail-rate 0.2

# 3) 브라우저: http://localhost:8000  (API 문서: /docs)
```

Docker로 실행: `docker compose up --build`

## 테스트

```bash
cd server && pytest -q && ruff check . ../tools   # 서버
make -C firmware/tests                              # 펌웨어 판단 로직
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
  "model_version": "v0-phone"
}
```

- `label`: `can` | `pet` | `paper` | `general` | `empty`
- `bin`: 0 캔, 1 페트, 2 종이, 3 일반
- `needs_review`: 신뢰도가 기준(0.70) 미만이라 일반 칸으로 보낸 경우

그 외: `POST /api/events/batch`, `GET /api/events`, `GET /api/stats`, `POST /api/feedback`, `POST /api/heartbeat`, `GET /api/devices`

## 진행 상황

- [x] 서버 · DB · 대시보드 · 가짜 장치 · CI
- [x] 펌웨어 골격 (가짜 분류기로 서보/서버 연동 가능)
- [ ] 스마트폰으로 데이터 수집 → Edge Impulse v0 모델
- [ ] (보드) 카메라·서보·센서 동작 확인
- [ ] (보드) ESP32 카메라 데이터로 v1 모델 재학습 → `USE_EDGE_IMPULSE=1`
- [ ] 상자 조립 · 시연 영상

## 보드 연결 (하드웨어 준비 후)

1. Arduino IDE에 ESP32 보드 매니저 추가, 보드 `XIAO_ESP32S3`, PSRAM `OPI PSRAM`
2. 라이브러리 `ESP32Servo` 설치
3. `firmware/smart_bin/secrets.example.h` → `secrets.h` 복사 후 Wi-Fi·서버 주소 입력
4. 배선: 서보 D0, 초음파 TRIG D1 / ECHO D2 (**서보는 별도 5V 전원, GND 공통**)
