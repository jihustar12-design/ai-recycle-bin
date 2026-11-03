// 보드/핀/동작 설정. 기준 보드: Seeed XIAO ESP32S3 Sense
#pragma once

#define FW_VERSION    "0.1.0"
// 1: 학습한 TFLite 모델(model_data.h) 사용 / 0: 가짜 분류기(모델 없이 서보·서버 연동 테스트)
#ifndef USE_TFLITE_MODEL
#define USE_TFLITE_MODEL 1
#endif

#if USE_TFLITE_MODEL
#define MODEL_VERSION "v1-trashnet-160-fast"  // 모델을 다시 학습하면 바꿔서 대시보드에서 비교
#else
#define MODEL_VERSION "stub"
#endif

// 모델 계산 메모리 (PC 측정 445KB + 여유). PSRAM 에 할당
#define TENSOR_ARENA_SIZE (600 * 1024)

// 판단 기준
#define CONF_THRESHOLD 0.70f   // tools/fake_device.py 와 동일하게 유지

// 핀 (XIAO ESP32S3 기준 GPIO 번호)
#define PIN_SERVO     1   // D0
#define PIN_US_TRIG   2   // D1  초음파 TRIG
#define PIN_US_ECHO   3   // D2  초음파 ECHO (5V 센서면 분압 필요)
#define PIN_LED       21  // 내장 LED (LOW = 켜짐)

// 투입 감지: 이 거리(cm)보다 가까우면 물체가 들어온 것으로 판단
#define DETECT_CM          12
#define DETECT_COOLDOWN_MS 2500

// 칸별 서보 각도 (0:캔 1:페트 2:종이 3:일반)
static const int BIN_ANGLES[4] = {20, 70, 120, 170};
#define SERVO_HOME_ANGLE 95
#define SERVO_HOLD_MS    900

// 로그
#define SEND_IMAGES           1   // 분류 순간의 사진을 대시보드로 보낼지
#define JPEG_QUALITY          80  // 0~100, 높을수록 선명 (QVGA 기준 약 10~25KB)
#define LOG_BUFFER_SIZE       64
#define HEARTBEAT_INTERVAL_MS 30000
#define HTTP_TIMEOUT_MS       3000
