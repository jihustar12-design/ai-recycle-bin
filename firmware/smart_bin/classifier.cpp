#include "classifier.h"

#include <Arduino.h>

#include "config.h"

#if USE_TFLITE_MODEL
// ─────────────────────────────────────────────────────────────────────────────
// 실제 모델: model/train.py 로 학습한 int8 MobileNetV2 (model_data.h)
//   카메라 RGB565 → 가운데 크롭 + 160x160 축소(preprocess.h) → TFLite Micro 추론(model_runner)
//   같은 C++ 코드를 PC 에서 test 세트로 검증한다: make -C firmware/tests model
// ─────────────────────────────────────────────────────────────────────────────
#include "esp_heap_caps.h"
#include "model_runner.h"
#include "preprocess.h"

static smartbin::ModelRunner runner;
static bool ready = false;

bool classifierBegin() {
  // 모델 계산 메모리(약 450KB)는 PSRAM 에 둔다
  uint8_t* arena = (uint8_t*)heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!arena) {
    Serial.println("[model] PSRAM arena alloc failed (Tools > PSRAM: OPI PSRAM 확인)");
    return false;
  }
  ready = runner.begin(arena, TENSOR_ARENA_SIZE);
  if (ready) {
    Serial.printf("[model] ready, arena %u/%u bytes\n", (unsigned)runner.arenaUsedBytes(),
                  (unsigned)TENSOR_ARENA_SIZE);
  } else {
    Serial.printf("[model] init failed: %s\n", runner.error());
  }
  return ready;
}

bool classify(camera_fb_t* fb, float* scores) {
  if (!ready || !fb || fb->format != PIXFORMAT_RGB565) return false;
  smartbin::toModelInput(smartbin::Rgb565BE{fb->buf, (int)fb->width}, fb->width, fb->height,
                         runner.inputSize(), runner.inputScale(), runner.inputZeroPoint(), runner.input());
  return runner.invoke(scores);
}

#else
// ─────────────────────────────────────────────────────────────────────────────
// 가짜 분류기: 모델 없이 서보/서버 연동을 확인할 때 (config.h USE_TFLITE_MODEL 0)
// ─────────────────────────────────────────────────────────────────────────────
bool classifierBegin() { return true; }

bool classify(camera_fb_t* fb, float* scores) {
  (void)fb;
  int winner = random(0, smartbin::LBL_GENERAL + 1);
  float top = random(0, 100) < 15 ? random(40, 69) / 100.0f : random(75, 99) / 100.0f;
  float rest = (1.0f - top) / (smartbin::LABEL_COUNT - 1);
  for (int i = 0; i < smartbin::LABEL_COUNT; ++i) scores[i] = (i == winner) ? top : rest;
  return true;
}

#endif
