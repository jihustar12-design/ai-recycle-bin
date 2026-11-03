#include "classifier.h"

#include <Arduino.h>

#include "config.h"

#if USE_EDGE_IMPULSE
// ─────────────────────────────────────────────────────────────────────────────
// Edge Impulse 모델 연결 (보드를 다시 잡은 뒤 진행)
//  1. Edge Impulse Studio → Deployment → "Arduino library" 로 zip 다운로드
//  2. Arduino IDE: 스케치 → 라이브러리 포함 → .ZIP 라이브러리 추가
//  3. 아래 include 이름을 내보낸 라이브러리 이름에 맞게 변경
//  4. Edge Impulse 예제(esp32_camera)의 ei_camera_get_data / 리사이즈 코드를 참고해 구현
//  5. config.h 의 USE_EDGE_IMPULSE 를 1, MODEL_VERSION 을 "v0-phone" 등으로 변경
//  * 라벨 순서는 decision.h 의 Label enum 과 같아야 한다 (ei_classifier_inferencing_categories 확인)
// ─────────────────────────────────────────────────────────────────────────────
#error "Edge Impulse 라이브러리를 연결한 뒤 이 줄을 지우고 classify() 를 구현하세요"
#else

bool classify(camera_fb_t* fb, float* scores) {
  (void)fb;
  // 가짜 분류기: 한 라벨에 높은 점수, 가끔 낮은 신뢰도도 섞음
  int winner = random(0, smartbin::LBL_GENERAL + 1);
  float top = random(0, 100) < 15 ? random(40, 69) / 100.0f : random(75, 99) / 100.0f;
  float rest = (1.0f - top) / (smartbin::LABEL_COUNT - 1);
  for (int i = 0; i < smartbin::LABEL_COUNT; ++i) scores[i] = (i == winner) ? top : rest;
  return true;
}

#endif
