#pragma once
#include "decision.h"
#include "esp_camera.h"

// 부팅 시 한 번 호출 (모델 로드). 성공하면 true.
bool classifierBegin();

// 촬영된 프레임을 분류해 라벨별 점수를 scores[LABEL_COUNT] (Label enum 순서) 에 채운다. 성공하면 true.
// USE_TFLITE_MODEL=0 이면 가짜 점수를 만든다 (모델 없이 서보/서버 연동을 먼저 확인하기 위함).
bool classify(camera_fb_t* fb, float* scores);
