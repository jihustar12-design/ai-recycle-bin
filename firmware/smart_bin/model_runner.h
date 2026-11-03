// TensorFlow Lite Micro 로 int8 모델을 실행한다.
// Arduino ESP32 코어(3.x)에 포함된 esp-tflite-micro 를 사용하므로 별도 라이브러리 설치가 필요 없다.
// Arduino 헤더에 의존하지 않아 PC 에서도 같은 코드로 테스트한다 (firmware/tests/test_model.cpp).
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace smartbin {

class ModelRunner {
 public:
  // arena: 모델 계산용 메모리 (ESP32 에서는 PSRAM). 성공하면 true.
  bool begin(uint8_t* arena, size_t arenaSize);

  int8_t* input();           // size*size*3 int8 (HWC RGB)
  int inputSize() const;     // 160
  float inputScale() const;
  int inputZeroPoint() const;

  // 추론 후 scores[LABEL_COUNT] 를 Label enum 순서로 채운다 (모델 라벨 순서와 무관하게 이름으로 매핑).
  bool invoke(float* scores);

  size_t arenaUsedBytes() const;
  const char* error() const { return error_; }

 private:
  void* interpreter_ = nullptr;
  const char* error_ = "not started";
};

}  // namespace smartbin
