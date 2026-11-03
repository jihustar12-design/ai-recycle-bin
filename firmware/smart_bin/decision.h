// 하드웨어와 무관한 판단 로직. Arduino 헤더를 쓰지 않으므로 PC에서 g++로 단위 테스트할 수 있다.
// (firmware/tests/test_decision.cpp, CI의 firmware-logic 잡)
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace smartbin {

// 주의: 모델(model/train.py, Edge Impulse 모두)의 라벨은 알파벳순(can, empty, general, paper, pet)이다.
// 순서가 이 enum 과 다르므로, 결과는 반드시 labelFromName() 으로 '이름'을 기준으로 옮길 것 (model_runner.cpp).
enum Label : uint8_t { LBL_CAN = 0, LBL_PET, LBL_PAPER, LBL_GENERAL, LBL_EMPTY, LABEL_COUNT };

inline const char* labelName(Label l) {
  static const char* names[] = {"can", "pet", "paper", "general", "empty"};
  return l < LABEL_COUNT ? names[l] : "general";
}

inline int labelFromName(const char* s) {
  for (int i = 0; i < LABEL_COUNT; ++i)
    if (strcmp(s, labelName(static_cast<Label>(i))) == 0) return i;
  return -1;
}

// 라벨 → 서보가 향할 칸 번호
inline uint8_t binFor(Label l) {
  switch (l) {
    case LBL_CAN:   return 0;
    case LBL_PET:   return 1;
    case LBL_PAPER: return 2;
    default:    return 3;  // general
  }
}

struct Decision {
  Label label;        // 최고 점수 라벨
  float confidence;   // 최고 점수
  uint8_t bin;        // 실제로 보낼 칸
  bool needsReview;   // 신뢰도가 낮아 일반 칸으로 보냈는지
  bool actuate;       // 서보를 움직여야 하는지 (빈칸이면 false)
};

// scores: 라벨별 확률 (LABEL_COUNT개). threshold 미만이면 일반 칸 + 확인 필요.
inline Decision decide(const float* scores, size_t n, float threshold) {
  size_t best = 0;
  for (size_t i = 1; i < n; ++i)
    if (scores[i] > scores[best]) best = i;

  Decision d;
  d.label = best < LABEL_COUNT ? static_cast<Label>(best) : LBL_GENERAL;
  d.confidence = scores[best];
  if (d.label == LBL_EMPTY) {
    d.bin = binFor(LBL_GENERAL);
    d.needsReview = false;
    d.actuate = false;
    return d;
  }
  d.needsReview = d.confidence < threshold;
  d.bin = d.needsReview ? binFor(LBL_GENERAL) : binFor(d.label);
  d.actuate = true;
  return d;
}

struct SortEvent {
  Label label;
  float confidence;
  uint32_t latencyMs;
  uint8_t bin;
  bool needsReview;
};

// Wi-Fi가 끊겼을 때 로그를 잃지 않도록 보관하는 원형 버퍼.
// 가득 차면 가장 오래된 로그를 버리고 dropped() 를 증가시킨다.
template <size_t N>
class LogBuffer {
 public:
  bool push(const SortEvent& e) {
    bool overwrote = false;
    if (count_ == N) {
      head_ = (head_ + 1) % N;
      --count_;
      ++dropped_;
      overwrote = true;
    }
    items_[(head_ + count_) % N] = e;
    ++count_;
    return !overwrote;
  }
  const SortEvent& front() const { return items_[head_]; }
  void pop() {
    if (count_ == 0) return;
    head_ = (head_ + 1) % N;
    --count_;
  }
  size_t size() const { return count_; }
  bool empty() const { return count_ == 0; }
  uint32_t dropped() const { return dropped_; }

 private:
  SortEvent items_[N];
  size_t head_ = 0, count_ = 0;
  uint32_t dropped_ = 0;
};

// 서버 규격(server/app/schemas.py SortEventIn)에 맞춘 JSON. 반환값: 쓴 길이(잘렸으면 -1)
inline int toJson(const SortEvent& e, const char* deviceId, const char* fw, const char* model,
                  char* out, size_t len) {
  int n = snprintf(out, len,
                   "{\"device_id\":\"%s\",\"label\":\"%s\",\"confidence\":%.3f,"
                   "\"latency_ms\":%u,\"bin\":%u,\"needs_review\":%s,"
                   "\"fw_version\":\"%s\",\"model_version\":\"%s\"}",
                   deviceId, labelName(e.label), e.confidence, (unsigned)e.latencyMs, (unsigned)e.bin,
                   e.needsReview ? "true" : "false", fw, model);
  return (n < 0 || (size_t)n >= len) ? -1 : n;
}

}  // namespace smartbin
