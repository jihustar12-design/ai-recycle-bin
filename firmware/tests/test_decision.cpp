// PC에서 실행하는 판단 로직 단위 테스트: make -C firmware/tests
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../smart_bin/decision.h"

using namespace smartbin;

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);          \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

static void test_high_confidence_goes_to_its_bin() {
  float s[LABEL_COUNT] = {0.02f, 0.92f, 0.03f, 0.02f, 0.01f};
  Decision d = decide(s, LABEL_COUNT, 0.7f);
  CHECK(d.label == LBL_PET);
  CHECK(d.bin == 1);
  CHECK(!d.needsReview);
  CHECK(d.actuate);
}

static void test_low_confidence_goes_to_general_and_flags_review() {
  float s[LABEL_COUNT] = {0.45f, 0.30f, 0.20f, 0.03f, 0.02f};
  Decision d = decide(s, LABEL_COUNT, 0.7f);
  CHECK(d.label == LBL_CAN);
  CHECK(d.bin == binFor(LBL_GENERAL));
  CHECK(d.needsReview);
  CHECK(d.actuate);
}

static void test_empty_does_not_actuate() {
  float s[LABEL_COUNT] = {0.01f, 0.01f, 0.01f, 0.02f, 0.95f};
  Decision d = decide(s, LABEL_COUNT, 0.7f);
  CHECK(d.label == LBL_EMPTY);
  CHECK(!d.actuate);
}

static void test_bin_mapping() {
  CHECK(binFor(LBL_CAN) == 0);
  CHECK(binFor(LBL_PET) == 1);
  CHECK(binFor(LBL_PAPER) == 2);
  CHECK(binFor(LBL_GENERAL) == 3);
}

static void test_label_names_roundtrip() {
  for (int i = 0; i < LABEL_COUNT; ++i) CHECK(labelFromName(labelName((Label)i)) == i);
  CHECK(labelFromName("glass") == -1);
}

// Edge Impulse 는 알파벳순으로 결과를 준다 → 이름으로 매핑해야 올바른 칸으로 간다
static void test_edge_impulse_alphabetical_mapping() {
  const char* eiLabels[] = {"can", "empty", "general", "paper", "pet"};
  const float eiValues[] = {0.02f, 0.01f, 0.03f, 0.04f, 0.90f};
  float scores[LABEL_COUNT] = {0};
  for (int i = 0; i < 5; ++i) {
    int idx = labelFromName(eiLabels[i]);
    CHECK(idx >= 0);
    if (idx >= 0) scores[idx] = eiValues[i];
  }
  Decision d = decide(scores, LABEL_COUNT, 0.7f);
  CHECK(d.label == LBL_PET);
  CHECK(d.bin == 1);
}

static void test_buffer_keeps_order_and_drops_oldest() {
  LogBuffer<3> buf;
  for (int i = 0; i < 5; ++i) {
    SortEvent e{LBL_PET, 0.9f, (uint32_t)i, 1, false};
    buf.push(e);
  }
  CHECK(buf.size() == 3);
  CHECK(buf.dropped() == 2);
  CHECK(buf.front().latencyMs == 2);  // 0,1 은 버려짐
  buf.pop();
  CHECK(buf.front().latencyMs == 3);
  buf.pop();
  buf.pop();
  CHECK(buf.empty());
  buf.pop();  // 빈 버퍼 pop 은 안전해야 함
  CHECK(buf.empty());
}

static void test_json_matches_server_schema() {
  SortEvent e{LBL_PAPER, 0.8765f, 182, 2, false};
  char out[256];
  int n = toJson(e, "bin-01", "0.1.0", "v1-esp32", out, sizeof(out));
  CHECK(n > 0);
  CHECK(strstr(out, "\"label\":\"paper\"") != nullptr);
  CHECK(strstr(out, "\"confidence\":0.877") != nullptr);
  CHECK(strstr(out, "\"needs_review\":false") != nullptr);
  CHECK(strstr(out, "\"model_version\":\"v1-esp32\"") != nullptr);

  char tiny[16];
  CHECK(toJson(e, "bin-01", "0.1.0", "v1", tiny, sizeof(tiny)) == -1);  // 잘림 감지
}

static void test_parse_event_id() {
  CHECK(parseEventId("{\"id\":123}") == 123);
  CHECK(parseEventId("{\"id\": 7, \"x\":1}") == 7);
  CHECK(parseEventId("{\"ids\":[1]}") == -1);
  CHECK(parseEventId("{\"id\":\"a\"}") == -1);
  CHECK(parseEventId("") == -1);
  CHECK(parseEventId(nullptr) == -1);
}

int main() {
  test_high_confidence_goes_to_its_bin();
  test_low_confidence_goes_to_general_and_flags_review();
  test_empty_does_not_actuate();
  test_bin_mapping();
  test_label_names_roundtrip();
  test_edge_impulse_alphabetical_mapping();
  test_buffer_keeps_order_and_drops_oldest();
  test_json_matches_server_schema();
  test_parse_event_id();
  if (failures) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("all firmware logic tests passed\n");
  return 0;
}
