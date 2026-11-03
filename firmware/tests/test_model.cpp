// 펌웨어와 '같은 C++ 코드'(preprocess.h + model_runner.cpp + TFLite Micro)로 PC 에서 test 세트를 분류한다.
//   1) Python(evaluate.py, LiteRT) 예측과 95% 이상 일치하는지 → 펌웨어 전처리/추론 코드가 맞는지
//      (TFLite Micro 참조 커널과 LiteRT 의 int8 반올림 차이로 100% 일치는 아님)
//   2) RGB565(카메라 형식)로 바꿔 넣었을 때 정확도
// 데이터: python model/evaluate.py --dump DIR  /  빌드·실행: make -C firmware/tests model TFLM=... DUMP=...
// 인자: DUMP_DIR [최소 정확도 RGB888] [최소 정확도 RGB565]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "../smart_bin/decision.h"
#include "../smart_bin/model_runner.h"
#include "../smart_bin/preprocess.h"

using namespace smartbin;

static const char* const MODEL_ORDER[] = {"can", "empty", "general", "paper", "pet"};  // labels.txt 순서

static int argmaxLabel(const float* scores) {
  int best = 0;
  for (int i = 1; i < LABEL_COUNT; ++i)
    if (scores[i] > scores[best]) best = i;
  return best;  // Label enum
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s DUMP_DIR [min_acc_rgb888] [min_acc_rgb565]\n", argv[0]);
    return 2;
  }
  const char* dir = argv[1];
  const double minAcc888 = argc > 2 ? atof(argv[2]) : 0.0;
  const double minAcc565 = argc > 3 ? atof(argv[3]) : 0.0;

  static uint8_t arena[2 * 1024 * 1024];
  ModelRunner runner;
  if (!runner.begin(arena, sizeof(arena))) {
    fprintf(stderr, "model init failed: %s\n", runner.error());
    return 1;
  }
  printf("arena used: %zu KB\n", runner.arenaUsedBytes() / 1024);

  char path[512];
  snprintf(path, sizeof(path), "%s/index.txt", dir);
  FILE* idx = fopen(path, "r");
  if (!idx) {
    fprintf(stderr, "cannot open %s\n", path);
    return 2;
  }

  int n = 0, agreePy = 0, correct888 = 0, correct565 = 0, agree565 = 0;
  char name[64];
  int w, h, truthModel, predPyModel;
  std::vector<uint8_t> rgb, rgb565;
  while (fscanf(idx, "%63s %d %d %d %d", name, &w, &h, &truthModel, &predPyModel) == 5) {
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) continue;
    rgb.resize((size_t)w * h * 3);
    size_t got = fread(rgb.data(), 1, rgb.size(), f);
    fclose(f);
    if (got != rgb.size()) continue;

    // index.txt 의 번호는 모델(알파벳순) 기준 → Label enum 으로 변환
    int truth = labelFromName(MODEL_ORDER[truthModel]);
    int predPy = labelFromName(MODEL_ORDER[predPyModel]);
    float scores[LABEL_COUNT];

    // RGB888 (Python 과 같은 입력)
    toModelInput(Rgb888{rgb.data(), w}, w, h, runner.inputSize(), runner.inputScale(),
                 runner.inputZeroPoint(), runner.input());
    runner.invoke(scores);
    int p888 = argmaxLabel(scores);

    // RGB565 빅엔디언 (실제 카메라 형식)
    rgb565.resize((size_t)w * h * 2);
    for (int i = 0; i < w * h; ++i) {
      uint16_t v = (uint16_t)(((rgb[3 * i] >> 3) << 11) | ((rgb[3 * i + 1] >> 2) << 5) | (rgb[3 * i + 2] >> 3));
      rgb565[2 * i] = v >> 8;
      rgb565[2 * i + 1] = v & 0xFF;
    }
    toModelInput(Rgb565BE{rgb565.data(), w}, w, h, runner.inputSize(), runner.inputScale(),
                 runner.inputZeroPoint(), runner.input());
    runner.invoke(scores);
    int p565 = argmaxLabel(scores);

    ++n;
    agreePy += (p888 == predPy);
    correct888 += (p888 == truth);
    correct565 += (p565 == truth);
    agree565 += (p565 == p888);
  }
  fclose(idx);
  if (!n) {
    fprintf(stderr, "no samples\n");
    return 2;
  }

  double acc888 = (double)correct888 / n, acc565 = (double)correct565 / n;
  printf("samples                         : %d\n", n);
  printf("C++ vs Python prediction match  : %d/%d (%.1f%%)\n", agreePy, n, 100.0 * agreePy / n);
  printf("accuracy  RGB888 (C++/TFLM)     : %.1f%%\n", 100 * acc888);
  printf("accuracy  RGB565 (camera format): %.1f%%  (RGB888 과 예측 일치 %.1f%%)\n", 100 * acc565,
         100.0 * agree565 / n);

  bool ok = agreePy >= n * 0.95 && acc888 >= minAcc888 && acc565 >= minAcc565;
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
