#include "config.h"
#if USE_TFLITE_MODEL

// esp-tflite-micro 라이브러리는 이 옵션으로 빌드되어 있다. 헤더도 같은 구조체 정의를 보도록 맞춘다.
#ifndef TF_LITE_STATIC_MEMORY
#define TF_LITE_STATIC_MEMORY
#endif

#include "model_runner.h"

#include <new>

#include "decision.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"  // schema(GetModel)는 micro_interpreter.h 가 함께 가져옴

namespace smartbin {

namespace {
// 모델에 쓰인 연산만 등록 (MobileNetV2 + 분류기)
using Resolver = tflite::MicroMutableOpResolver<8>;
alignas(16) uint8_t resolverBuf[sizeof(Resolver)];
alignas(16) uint8_t interpreterBuf[sizeof(tflite::MicroInterpreter)];

tflite::MicroInterpreter* interp(void* p) { return static_cast<tflite::MicroInterpreter*>(p); }
}  // namespace

bool ModelRunner::begin(uint8_t* arena, size_t arenaSize) {
  const tflite::Model* model = tflite::GetModel(g_model);
#ifdef TFLITE_SCHEMA_VERSION
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_ = "model schema version mismatch";
    return false;
  }
#endif

  Resolver* resolver = new (resolverBuf) Resolver();
  resolver->AddAdd();
  resolver->AddConv2D();
  resolver->AddDepthwiseConv2D();
  resolver->AddFullyConnected();
  resolver->AddMean();
  resolver->AddMul();
  resolver->AddSoftmax();
  resolver->AddQuantize();

  auto* it = new (interpreterBuf) tflite::MicroInterpreter(model, *resolver, arena, arenaSize);
  if (it->AllocateTensors() != kTfLiteOk) {
    error_ = "AllocateTensors failed (arena too small?)";
    return false;
  }
  TfLiteTensor* in = it->input(0);
  if (in->type != kTfLiteInt8 || in->dims->size != 4 || in->dims->data[1] != MODEL_INPUT_SIZE) {
    error_ = "unexpected input tensor";
    return false;
  }
  interpreter_ = it;
  error_ = nullptr;
  return true;
}

int8_t* ModelRunner::input() { return interp(interpreter_)->input(0)->data.int8; }
int ModelRunner::inputSize() const { return MODEL_INPUT_SIZE; }
float ModelRunner::inputScale() const { return interp(interpreter_)->input(0)->params.scale; }
int ModelRunner::inputZeroPoint() const { return interp(interpreter_)->input(0)->params.zero_point; }
size_t ModelRunner::arenaUsedBytes() const { return interp(interpreter_)->arena_used_bytes(); }

bool ModelRunner::invoke(float* scores) {
  auto* it = interp(interpreter_);
  if (it->Invoke() != kTfLiteOk) {
    error_ = "Invoke failed";
    return false;
  }
  TfLiteTensor* out = it->output(0);
  const float s = out->params.scale;
  const int z = out->params.zero_point;
  for (int i = 0; i < LABEL_COUNT; ++i) scores[i] = 0.0f;
  for (int i = 0; i < MODEL_LABEL_COUNT; ++i) {
    int idx = labelFromName(MODEL_LABELS[i]);  // 모델(알파벳순) → Label enum
    if (idx >= 0) scores[idx] = (out->data.int8[i] - z) * s;
  }
  return true;
}

}  // namespace smartbin

#endif  // USE_TFLITE_MODEL
