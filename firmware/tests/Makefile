CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Werror

# ---- 판단 로직 (하드웨어 무관) ----
test: test_decision
	./test_decision

test_decision: test_decision.cpp ../smart_bin/decision.h
	$(CXX) $(CXXFLAGS) -o $@ test_decision.cpp

# ---- 모델 추론: 펌웨어와 같은 C++ 코드 + TFLite Micro 를 PC 에서 실행 ----
#   make model TFLM=<esp-tflite-micro 경로> DUMP=<evaluate.py --dump 폴더>
TFLM ?= ../../build/esp-tflite-micro
DUMP ?= ../../build/dump
MIN888 ?= 0.75
MIN565 ?= 0.70
TFLM_INC := -I$(TFLM) -I$(TFLM)/third_party/flatbuffers/include -I$(TFLM)/third_party/kissfft
TFLM_FLAGS := -std=gnu++17 -O2 -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -fno-rtti -fno-exceptions -w
M := $(TFLM)/tensorflow/lite/micro
TFLM_SRCS = $(addprefix $(M)/,debug_log.cc flatbuffer_conversions.cc flatbuffer_utils.cc memory_helpers.cc \
  micro_allocation_info.cc micro_allocator.cc micro_common.cc micro_context.cc micro_interpreter_context.cc \
  micro_interpreter_graph.cc micro_interpreter.cc micro_log.cc micro_op_resolver.cc micro_profiler.cc \
  micro_resource_variable.cc micro_time.cc micro_utils.cc recording_micro_allocator.cc system_setup.cc \
  memory_planner/greedy_memory_planner.cc memory_planner/linear_memory_planner.cc \
  kernels/internal/common.cc kernels/internal/portable_tensor_utils.cc kernels/internal/quantization_util.cc \
  kernels/internal/runtime_shape.cc kernels/internal/tensor_ctypes.cc) \
  $(wildcard $(M)/kernels/*.cc) $(filter-out %_test.cc,$(wildcard $(M)/arena_allocator/*.cc))
TFLM_OBJS = $(patsubst $(TFLM)/%.cc,build/tflm/%.o,$(TFLM_SRCS))

build/tflm/%.o: $(TFLM)/%.cc
	@mkdir -p $(dir $@)
	@$(CXX) $(TFLM_FLAGS) $(TFLM_INC) -c $< -o $@

build/libtflm.a: $(TFLM_OBJS)
	ar rcs $@ $^

test_model: test_model.cpp ../smart_bin/model_runner.cpp ../smart_bin/model_runner.h ../smart_bin/preprocess.h ../smart_bin/model_data.h build/libtflm.a
	$(CXX) -std=gnu++17 -O2 -DTF_LITE_STATIC_MEMORY -Wall -Wextra $(TFLM_INC) -o $@ test_model.cpp ../smart_bin/model_runner.cpp build/libtflm.a

model: test_model
	./test_model $(DUMP) $(MIN888) $(MIN565)

clean:
	rm -rf test_decision test_model build

.PHONY: test model clean
