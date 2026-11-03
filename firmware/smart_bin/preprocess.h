// 카메라 이미지 → 모델 입력 변환. 하드웨어 무관(순수 C++)이라 PC에서 테스트한다.
//
// 학습(model/train.py)과 똑같이:
//   1) 가운데 정사각형 크롭 (320x240 → 240x240)
//   2) 양선형 보간으로 size x size 축소 (TensorFlow tf.image.resize 와 같은 half-pixel 방식)
//   3) int8 양자화: q = round(pixel / scale) + zero_point
#pragma once
#include <math.h>
#include <stdint.h>

namespace smartbin {

// esp32-camera 의 RGB565 는 빅엔디언 (Zero-Touch Media 와 동일)
struct Rgb565BE {
  const uint8_t* buf;
  int width;
  inline void get(int x, int y, float* rgb) const {
    const uint8_t* p = buf + 2 * (y * width + x);
    uint16_t v = (uint16_t)((p[0] << 8) | p[1]);
    uint8_t r = (v >> 11) & 0x1F, g = (v >> 5) & 0x3F, b = v & 0x1F;
    rgb[0] = (float)((r << 3) | (r >> 2));
    rgb[1] = (float)((g << 2) | (g >> 4));
    rgb[2] = (float)((b << 3) | (b >> 2));
  }
};

// PC 테스트용 RGB888
struct Rgb888 {
  const uint8_t* buf;
  int width;
  inline void get(int x, int y, float* rgb) const {
    const uint8_t* p = buf + 3 * (y * width + x);
    rgb[0] = p[0];
    rgb[1] = p[1];
    rgb[2] = p[2];
  }
};

// src(width x height) → out[size*size*3] (HWC, RGB, int8)
template <typename Src>
void toModelInput(const Src& src, int width, int height, int size, float scale, int zeroPoint,
                  int8_t* out) {
  const int s = width < height ? width : height;
  const int ox = (width - s) / 2, oy = (height - s) / 2;
  const float ratio = (float)s / (float)size;

  for (int y = 0; y < size; ++y) {
    float sy = ((float)y + 0.5f) * ratio - 0.5f;
    if (sy < 0) sy = 0;
    int y0 = (int)sy;
    if (y0 > s - 1) y0 = s - 1;
    int y1 = y0 + 1 < s ? y0 + 1 : s - 1;
    float fy = sy - (float)y0;

    for (int x = 0; x < size; ++x) {
      float sx = ((float)x + 0.5f) * ratio - 0.5f;
      if (sx < 0) sx = 0;
      int x0 = (int)sx;
      if (x0 > s - 1) x0 = s - 1;
      int x1 = x0 + 1 < s ? x0 + 1 : s - 1;
      float fx = sx - (float)x0;

      float a[3], b[3], c[3], d[3];
      src.get(ox + x0, oy + y0, a);
      src.get(ox + x1, oy + y0, b);
      src.get(ox + x0, oy + y1, c);
      src.get(ox + x1, oy + y1, d);

      int8_t* o = out + 3 * (y * size + x);
      for (int k = 0; k < 3; ++k) {
        float top = a[k] + (b[k] - a[k]) * fx;
        float bot = c[k] + (d[k] - c[k]) * fx;
        float v = top + (bot - top) * fy;
        int q = (int)lroundf(v / scale) + zeroPoint;
        o[k] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
      }
    }
  }
}

}  // namespace smartbin
