#include "camera.h"

#include <Arduino.h>

// XIAO ESP32S3 Sense 카메라 핀
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

bool cameraInit() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_RGB565;  // JPEG 디코딩 비용 없이 바로 모델 입력으로 (Zero-Touch Media 와 동일)
  c.frame_size = FRAMESIZE_QVGA;      // 320x240 → 가운데 240x240 크롭 → 160x160
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;   // 투입 순간의 최신 프레임

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("[camera] init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

camera_fb_t* cameraCapture() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) Serial.println("[camera] capture failed");
  return fb;
}
