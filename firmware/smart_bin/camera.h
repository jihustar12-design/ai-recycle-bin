#pragma once
#include "esp_camera.h"

// XIAO ESP32S3 Sense 카메라(OV2640) 초기화. 성공하면 true.
bool cameraInit();

// 한 장 촬영. 사용 후 esp_camera_fb_return(fb) 로 반드시 반환할 것.
camera_fb_t* cameraCapture();
