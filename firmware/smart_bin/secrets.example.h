// 이 파일을 secrets.h 로 복사한 뒤 값을 채우세요. secrets.h 는 .gitignore 되어 커밋되지 않습니다.
//   cp secrets.example.h secrets.h
#pragma once

#define WIFI_SSID     "your-wifi"
#define WIFI_PASSWORD "your-password"

// 서버 주소: 노트북에서 docker compose 로 띄웠다면 노트북의 내부 IP (예: 192.168.0.12)
#define SERVER_URL    "http://192.168.0.12:8000"
#define DEVICE_ID     "bin-01"
