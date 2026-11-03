// AI 스마트 분리수거함 펌웨어
// 흐름: 초음파로 투입 감지 → 카메라 촬영 → 온디바이스 분류 → 서보로 칸 이동 → 로그 버퍼 → 서버 전송
#include <ESP32Servo.h>
#include <HTTPClient.h>
#include <WiFi.h>

#include "camera.h"
#include "classifier.h"
#include "config.h"
#include "decision.h"
#include "secrets.h"

using namespace smartbin;

static Servo servo;
static LogBuffer<LOG_BUFFER_SIZE> logs;
static bool cameraOk = false;
static uint32_t lastDetectMs = 0;
static uint32_t lastHeartbeatMs = 0;

// ---------- 센서 ----------
static float readDistanceCm() {
  digitalWrite(PIN_US_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_US_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_US_TRIG, LOW);
  unsigned long us = pulseIn(PIN_US_ECHO, HIGH, 30000);  // 최대 약 5m
  return us == 0 ? 999.0f : us / 58.0f;
}

// ---------- 서보 ----------
static void moveToBin(uint8_t bin) {
  servo.write(BIN_ANGLES[bin]);
  delay(SERVO_HOLD_MS);
  servo.write(SERVO_HOME_ANGLE);
}

// ---------- 네트워크 ----------
static bool postJson(const char* path, const char* body) {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.begin(String(SERVER_URL) + path);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST((uint8_t*)body, strlen(body));
  http.end();
  return code >= 200 && code < 300;
}

// 버퍼에 쌓인 로그를 앞에서부터 전송. 실패하면 남겨두고 다음 루프에서 재시도.
static void flushLogs() {
  int maxPerLoop = 5;  // 한 루프에서 최대 전송 건수
  char json[320];
  while (!logs.empty() && maxPerLoop-- > 0) {
    if (toJson(logs.front(), DEVICE_ID, FW_VERSION, MODEL_VERSION, json, sizeof(json)) < 0) {
      logs.pop();  // 비정상 로그는 버림
      continue;
    }
    if (!postJson("/api/events", json)) break;
    logs.pop();
  }
}

static void sendHeartbeat() {
  char json[200];
  snprintf(json, sizeof(json),
           "{\"device_id\":\"%s\",\"rssi\":%d,\"free_heap\":%u,\"uptime_s\":%lu,\"pending_logs\":%u}",
           DEVICE_ID, (int)WiFi.RSSI(), (unsigned)ESP.getFreeHeap(), millis() / 1000UL, (unsigned)logs.size());
  postJson("/api/heartbeat", json);
}

// ---------- 한 번의 투입 처리 ----------
static void handleItem() {
  digitalWrite(PIN_LED, LOW);
  uint32_t t0 = millis();

  camera_fb_t* fb = cameraOk ? cameraCapture() : nullptr;
  float scores[LABEL_COUNT] = {0};
  bool ok = classify(fb, scores);
  if (fb) esp_camera_fb_return(fb);
  uint32_t latency = millis() - t0;

  if (!ok) {
    Serial.println("[sort] classify failed");
    digitalWrite(PIN_LED, HIGH);
    return;
  }

  Decision d = decide(scores, LABEL_COUNT, CONF_THRESHOLD);
  Serial.printf("[sort] %s %.2f → bin %u%s (%u ms)\n", labelName(d.label), d.confidence, d.bin,
                d.needsReview ? " [review]" : "", (unsigned)latency);

  if (d.actuate) {
    moveToBin(d.bin);
    SortEvent e{d.label, d.confidence, latency, d.bin, d.needsReview};
    if (!logs.push(e)) Serial.println("[log] buffer full, oldest dropped");
  }
  digitalWrite(PIN_LED, HIGH);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_US_TRIG, OUTPUT);
  pinMode(PIN_US_ECHO, INPUT);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, HIGH);

  servo.attach(PIN_SERVO);
  servo.write(SERVO_HOME_ANGLE);

  cameraOk = cameraInit();
  Serial.printf("[boot] fw %s, model %s, camera %s\n", FW_VERSION, MODEL_VERSION, cameraOk ? "ok" : "FAIL");

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);  // 연결을 기다리지 않음: 오프라인이어도 분류는 동작
}

void loop() {
  uint32_t now = millis();

  if (now - lastDetectMs > DETECT_COOLDOWN_MS && readDistanceCm() < DETECT_CM) {
    lastDetectMs = now;
    delay(300);  // 물체가 자리 잡을 때까지 잠깐 대기
    handleItem();
  }

  flushLogs();

  if (now - lastHeartbeatMs > HEARTBEAT_INTERVAL_MS) {
    lastHeartbeatMs = now;
    sendHeartbeat();
  }
  delay(50);
}
