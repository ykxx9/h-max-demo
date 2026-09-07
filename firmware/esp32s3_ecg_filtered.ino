/*
  ESP32-S3 ECG + IMU with on-device filtering and beat detection
  Wiring: BioShield OUT -> GPIO4, both BioShield and MPU6050 at 3.3V,
  MPU6050 SDA -> GPIO8, SCL -> GPIO9, common GND.

  Sample rate: 250Hz (software-timed analogRead, no hardware rate lock
  like MAX30003 would have -- lets us match the already-validated
  250Hz filter/detector design exactly, no resampling needed).

  Filter chain: 0.5Hz highpass -> 40Hz lowpass -> 50Hz notch (mains hum),
  each a standard RBJ-cookbook biquad. Beat detection: derivative ->
  square -> moving-window integration -> adaptive threshold, standard
  Pan-Tompkins method, 2-second warm-up before reporting confirmed beats.

  NOTE: this is a fresh implementation of the same documented method
  already validated elsewhere in this project's browser tool -- it has
  NOT itself been through that same validation yet. Treat what you see
  now as the first real test of this specific code.

  Serial output (115200 baud), one line per sample:
    raw,filtered,bpm,ax,ay,az
  bpm is 0 until the first confirmed beat after warm-up.
*/

#include <Wire.h>
#include <math.h>

#define ECG_PIN 4
#define FS 250.0f
#define SAMPLE_INTERVAL_US (uint32_t)(1000000.0f / FS)

#define MPU_ADDR 0x68
#define MPU_PWR_MGMT_1 0x6B
#define MPU_ACCEL_XOUT_H 0x3B

// ---------- generic biquad (RBJ cookbook) ----------
struct Biquad {
  float b0, b1, b2, a1, a2;
  float z1 = 0, z2 = 0;
  float process(float in) {
    float out = b0 * in + z1;
    z1 = b1 * in + z2 - a1 * out;
    z2 = b2 * in - a2 * out;
    return out;
  }
};

void makeHighpass(Biquad &f, float f0, float fs, float Q) {
  float w0 = 2.0f * PI * f0 / fs;
  float alpha = sin(w0) / (2.0f * Q);
  float cosw0 = cos(w0);
  float a0 = 1 + alpha;
  f.b0 = ((1 + cosw0) / 2) / a0;
  f.b1 = (-(1 + cosw0)) / a0;
  f.b2 = ((1 + cosw0) / 2) / a0;
  f.a1 = (-2 * cosw0) / a0;
  f.a2 = (1 - alpha) / a0;
}

void makeLowpass(Biquad &f, float f0, float fs, float Q) {
  float w0 = 2.0f * PI * f0 / fs;
  float alpha = sin(w0) / (2.0f * Q);
  float cosw0 = cos(w0);
  float a0 = 1 + alpha;
  f.b0 = ((1 - cosw0) / 2) / a0;
  f.b1 = (1 - cosw0) / a0;
  f.b2 = ((1 - cosw0) / 2) / a0;
  f.a1 = (-2 * cosw0) / a0;
  f.a2 = (1 - alpha) / a0;
}

void makeNotch(Biquad &f, float f0, float fs, float Q) {
  float w0 = 2.0f * PI * f0 / fs;
  float alpha = sin(w0) / (2.0f * Q);
  float cosw0 = cos(w0);
  float a0 = 1 + alpha;
  f.b0 = 1.0f / a0;
  f.b1 = (-2 * cosw0) / a0;
  f.b2 = 1.0f / a0;
  f.a1 = (-2 * cosw0) / a0;
  f.a2 = (1 - alpha) / a0;
}

Biquad hp, lp, notch;

// ---------- Pan-Tompkins-style beat detection ----------
#define DERIV_LEN 5
float derivBuf[DERIV_LEN] = {0};

#define INTEG_WINDOW 38  // ~150ms at 250Hz
float integBuf[INTEG_WINDOW] = {0};
int integIdx = 0;
float integSum = 0;

float SPKI = 0, NPKI = 0, THRESHOLD = 0;
bool thresholdsInit = false;

unsigned long lastPeakMs = 0;
const unsigned long REFRACTORY_MS = 250; // ~240bpm cap
const unsigned long WARMUP_MS = 2000;
unsigned long startMs = 0;

float bpmHistory[4] = {0, 0, 0, 0};
int bpmHistIdx = 0;
float currentBpm = 0;

float derivative(float x) {
  for (int i = 0; i < DERIV_LEN - 1; i++) derivBuf[i] = derivBuf[i + 1];
  derivBuf[DERIV_LEN - 1] = x;
  // simple 5-point derivative approximation
  return (2 * derivBuf[4] + derivBuf[3] - derivBuf[1] - 2 * derivBuf[0]) / 8.0f;
}

float movingWindowIntegrate(float x) {
  integSum -= integBuf[integIdx];
  integBuf[integIdx] = x;
  integSum += x;
  integIdx = (integIdx + 1) % INTEG_WINDOW;
  return integSum / INTEG_WINDOW;
}

void updateBpm(unsigned long nowMs) {
  if (lastPeakMs > 0) {
    unsigned long rr = nowMs - lastPeakMs;
    if (rr > 272 && rr < 2000) { // sane RR bounds (30-220bpm)
      float bpm = 60000.0f / rr;
      bpmHistory[bpmHistIdx] = bpm;
      bpmHistIdx = (bpmHistIdx + 1) % 4;
      float sum = 0; int n = 0;
      for (int i = 0; i < 4; i++) if (bpmHistory[i] > 0) { sum += bpmHistory[i]; n++; }
      if (n > 0) currentBpm = sum / n;
    }
  }
  lastPeakMs = nowMs;
}

void processDetector(float filteredSample, unsigned long nowMs) {
  float d = derivative(filteredSample);
  float sq = d * d;
  float integrated = movingWindowIntegrate(sq);

  if (!thresholdsInit) {
    // seed thresholds from the first bit of signal
    SPKI = integrated;
    NPKI = integrated * 0.5f;
    THRESHOLD = NPKI + 0.25f * (SPKI - NPKI);
    thresholdsInit = true;
    return;
  }

  THRESHOLD = NPKI + 0.25f * (SPKI - NPKI);

  bool pastRefractory = (nowMs - lastPeakMs) > REFRACTORY_MS;
  bool pastWarmup = (nowMs - startMs) > WARMUP_MS;

  if (integrated > THRESHOLD && pastRefractory) {
    SPKI = 0.125f * integrated + 0.875f * SPKI;
    if (pastWarmup) updateBpm(nowMs);
    else lastPeakMs = nowMs; // keep refractory logic sane during warmup, don't report bpm yet
  } else {
    NPKI = 0.125f * integrated + 0.875f * NPKI;
  }
}

// ---------- MPU6050 ----------
void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool mpuReadAccel(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(MPU_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 6) != 6) return false;
  int16_t rawX = (Wire.read() << 8) | Wire.read();
  int16_t rawY = (Wire.read() << 8) | Wire.read();
  int16_t rawZ = (Wire.read() << 8) | Wire.read();
  ax = rawX / 16384.0f * 9.80665f;
  ay = rawY / 16384.0f * 9.80665f;
  az = rawZ / 16384.0f * 9.80665f;
  return true;
}

unsigned long lastSampleTime = 0;
unsigned long lastAccelReadTime = 0;
const unsigned long ACCEL_INTERVAL_MS = 100;
float ax = 0, ay = 0, az = 0;

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12);

  makeHighpass(hp, 0.5f, FS, 0.707f);
  makeLowpass(lp, 40.0f, FS, 0.707f);
  makeNotch(notch, 50.0f, FS, 10.0f);

  Wire.begin(8, 9);
  mpuWrite(MPU_PWR_MGMT_1, 0x00);
  delay(100);

  startMs = millis();
  Serial.println("raw,filtered,bpm,ax,ay,az");
}

void loop() {
  unsigned long nowUs = micros();
  if (nowUs - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = nowUs;
    unsigned long nowMs = millis();

    int raw = analogRead(ECG_PIN);
    float filtered = notch.process(lp.process(hp.process((float)raw)));

    processDetector(filtered, nowMs);

    if (nowMs - lastAccelReadTime >= ACCEL_INTERVAL_MS) {
      lastAccelReadTime = nowMs;
      mpuReadAccel(ax, ay, az);
    }

    Serial.print(raw);
    Serial.print(",");
    Serial.print(filtered, 2);
    Serial.print(",");
    Serial.print(currentBpm, 1);
    Serial.print(",");
    Serial.print(ax, 2);
    Serial.print(",");
    Serial.print(ay, 2);
    Serial.print(",");
    Serial.println(az, 2);
  }
}