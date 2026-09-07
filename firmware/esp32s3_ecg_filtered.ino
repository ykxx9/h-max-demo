/*
  ESP32-S3 ECG + IMU with on-device filtering and beat detection

  Wiring: BioShield OUT -> GPIO4, both BioShield and MPU6050 at 3.3V,
  MPU6050 SDA -> GPIO8, SCL -> GPIO9, common GND.

  Sample rate: 250Hz (software-timed analogRead, no hardware rate lock
  like MAX30003 would have -- lets us match the already-validated
  250Hz filter/detector design exactly, no resampling needed).

  Filter + detector: ecg_pipeline.h, a faithful port of the health-companion
  reference project's validated ECG pipeline. See that file's header for
  provenance and validation. Re-confirmed by ../tools/validate_filter.py
  (reproduces the reference project's documented 98.86% Se / 99.03% +P
  against MIT-BIH exactly) and ../tools/test_ecg_pipeline_desktop.cpp
  (this exact header, desktop-tested against a synthetic 72 BPM signal).

  Serial output (115200 baud), one line per sample:
    raw,filtered,bpm,ax,ay,az
  bpm is 0 until the first confirmed beat after warm-up. ax/ay/az are in
  m/s^2 (+-2g range -> +-19.61 m/s^2).
*/

#include <Wire.h>
#include "ecg_pipeline.h"

#define ECG_PIN 4
#define SAMPLE_INTERVAL_US (uint32_t)(1000000.0f / EcgPipeline::FS)

#define MPU_ADDR 0x68
#define MPU_PWR_MGMT_1 0x6B
#define MPU_ACCEL_XOUT_H 0x3B
#define MPU_WHO_AM_I 0x75

EcgPipeline::EcgSampleProcessor ecg;
float currentBpm = 0;

// =============================================================================
// MPU6050 -- fixed byte-order bug (see mpuReadAccel below).
// =============================================================================
void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t mpuReadByte(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((int)MPU_ADDR, 1);
  return Wire.read();
}

bool mpuReadAccel(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(MPU_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 6) != 6) return false;

  // Read each byte into its own sequenced statement -- do NOT combine two
  // Wire.read() calls in one expression like `(Wire.read()<<8)|Wire.read()`.
  // C++ does not guarantee left-to-right evaluation order for the operands
  // of `<<`/`|`, so a compiler is free to call the two Wire.read()s in
  // either order; since each call also advances Wire's internal read
  // pointer, evaluating them in the "wrong" order silently swaps the
  // MSB/LSB bytes. That byte swap was the actual root cause of the
  // near-saturated (-19.6 m/s^2, i.e. raw=0x8000) Accel Y readings seen
  // while the board was stationary: a genuinely small Y reading (MSB
  // near 0x00) got reassembled with the LSB byte in the high position.
  uint8_t xh = Wire.read(); uint8_t xl = Wire.read();
  uint8_t yh = Wire.read(); uint8_t yl = Wire.read();
  uint8_t zh = Wire.read(); uint8_t zl = Wire.read();

  int16_t rawX = (int16_t)((xh << 8) | xl);
  int16_t rawY = (int16_t)((yh << 8) | yl);
  int16_t rawZ = (int16_t)((zh << 8) | zl);

  ax = rawX / 16384.0f * 9.80665f;
  ay = rawY / 16384.0f * 9.80665f;
  az = rawZ / 16384.0f * 9.80665f;
  return true;
}

unsigned long lastSampleTime = 0;
unsigned long lastAccelReadTime = 0;
const unsigned long ACCEL_INTERVAL_MS = 100;
float ax = 0, ay = 0, az = 0;
bool mpuOk = false;

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12);

  Wire.begin(8, 9);
  uint8_t whoAmI = mpuReadByte(MPU_WHO_AM_I);
  mpuOk = (whoAmI == 0x68);
  if (mpuOk) {
    mpuWrite(MPU_PWR_MGMT_1, 0x01); // wake from sleep, clock = PLL w/ X-axis gyro reference
    delay(100);
  } else {
    Serial.print("# MPU6050 not responding (WHO_AM_I=0x");
    Serial.print(whoAmI, HEX);
    Serial.println(", expected 0x68) -- check wiring/power/I2C address, ax/ay/az will read 0");
  }

  Serial.println("raw,filtered,bpm,ax,ay,az");
}

void loop() {
  unsigned long nowUs = micros();
  if (nowUs - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = nowUs;
    unsigned long nowMs = millis();

    int raw = analogRead(ECG_PIN);
    float filtered, bpm;
    if (ecg.process((float)raw, &filtered, &bpm)) currentBpm = bpm;

    if (mpuOk && (nowMs - lastAccelReadTime >= ACCEL_INTERVAL_MS)) {
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
