#pragma once
// =============================================================================
// ECG filter + R-peak detector -- a faithful port of the health-companion
// reference project's already-validated pipeline (median despike -> 0.5Hz
// highpass -> 40Hz lowpass -> 50Hz/Q30 notch -> Pan-Tompkins-style adaptive
// QRS detector, 2s warm-up). Same biquad coefficients, same detector
// thresholds -- not a re-derivation. Re-confirmed by ../tools/validate_filter.py
// (reproduces that project's documented 98.86% Se / 99.03% +P against
// MIT-BIH at FS=250Hz exactly) and by ../tools/test_ecg_pipeline_desktop.cpp
// (this exact header, g++-compiled, tested against a synthetic 72 BPM
// signal -- no Arduino dependency, so both the firmware .ino and the
// desktop test compile the identical logic).
// =============================================================================

#include <math.h>

namespace EcgPipeline {

constexpr float FS = 250.0f;
constexpr float DT_MS = 1000.0f / FS;

struct Biquad {
  float b0, b1, b2, a1, a2;
  float x1 = 0, x2 = 0, y1 = 0, y2 = 0;

  float process(float x) {
    float y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x;
    y2 = y1; y1 = y;
    return y;
  }
};

inline Biquad makeHighpass(float fs, float f0, float Q) {
  float w0 = 2.0f * (float)M_PI * f0 / fs;
  float alpha = sinf(w0) / (2.0f * Q);
  float c = cosf(w0);
  float a0 = 1.0f + alpha;
  return Biquad{ (1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0, -2 * c / a0, (1 - alpha) / a0 };
}
inline Biquad makeLowpass(float fs, float f0, float Q) {
  float w0 = 2.0f * (float)M_PI * f0 / fs;
  float alpha = sinf(w0) / (2.0f * Q);
  float c = cosf(w0);
  float a0 = 1.0f + alpha;
  return Biquad{ (1 - c) / 2 / a0, (1 - c) / a0, (1 - c) / 2 / a0, -2 * c / a0, (1 - alpha) / a0 };
}
inline Biquad makeNotch(float fs, float f0, float Q) {
  float w0 = 2.0f * (float)M_PI * f0 / fs;
  float alpha = sinf(w0) / (2.0f * Q);
  float c = cosf(w0);
  float a0 = 1.0f + alpha;
  return Biquad{ 1 / a0, -2 * c / a0, 1 / a0, -2 * c / a0, (1 - alpha) / a0 };
}

// 3-tap median despike.
struct MedianDespike3 {
  float buf[3] = { 0, 0, 0 };
  int count = 0;
  float process(float x) {
    buf[0] = buf[1]; buf[1] = buf[2]; buf[2] = x;
    if (count < 3) { count++; return x; }
    float a = buf[0], b = buf[1], c = buf[2];
    if (a > b) { float t = a; a = b; b = t; }
    if (b > c) { float t = b; b = c; c = t; }
    if (a > b) { float t = a; a = b; b = t; }
    return b;
  }
};

// Lightweight Pan-Tompkins-style QRS detector: derivative -> square ->
// moving-window integration -> adaptive threshold, 2s warm-up.
class QRSDetector {
 public:
  explicit QRSDetector(float fs) : dtMs_(1000.0f / fs) {
    mwiWindow_ = (int)roundf(0.15f * fs);
    mwiBuf_ = new float[mwiWindow_]();
    warmupSamples_ = (long)roundf(2.0f * fs);
  }
  ~QRSDetector() { delete[] mwiBuf_; }

  // n = global sample index of the sample just processed. Returns true and
  // sets *outPeakIdx when a QRS beat is confirmed (post warm-up).
  bool process(long n, float filtSample, long *outPeakIdx) {
    derivBuf_[0] = derivBuf_[1]; derivBuf_[1] = derivBuf_[2];
    derivBuf_[2] = derivBuf_[3]; derivBuf_[3] = derivBuf_[4];
    derivBuf_[4] = filtSample;
    if (derivCount_ < 5) derivCount_++;

    float deriv = 0;
    if (derivCount_ == 5) {
      deriv = (2 * derivBuf_[4] + derivBuf_[3] - derivBuf_[1] - 2 * derivBuf_[0]) / 8.0f;
    }
    float sq = deriv * deriv;

    mwiSum_ -= mwiBuf_[mwiIdx_];
    mwiBuf_[mwiIdx_] = sq;
    mwiSum_ += sq;
    mwiIdx_ = (mwiIdx_ + 1) % mwiWindow_;
    float integrated = mwiSum_ / mwiWindow_;

    bool result = false;
    if (iPrev1_ > iPrev2_ && iPrev1_ >= integrated && iPrev1_ > 0) {
      float V = iPrev1_;
      long idx = n - 1;
      float threshold1 = NPKI_ + 0.25f * (SPKI_ - NPKI_);
      float sinceLastMs = (lastPeakIdx_ < 0) ? 1e30f : (idx - lastPeakIdx_) * dtMs_;

      if (V > threshold1 && sinceLastMs > refractoryMs_) {
        SPKI_ = 0.125f * V + 0.875f * SPKI_;
        lastPeakIdx_ = idx;
        if (idx >= warmupSamples_) {
          *outPeakIdx = idx;
          result = true;
        }
      } else {
        NPKI_ = 0.125f * V + 0.875f * NPKI_;
      }
    }
    iPrev2_ = iPrev1_;
    iPrev1_ = integrated;
    return result;
  }

 private:
  float dtMs_;
  float derivBuf_[5] = { 0, 0, 0, 0, 0 };
  int derivCount_ = 0;
  int mwiWindow_;
  float *mwiBuf_;
  int mwiIdx_ = 0;
  float mwiSum_ = 0;
  float SPKI_ = 0, NPKI_ = 0;
  float iPrev1_ = 0, iPrev2_ = 0;
  long lastPeakIdx_ = -1;
  const float refractoryMs_ = 200.0f;
  long warmupSamples_;
};

constexpr float BPM_MIN = 30.0f;
constexpr float BPM_MAX = 220.0f;

// Wires despike -> highpass -> lowpass -> notch -> QRS detector -> RR-to-BPM
// together, one raw sample at a time. Used identically by the firmware
// (fed real analogRead() samples) and the desktop test (fed a synthetic
// signal) -- only one copy of this wiring to ever be wrong.
class EcgSampleProcessor {
 public:
  EcgSampleProcessor()
      : hp_(makeHighpass(FS, 0.5f, 0.707f)),
        lp_(makeLowpass(FS, 40.0f, 0.707f)),
        notch_(makeNotch(FS, 50.0f, 30.0f)),
        detector_(FS) {}

  // Always writes *outFiltered. Returns true and updates *outBpm only when
  // this sample confirmed a new QRS beat with a valid (sanity-bounded)
  // RR-derived BPM (i.e. not the very first beat).
  bool process(float rawSample, float *outFiltered, float *outBpm) {
    sampleIndex_++;
    float despiked = despike_.process(rawSample);
    float filt = notch_.process(lp_.process(hp_.process(despiked)));
    *outFiltered = filt;

    long peakIdx;
    if (!detector_.process(sampleIndex_, filt, &peakIdx)) return false;

    bool gotBpm = false;
    if (lastPeakIdx_ >= 0) {
      float rrMs = (peakIdx - lastPeakIdx_) * DT_MS;
      if (rrMs > 0) {
        float bpm = 60000.0f / rrMs;
        if (bpm >= BPM_MIN && bpm <= BPM_MAX) {
          *outBpm = bpm;
          gotBpm = true;
        }
      }
    }
    lastPeakIdx_ = peakIdx;
    return gotBpm;
  }

 private:
  Biquad hp_, lp_, notch_;
  MedianDespike3 despike_;
  QRSDetector detector_;
  long sampleIndex_ = -1;
  long lastPeakIdx_ = -1;
};

} // namespace EcgPipeline
