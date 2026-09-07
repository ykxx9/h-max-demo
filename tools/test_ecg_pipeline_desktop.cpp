// Desktop test (g++, no Arduino toolchain needed) for the hardware-
// independent logic in ../firmware/ecg_pipeline.h -- the exact header the
// .ino compiles, not a re-typed copy. Feeds a synthetic 72 BPM signal and
// checks the detector recovers ~72 BPM, same methodology as the
// health-companion reference project's own test/test_ecg_pipeline.cpp.
//
// Run: g++ -O2 -std=c++14 -I../firmware test_ecg_pipeline_desktop.cpp -o test_ecg_pipeline_desktop && ./test_ecg_pipeline_desktop

#include <cstdio>
#include <cmath>
#include <vector>
#include "ecg_pipeline.h"

using namespace EcgPipeline;

int main() {
  const float targetBpm = 72.0f;
  const float rrSec = 60.0f / targetBpm;
  const float durationSec = 30.0f;
  const int n = (int)(durationSec * FS);

  std::vector<float> beatCenters;
  for (float t = 0.3f; t < durationSec; t += rrSec) beatCenters.push_back(t);

  unsigned int seed = 42;
  auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return (float)(seed & 0x7fffffff) / (float)0x7fffffff; };

  auto qrsPulse = [](float t, float center) {
    float dt = t - center;
    float sigma = 0.02f;
    return 300.0f * expf(-(dt * dt) / (2 * sigma * sigma));
  };

  EcgSampleProcessor proc;
  std::vector<float> bpms;
  int detectedBeats = 0;

  for (int i = 0; i < n; i++) {
    float t = i / FS;
    float v = 512.0f;
    v += 15.0f * sinf(2 * (float)M_PI * 0.2f * t);       // baseline wander
    v += 8.0f * sinf(2 * (float)M_PI * 50.0f * t);        // mains hum
    v += (rnd() - 0.5f) * 10.0f;                          // noise
    for (float c : beatCenters) if (fabsf(t - c) < 0.1f) v += qrsPulse(t, c);

    float filtered, bpm;
    if (proc.process(v, &filtered, &bpm)) {
      detectedBeats++;
      bpms.push_back(bpm);
    }
  }

  float avgBpm = 0;
  for (float b : bpms) avgBpm += b;
  avgBpm /= (bpms.empty() ? 1 : bpms.size());

  printf("True beats placed: %zu, RR-derived BPM samples: %d\n", beatCenters.size(), detectedBeats);
  printf("Observed avg BPM: %.2f (target %.1f)\n", avgBpm, targetBpm);

  bool pass = fabsf(avgBpm - targetBpm) < 3.0f && detectedBeats >= (int)beatCenters.size() - 5;
  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
