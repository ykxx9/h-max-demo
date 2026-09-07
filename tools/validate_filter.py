"""
Python "ground truth" for the ESP32 firmware's ECG filter + beat detector.

This is the SAME filter + detector design already validated in the
health-companion reference project (98.86% Se / 99.03% +P against MIT-BIH,
see that project's classifier/results/detector_validation_report.md) --
the Biquad/highpass/lowpass/notch coefficients, MedianDespike3, QRSDetector
adaptive-threshold logic, and backtrack search below are a direct copy of
classifier/src/validate_detector_on_mitbih.py in that project, not a
re-derivation. The AAMI non-beat/paced-record constants are copied from
that project's classifier/src/aami_mapping.py (same citation: de Chazal
et al. 2004) for the same reason -- reuse, not re-derive.

Two checks:
  1. Re-run the MIT-BIH validation to confirm this copy reproduces the
     already-documented 98.86%/99.03% figures (`--mitbih`).
  2. Run against this project's own real recorded ECG sessions (the
     health-companion project's database/ecg_recordings/, captured with
     the same algorithm) as a self-consistency check, and report signal
     quality (`--sessions`, the default).

Usage:
    python3 tools/validate_filter.py              # sessions check only (fast)
    python3 tools/validate_filter.py --mitbih      # + full MIT-BIH reproduction (slow, ~2 min)
"""

import argparse
import csv
import glob
import os

import numpy as np

HEALTH_COMPANION_DIR = "/Users/ykxx/codes/health companion"
ECG_RECORDINGS_DIR = os.path.join(HEALTH_COMPANION_DIR, "database", "ecg_recordings")
MITBIH_DIR = "/Users/ykxx/ecg ai/db/mit-bih/mit-bih-arrhythmia-database-1.0.0"

FS = 250.0
SOURCE_FS = 360  # MIT-BIH native rate
RESAMPLE_UP, RESAMPLE_DOWN = 25, 36  # 360 * 25/36 = 250
TOLERANCE_MS = 150  # WFDB bxb convention

# Copied from health-companion classifier/src/aami_mapping.py (de Chazal et
# al. 2004 AAMI EC57 mapping) -- non-beat annotation symbols must never be
# treated as beats, and these 4 records are conventionally excluded because
# paced-beat morphology isn't representative of intrinsic rhythm.
NON_BEAT_SYMBOLS = {
    "+", "~", '"', "x", "|", "[", "]", "!", "(", ")", "'", "^", "s", "T",
    "*", "D", "=", "p", "u", "`",
}
PACED_RECORDS_EXCLUDED = {"102", "104", "107", "217"}


# =============================================================================
# Direct copy of health-companion/classifier/src/validate_detector_on_mitbih.py's
# filter chain + QRS detector (same biquad coefficients, same Pan-Tompkins-
# style adaptive detector, same 2s warm-up, same backtrack window).
# =============================================================================
class Biquad:
    def __init__(self, b0, b1, b2, a1, a2):
        self.b0, self.b1, self.b2, self.a1, self.a2 = b0, b1, b2, a1, a2
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, x):
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y


def highpass(fs, f0, q):
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    a0 = 1 + alpha
    return Biquad((1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0, -2 * c / a0, (1 - alpha) / a0)


def lowpass(fs, f0, q):
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    a0 = 1 + alpha
    return Biquad((1 - c) / 2 / a0, (1 - c) / a0, (1 - c) / 2 / a0, -2 * c / a0, (1 - alpha) / a0)


def notch(fs, f0, q):
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    a0 = 1 + alpha
    return Biquad(1 / a0, -2 * c / a0, 1 / a0, -2 * c / a0, (1 - alpha) / a0)


class MedianDespike3:
    def __init__(self):
        self.buf = []

    def process(self, x):
        self.buf.append(x)
        if len(self.buf) > 3:
            self.buf.pop(0)
        if len(self.buf) < 3:
            return x
        return sorted(self.buf)[1]


class QRSDetector:
    def __init__(self, fs):
        self.fs = fs
        self.dt_ms = 1000 / fs
        self.deriv_buf = []
        self.mwi_window = round(0.15 * fs)
        self.mwi_buf = np.zeros(self.mwi_window)
        self.mwi_idx = 0
        self.mwi_sum = 0.0
        self.spki = 0.0
        self.npki = 0.0
        self.i_prev1 = 0.0
        self.i_prev2 = 0.0
        self.last_peak_idx = -1
        self.refractory_ms = 200
        self.warmup_samples = round(2 * fs)

    def process(self, n, filt_sample):
        self.deriv_buf.append(filt_sample)
        if len(self.deriv_buf) > 5:
            self.deriv_buf.pop(0)

        deriv = 0.0
        if len(self.deriv_buf) == 5:
            x4, x3, _, x1, x0 = self.deriv_buf
            deriv = (2 * x0 + x1 - x3 - 2 * x4) / 8

        sq = deriv * deriv
        self.mwi_sum -= self.mwi_buf[self.mwi_idx]
        self.mwi_buf[self.mwi_idx] = sq
        self.mwi_sum += sq
        self.mwi_idx = (self.mwi_idx + 1) % self.mwi_window
        integrated = self.mwi_sum / self.mwi_window

        result = None
        if self.i_prev1 > self.i_prev2 and self.i_prev1 >= integrated and self.i_prev1 > 0:
            v = self.i_prev1
            idx = n - 1
            threshold1 = self.npki + 0.25 * (self.spki - self.npki)
            since_last_ms = float("inf") if self.last_peak_idx < 0 else (idx - self.last_peak_idx) * self.dt_ms

            if v > threshold1 and since_last_ms > self.refractory_ms:
                self.spki = 0.125 * v + 0.875 * self.spki
                self.last_peak_idx = idx
                if idx >= self.warmup_samples:
                    result = idx
            else:
                self.npki = 0.125 * v + 0.875 * self.npki

        self.i_prev2 = self.i_prev1
        self.i_prev1 = integrated
        return result


BACKTRACK_DELAY = round(round(0.15 * FS) / 2) + 2
BACKTRACK_SEARCH = 20


def run_detector(signal, fs=FS):
    """Run the full filter chain + QRS detector over one ECG signal (already
    at `fs` Hz). Returns (filtered_signal, detected_peak_indices)."""
    hp = highpass(fs, 0.5, 0.707)
    lp = lowpass(fs, 40, 0.707)
    nt = notch(fs, 50, 30)
    despike = MedianDespike3()
    detector = QRSDetector(fs)

    filt_out = np.empty(len(signal))
    filt_history = []
    detected_peaks = []

    for n, raw in enumerate(signal):
        filt = nt.process(lp.process(hp.process(despike.process(raw))))
        filt_out[n] = filt
        filt_history.append(filt)
        if len(filt_history) > 200:
            filt_history.pop(0)
        hist_start_idx = n - len(filt_history) + 1

        integrated_peak_idx = detector.process(n, filt)
        if integrated_peak_idx is not None:
            estimated_r = integrated_peak_idx - BACKTRACK_DELAY
            best_idx, best_val = -1, -np.inf
            for g in range(estimated_r - BACKTRACK_SEARCH, estimated_r + BACKTRACK_SEARCH + 1):
                pos = g - hist_start_idx
                if 0 <= pos < len(filt_history):
                    if filt_history[pos] > best_val:
                        best_val = filt_history[pos]
                        best_idx = g
            if best_idx >= 0:
                detected_peaks.append(best_idx)

    return filt_out, np.array(detected_peaks)


# =============================================================================
# Check 1: MIT-BIH reproduction (confirms this copy matches the
# already-documented 98.86%/99.03% Se/+P)
# =============================================================================
def evaluate_mitbih_record(wfdb, resample_poly, record_name):
    path = os.path.join(MITBIH_DIR, record_name)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    ch = 0
    for i, name in enumerate(rec.sig_name):
        if name.strip() == "MLII":
            ch = i
            break
    raw_signal = rec.p_signal[:, ch]
    resampled = resample_poly(raw_signal, RESAMPLE_UP, RESAMPLE_DOWN)

    beat_mask = np.array([s not in NON_BEAT_SYMBOLS for s in ann.symbol])
    true_samples_360 = ann.sample[beat_mask]
    true_samples_250 = np.round(true_samples_360 * FS / SOURCE_FS).astype(int)
    true_samples_250 = true_samples_250[true_samples_250 >= round(2 * FS)]

    _, detected = run_detector(resampled)

    tolerance_samples = TOLERANCE_MS * FS / 1000
    matched_true = np.zeros(len(true_samples_250), dtype=bool)
    matched_det = np.zeros(len(detected), dtype=bool)

    for i, d in enumerate(detected):
        diffs = np.abs(true_samples_250 - d)
        candidates = np.where((diffs <= tolerance_samples) & (~matched_true))[0]
        if len(candidates) > 0:
            j = candidates[np.argmin(diffs[candidates])]
            matched_true[j] = True
            matched_det[i] = True

    tp = matched_det.sum()
    fp = (~matched_det).sum()
    fn = (~matched_true).sum()
    return tp, fp, fn


def run_mitbih_check():
    import wfdb
    from scipy.signal import resample_poly

    print("=" * 70)
    print("CHECK 1: MIT-BIH reproduction")
    print("=" * 70)

    if not os.path.isdir(MITBIH_DIR):
        print(f"MIT-BIH database not found at {MITBIH_DIR} -- skipping.")
        return

    hea_files = sorted(glob.glob(os.path.join(MITBIH_DIR, "*.hea")))
    records = [os.path.splitext(os.path.basename(f))[0] for f in hea_files
               if os.path.splitext(os.path.basename(f))[0] not in PACED_RECORDS_EXCLUDED]

    total_tp = total_fp = total_fn = 0
    for rec_name in records:
        tp, fp, fn = evaluate_mitbih_record(wfdb, resample_poly, rec_name)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    se = total_tp / (total_tp + total_fn)
    pp = total_tp / (total_tp + total_fp)
    print(f"Evaluated {len(records)} records. TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Sensitivity (Se):           {se*100:.2f}%  (reference: 98.86%)")
    print(f"Positive Predictivity (+P): {pp*100:.2f}%  (reference: 99.03%)")
    if abs(se * 100 - 98.86) < 0.05 and abs(pp * 100 - 99.03) < 0.05:
        print("MATCHES the documented reference figures -- this is a faithful port.")
    else:
        print("DOES NOT match the documented reference figures within 0.05pp -- investigate before porting.")


# =============================================================================
# Check 2: run against this project's own real recorded sessions
# (database/ecg_recordings/*.csv), as a self-consistency + signal-quality
# check. These recordings carry "peak_valid" from the SAME algorithm
# running live in the browser at capture time, so close agreement here is
# a meaningful sanity check on this offline Python copy.
# =============================================================================
def load_session_csv(path):
    t_ms, raw, ref_peak_valid = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t_ms.append(float(row["t_ms"]))
                raw.append(float(row["raw"]))
                ref_peak_valid.append(int(row["peak_valid"]))
            except (KeyError, ValueError):
                continue
    return np.array(t_ms), np.array(raw), np.array(ref_peak_valid)


def run_sessions_check():
    print("=" * 70)
    print("CHECK 2: real recorded ECG sessions (database/ecg_recordings/)")
    print("=" * 70)

    csv_paths = sorted(glob.glob(os.path.join(ECG_RECORDINGS_DIR, "**", "*.csv"), recursive=True))
    if not csv_paths:
        print(f"No recorded sessions found under {ECG_RECORDINGS_DIR}")
        return

    for path in csv_paths:
        t_ms, raw, ref_peak_valid = load_session_csv(path)
        if len(raw) == 0:
            print(f"{os.path.basename(path)}: no usable rows, skipping")
            continue

        dt = np.median(np.diff(t_ms))
        fs_est = 1000.0 / dt if dt > 0 else FS
        if abs(fs_est - FS) > 5:
            print(f"{os.path.basename(path)}: WARNING sample rate ~{fs_est:.1f}Hz, expected {FS}Hz -- results below may be off")

        _, detected = run_detector(raw, fs=FS)
        ref_peak_idx = np.where(ref_peak_valid == 1)[0]

        # Match this run's detections against the session's own recorded
        # peaks (same 150ms tolerance convention).
        tolerance_samples = TOLERANCE_MS * FS / 1000
        matched_ref = np.zeros(len(ref_peak_idx), dtype=bool)
        matched_det = np.zeros(len(detected), dtype=bool)
        for i, d in enumerate(detected):
            diffs = np.abs(ref_peak_idx - d)
            candidates = np.where((diffs <= tolerance_samples) & (~matched_ref))[0]
            if len(candidates) > 0:
                j = candidates[np.argmin(diffs[candidates])]
                matched_ref[j] = True
                matched_det[i] = True

        tp = matched_det.sum()
        fp = (~matched_det).sum()
        fn = (~matched_ref).sum()
        agreement = tp / max(len(ref_peak_idx), 1) * 100

        duration_s = (t_ms[-1] - t_ms[0]) / 1000.0
        est_bpm = len(detected) / duration_s * 60.0 if duration_s > 0 else 0

        print(f"\n{os.path.relpath(path, ECG_RECORDINGS_DIR)}")
        print(f"  {len(raw)} samples, {duration_s:.1f}s, ~{fs_est:.1f}Hz")
        print(f"  This script detected {len(detected)} beats (~{est_bpm:.0f} bpm); "
              f"session's own recorded peaks: {len(ref_peak_idx)}")
        print(f"  Agreement with session's own peaks: {agreement:.1f}% (TP={tp} FP={fp} FN={fn}, 150ms tolerance)")


def report_coefficients():
    print("=" * 70)
    print("Filter coefficients + detection thresholds @ 250Hz (for Phase 2 port)")
    print("=" * 70)
    for name, f in [("Highpass 0.5Hz Q=0.707", highpass(FS, 0.5, 0.707)),
                    ("Lowpass 40Hz Q=0.707", lowpass(FS, 40, 0.707)),
                    ("Notch 50Hz Q=30", notch(FS, 50, 30))]:
        print(f"{name}: b0={f.b0:.8f} b1={f.b1:.8f} b2={f.b2:.8f} a1={f.a1:.8f} a2={f.a2:.8f}")
    mwi_window = round(0.15 * FS)
    print(f"\nMWI window: round(0.15*{FS:.0f}) = {mwi_window} samples")
    print(f"Backtrack delay: round(round(0.15*{FS:.0f})/2)+2 = {BACKTRACK_DELAY} samples")
    print(f"Backtrack search: +/-{BACKTRACK_SEARCH} samples")
    print("Threshold update: SPKI = 0.125*V + 0.875*SPKI on confirmed peak, "
          "NPKI = 0.125*V + 0.875*NPKI otherwise")
    print("Detection threshold: threshold1 = NPKI + 0.25*(SPKI-NPKI)")
    print("Refractory: 200ms | Warm-up: 2s (round(2*fs) samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitbih", action="store_true", help="also re-run the full MIT-BIH reproduction (~2 min)")
    args = parser.parse_args()

    report_coefficients()
    print()
    run_sessions_check()
    print()
    if args.mitbih:
        run_mitbih_check()
    else:
        print("(skipping MIT-BIH reproduction check -- pass --mitbih to run it, ~2 min)")
