#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks


# ----------------------------
# Filters & helpers
# ----------------------------
def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    if low <= 0 or high >= 1 or low >= high:
        raise ValueError(f"Bad bandpass: low={low*nyq}, high={high*nyq}, fs={fs}")
    b, a = butter(order, [low, high], btype="bandpass")
    return b, a

def bandpass_filt_1d(x, fs, low, high, order=4):
    b, a = butter_bandpass(low, high, fs, order=order)
    return filtfilt(b, a, x)

def robust_z(x, eps=1e-9):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad + eps)

def moving_average(x, win):
    if win <= 1:
        return x
    w = np.ones(int(win), dtype=np.float64) / float(win)
    return np.convolve(x, w, mode="same")

def short_time_rms(x, win):
    """RMS envelope via moving average of squared signal."""
    win = max(1, int(win))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win, dtype=np.float64) / float(win)
    ma = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ma + 1e-12)

def event_match_score(audio_events, das_events, dt, tol=0.12):
    """Count how many shifted audio events match any das event within ±tol."""
    if len(audio_events) == 0 or len(das_events) == 0:
        return 0
    shifted = audio_events + dt
    j = 0
    score = 0
    for t in shifted:
        while j < len(das_events) and das_events[j] < t - tol:
            j += 1
        if j < len(das_events) and abs(das_events[j] - t) <= tol:
            score += 1
    return score

def search_best_dt(audio_events, das_candidates, dt_min=-20.0, dt_max=20.0, dt_step=0.005, tol=0.12):
    best_dt, best_score = 0.0, -1
    for dt in np.arange(dt_min, dt_max + 1e-12, dt_step):
        sc = event_match_score(audio_events, das_candidates, dt, tol=tol)
        if sc > best_score:
            best_score, best_dt = sc, dt
    return best_dt, best_score


# ----------------------------
# DAS -> 1D curve & candidates
# ----------------------------
def das_to_1d_curve(das_2d, fs=2000, win_ms=50, topk=10):
    """
    Convert DAS [T, C] into 1D curve E(t) using topK RMS across channels.
    This is more robust than choosing a single channel when footsteps are weak.
    """
    win = max(1, int(round(win_ms * 1e-3 * fs)))
    x2 = das_2d ** 2
    kernel = np.ones(win, dtype=np.float64) / float(win)
    ma = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, x2)
    rms = np.sqrt(ma + 1e-12)  # [T, C]

    if topk >= rms.shape[1]:
        E = rms.mean(axis=1)
    else:
        part = np.partition(rms, -topk, axis=1)[:, -topk:]
        E = part.mean(axis=1)
    return E

def detect_das_candidates(E, fs=2000, smooth_ms=80, min_interval=0.25, z_height=0.8):
    """
    Very loose DAS candidate peaks (for alignment only).
    """
    t = np.arange(len(E)) / fs
    x = np.log(E + 1e-12)
    xz = robust_z(x)

    win = max(1, int(round(smooth_ms * 1e-3 * fs)))
    xs = moving_average(xz, win)

    dist = max(1, int(round(min_interval * fs)))
    peaks, _ = find_peaks(xs, distance=dist, height=z_height)
    return t[peaks], peaks, xs


# ----------------------------
# Audio: 2k-10k bandpass -> RMS envelope -> events
# ----------------------------
def audio_bandpass_envelope(y, sr, bp_low=2000.0, bp_high=10000.0,
                            order=4, env_ms=20.0, smooth_ms=40.0):
    """
    Bandpass audio (2k-10k), then compute short-time RMS envelope for plotting & event detection.
    """
    y_bp = bandpass_filt_1d(y, fs=sr, low=bp_low, high=bp_high, order=order)

    env_win = max(1, int(round(env_ms * 1e-3 * sr)))
    env = short_time_rms(y_bp, env_win)

    sm_win = max(1, int(round(smooth_ms * 1e-3 * sr)))
    env_s = moving_average(env, sm_win)

    t = np.arange(len(env_s)) / sr
    return t, env_s, y_bp

def detect_audio_events(t, env_s, min_interval=0.35, prom=2.0, height=1.0):
    """
    Peak-pick on robust-z envelope.
    """
    dt = np.median(np.diff(t))
    dist = max(1, int(round(min_interval / dt)))

    z = robust_z(np.log(env_s + 1e-12))
    # optional: a bit more smoothing in event domain
    z = moving_average(z, max(1, int(round(0.02 / dt))))  # ~20ms in samples

    peaks, props = find_peaks(z, distance=dist, prominence=prom, height=height)
    return t[peaks], peaks, z


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Align DAS and audio, output QC plot (DAS 5-10Hz BP, Audio 2k-10kHz BP).")
    ap.add_argument("--das_csv", required=True, help="DAS csv file. Header: ch_0..ch_xxx, rows: samples")
    ap.add_argument("--audio", required=True, help="Audio file (wav/m4a/mp3).")
    ap.add_argument("--out_png", default="qc_align.png", help="Output QC image path.")

    # DAS fixed per your spec
    ap.add_argument("--das_fs", type=int, default=2000, help="DAS sampling rate (Hz), default 2000")
    ap.add_argument("--das_bp_low", type=float, default=5.0)
    ap.add_argument("--das_bp_high", type=float, default=10.0)
    ap.add_argument("--das_bp_order", type=int, default=4)

    # Audio bandpass per your new spec
    ap.add_argument("--audio_sr", type=int, default=48000, help="Audio resample rate (must support 10kHz).")
    ap.add_argument("--audio_bp_low", type=float, default=2000.0)
    ap.add_argument("--audio_bp_high", type=float, default=10000.0)
    ap.add_argument("--audio_bp_order", type=int, default=4)
    ap.add_argument("--audio_env_ms", type=float, default=20.0, help="RMS envelope window (ms)")
    ap.add_argument("--audio_smooth_ms", type=float, default=40.0, help="Envelope smoothing (ms)")

    # Event detection
    ap.add_argument("--step_min_interval", type=float, default=0.35, help="Min step interval (s)")
    ap.add_argument("--audio_prom", type=float, default=2.0)
    ap.add_argument("--audio_height", type=float, default=1.0)

    # DAS curve for alignment
    ap.add_argument("--das_win_ms", type=float, default=50.0)
    ap.add_argument("--das_topk", type=int, default=10)
    ap.add_argument("--das_candidate_height", type=float, default=0.8)
    ap.add_argument("--das_candidate_smooth_ms", type=float, default=80.0)

    # Alignment search
    ap.add_argument("--dt_min", type=float, default=-20.0)
    ap.add_argument("--dt_max", type=float, default=20.0)
    ap.add_argument("--dt_step", type=float, default=0.005, help="seconds, 5ms grid")
    ap.add_argument("--match_tol", type=float, default=0.12, help="event match tolerance (s)")

    # Plot range on AUDIO time axis
    ap.add_argument("--plot_start", type=float, default=0.0, help="Audio time start (s) for plot")
    ap.add_argument("--plot_dur", type=float, default=20.0, help="Duration (s) for plot window")
    args = ap.parse_args()

    # 1) Load DAS CSV: [T, C]
    df = pd.read_csv(args.das_csv)
    das = df.to_numpy(dtype=np.float64)
    T_das = das.shape[0]
    t_das = np.arange(T_das) / args.das_fs

    # 2) Load audio (mono) at sr supporting 10kHz
    y, sr = librosa.load(args.audio, sr=args.audio_sr, mono=True)

    # 3) Audio 2k-10k bandpass -> envelope -> audio events
    t_a, env_s, y_bp = audio_bandpass_envelope(
        y, sr,
        bp_low=args.audio_bp_low, bp_high=args.audio_bp_high,
        order=args.audio_bp_order,
        env_ms=args.audio_env_ms,
        smooth_ms=args.audio_smooth_ms
    )
    audio_events, audio_peaks, audio_z = detect_audio_events(
        t_a, env_s,
        min_interval=args.step_min_interval,
        prom=args.audio_prom,
        height=args.audio_height
    )

    # 4) DAS -> 1D energy curve + candidate peaks (loose)
    E = das_to_1d_curve(das, fs=args.das_fs, win_ms=args.das_win_ms, topk=args.das_topk)
    das_candidates, das_c_peaks, das_curve_z = detect_das_candidates(
        E, fs=args.das_fs,
        smooth_ms=args.das_candidate_smooth_ms,
        min_interval=0.25,
        z_height=args.das_candidate_height
    )

    # 5) Align by event matching: t_das ≈ t_audio + best_dt
    best_dt, best_score = search_best_dt(
        audio_events, das_candidates,
        dt_min=args.dt_min, dt_max=args.dt_max,
        dt_step=args.dt_step, tol=args.match_tol
    )
    print(f"[ALIGN] best_dt = {best_dt:.4f} s, matched = {best_score} events")

    # 6) Build QC curves on AUDIO time axis
    t0 = args.plot_start
    t1 = args.plot_start + args.plot_dur

    # Audio envelope for plotting: resample to DAS fs for clean overlay
    t_grid = np.arange(int(t0 * args.das_fs), int(t1 * args.das_fs)) / args.das_fs  # AUDIO time axis

    # Audio envelope -> interpolate on grid (audio time)
    env_grid = np.interp(t_grid, t_a, env_s, left=env_s[0], right=env_s[-1])

    # DAS 1D curve sampled at (t_audio + best_dt)
    t_query = np.clip(t_grid + best_dt, 0.0, t_das[-1])
    E_grid = np.interp(t_query, t_das, E, left=E[0], right=E[-1])

    # 7) Apply required bandpasses:
    # DAS: 5-10 Hz 4th order BP
    E_bp = bandpass_filt_1d(E_grid, fs=args.das_fs, low=args.das_bp_low, high=args.das_bp_high, order=args.das_bp_order)

    # Audio: 2k-10k already applied on raw audio, but we are plotting RMS envelope of that bandpassed audio.
    # For readability, we keep envelope (env_grid) as is (optionally normalize).
    env_plot = robust_z(np.log(env_grid + 1e-12))

    # 8) Event lines (in AUDIO time axis)
    ev = audio_events[(audio_events >= t0) & (audio_events <= t1)]

    # 9) Plot QC
    plt.figure(figsize=(16, 7))
    ax = plt.gca()

    ax.plot(t_grid, E_bp, linewidth=1.0, label="DAS 5–10 Hz BP (aligned to audio time)")
    ax.plot(t_grid, env_plot, linewidth=1.0, label="Audio 2–10 kHz BP → RMS envelope (audio time axis)")

    for tt in ev:
        ax.axvline(tt, color="cyan", linewidth=0.9, alpha=0.9)

    ax.set_title(f"QC Alignment (x-axis = audio time). best_dt={best_dt:.4f}s, matched={best_score}")
    ax.set_xlabel("Audio time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=180)
    plt.close()

    print(f"[OK] QC plot saved: {args.out_png}")
    print("[INFO] Cyan vertical lines = audio-detected step events (audio time axis)")
    print("[INFO] DAS sampled at (t_audio + best_dt), then 5–10Hz bandpass applied")
    print("[INFO] Audio shown as RMS envelope of 2–10kHz bandpassed waveform")


if __name__ == "__main__":
    main()
