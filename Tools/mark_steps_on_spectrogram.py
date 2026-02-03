#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def robust_z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Robust z-score using median and MAD."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad + eps)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    w = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, w, mode="same")


def compute_band_energy(D_mag: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Sum power in [fmin,fmax] over frequency bins. D_mag shape: [F, T]."""
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(idx) < 2:
        raise ValueError(f"No freq bins in band [{fmin}, {fmax}] Hz. Check sr/n_fft.")
    power = (D_mag[idx, :] ** 2).sum(axis=0)  # [T]
    return power


def main():
    ap = argparse.ArgumentParser(description="Mark footstep times on spectrogram and export CSV.")
    ap.add_argument("-i", "--input", required=True, help="Input audio file (wav/mp3/m4a/etc.)")
    ap.add_argument("-o", "--out_prefix", default="out", help="Output prefix for png/csv")
    ap.add_argument("--sr", type=int, default=16000, help="Target sampling rate")
    ap.add_argument("--n_fft", type=int, default=2048, help="STFT n_fft")
    ap.add_argument("--hop", type=int, default=160, help="STFT hop length (samples). 160 @16k = 10ms")
    ap.add_argument("--fmin", type=float, default=4000.0, help="Band energy min freq (Hz)")
    ap.add_argument("--fmax", type=float, default=10000.0, help="Band energy max freq (Hz)")
    ap.add_argument("--smooth_ms", type=float, default=60.0, help="Smoothing window for band curve (ms)")
    ap.add_argument("--min_interval", type=float, default=0.5, help="Minimum time between steps (s)")
    ap.add_argument("--prom", type=float, default=0.2,
                    help="Peak prominence threshold on robust-z curve (bigger => fewer peaks)")
    ap.add_argument("--peak_height", type=float, default=1.0,
                    help="Minimum peak height on robust-z curve (bigger => fewer peaks)")
    ap.add_argument("--top_db", type=float, default=80.0, help="Spectrogram dynamic range (dB)")
    args = ap.parse_args()

    # 1) Load audio (mono)
    y, sr = librosa.load(args.input, sr=args.sr, mono=True)

    # 2) STFT magnitude
    D = librosa.stft(y, n_fft=args.n_fft, hop_length=args.hop, window="hann", center=True)
    D_mag = np.abs(D)  # [F, T]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=args.n_fft)
    times = librosa.frames_to_time(np.arange(D_mag.shape[1]), sr=sr, hop_length=args.hop)

    # 3) Band energy curve (where footsteps are more visible)
    band = compute_band_energy(D_mag, freqs, args.fmin, args.fmax)
    band = np.log(band + 1e-12)  # log compress
    band_z = robust_z(band)

    # 4) Smooth in time
    frame_dt = args.hop / sr
    smooth_frames = max(1, int(round((args.smooth_ms / 1000.0) / frame_dt)))
    band_s = moving_average(band_z, smooth_frames)

    # 5) Peak detection
    min_dist_frames = max(1, int(round(args.min_interval / frame_dt)))
    peaks, props = find_peaks(
        band_s,
        distance=min_dist_frames,
        prominence=args.prom,
        height=args.peak_height
    )

    step_times = times[peaks]

    # 6) Save CSV
    csv_path = f"{args.out_prefix}_steps.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "time_sec", "band_score"])
        for i, (t, pidx) in enumerate(zip(step_times, peaks)):
            w.writerow([i, f"{t:.6f}", f"{band_s[pidx]:.6f}"])

    # 7) Plot spectrogram + step lines + band curve
    S_db = librosa.amplitude_to_db(D_mag, ref=np.max, top_db=args.top_db)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)

    ax0 = fig.add_subplot(gs[0])
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=args.hop,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax0
    )
    ax0.set_title(f"Spectrogram with detected steps  (band {args.fmin:.0f}-{args.fmax:.0f} Hz)")
    ax0.set_ylim(0, 6000)  # 你也可以改成 12000 看全频
    for t in step_times:
        ax0.axvline(t, color="cyan", linewidth=1.0, alpha=0.9)
    cbar = fig.colorbar(img, ax=ax0, format="%+2.0f dB")
    cbar.set_label("dB")

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(times, band_s, linewidth=1.0)
    ax1.scatter(step_times, band_s[peaks], s=18)
    ax1.set_ylabel("band(z) smoothed")
    ax1.set_xlabel("time (s)")
    ax1.grid(True, alpha=0.3)

    png_path = f"{args.out_prefix}_spectrogram_steps.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] detected steps: {len(step_times)}")
    print(f"[OK] saved: {csv_path}")
    print(f"[OK] saved: {png_path}")


if __name__ == "__main__":
    main()
