#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot a full-time, all-channel DAS wiggle plot after bandpass filtering.

Filtering is applied to the full signal. Optional downsampling is used only for
plotting, so the saved figure remains practical to browse.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.signal import butter, sosfiltfilt


def parse_args():
    p = argparse.ArgumentParser(description="Plot full-time all-channel bandpassed DAS wiggle plot.")
    p.add_argument("--input", "-i", required=True, help="Input CSV with ch_* columns.")
    p.add_argument("--output", "-o", required=True, help="Output PNG path.")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz.")
    p.add_argument("--lowcut", type=float, default=0.1, help="Bandpass low cut in Hz.")
    p.add_argument("--highcut", type=float, default=20.0, help="Bandpass high cut in Hz.")
    p.add_argument("--filter-order", type=int, default=4, help="Butterworth filter order.")
    p.add_argument(
        "--max-points-per-channel",
        type=int,
        default=12000,
        help="Maximum plotted points per channel after filtering. Default: 12000.",
    )
    p.add_argument(
        "--scale-percentile",
        type=float,
        default=99.0,
        help="Global abs percentile used to scale wiggle amplitude. Default: 99.",
    )
    p.add_argument("--fig-width", type=float, default=22.0, help="Figure width in inches.")
    p.add_argument("--fig-height", type=float, default=18.0, help="Figure height in inches.")
    p.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    return p.parse_args()


def channel_sort_key(name):
    try:
        return int(str(name).split("_", 1)[1])
    except (IndexError, ValueError):
        return str(name)


def butter_bandpass_sos(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-9)
    high = min(highcut / nyq, 0.999999)
    if not 0 < low < high < 1:
        raise ValueError(f"Invalid bandpass range {lowcut}-{highcut} Hz for fs={fs} Hz.")
    return butter(order, [low, high], btype="band", output="sos")


def bandpass_filter_array(X, fs, lowcut, highcut, order):
    sos = butter_bandpass_sos(lowcut, highcut, fs, order)
    Xf = np.zeros_like(X, dtype=np.float64)
    for c in range(X.shape[1]):
        col = X[:, c].astype(np.float64)
        col = col - np.nanmean(col)
        Xf[:, c] = sosfiltfilt(sos, col)
    return Xf


def build_segments(times, Y, scale):
    n_samples, n_channels = Y.shape
    segments = []
    for ch in range(n_channels):
        y = Y[:, ch] / scale + ch
        segments.append(np.column_stack([times, y]))
    return segments


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Loading CSV: {args.input}")
    df = pd.read_csv(args.input)
    ch_cols = [c for c in df.columns if str(c).lower().startswith("ch_")]
    ch_cols = sorted(ch_cols, key=channel_sort_key)
    if not ch_cols:
        raise ValueError("No ch_* columns found.")

    X = df[ch_cols].to_numpy(dtype=np.float64)
    n_samples, n_channels = X.shape
    duration = n_samples / args.fs
    print(f"[Info] Loaded {n_samples} samples, {n_channels} channels, duration {duration:.2f}s.")

    print(f"[Info] Filtering full signal with {args.lowcut:g}-{args.highcut:g} Hz bandpass.")
    Xf = bandpass_filter_array(
        X,
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
        order=args.filter_order,
    )

    step = max(1, int(np.ceil(n_samples / max(1, args.max_points_per_channel))))
    if step > 1:
        print(f"[Info] Downsampling for plotting only: every {step} samples.")
    Xp = Xf[::step, :]
    times = np.arange(0, n_samples, step, dtype=np.float64)[: Xp.shape[0]] / args.fs

    scale = np.nanpercentile(np.abs(Xp), args.scale_percentile)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(Xp))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    segments = build_segments(times, Xp, scale)
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    lc = LineCollection(segments, colors="#1f77b4", linewidths=0.35)
    ax.add_collection(lc)

    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.set_ylim(-1, n_channels)
    tick_step = 5 if n_channels <= 120 else 10
    yticks = np.arange(0, n_channels, tick_step)
    ax.set_yticks(yticks)
    ax.set_yticklabels([ch_cols[i] for i in yticks])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(f"Full-channel wiggle plot | bandpass {args.lowcut:g}-{args.highcut:g} Hz")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(f"[Finish] Saved: {output}")


if __name__ == "__main__":
    main()
