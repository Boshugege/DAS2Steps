#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export waveform plots around CSV2Steps predicted footstep events.

For each selected event, the script cuts a time/channel window from the
original DAS signal and writes two waveform plots:

  - raw_waveforms.png
  - bandpass_0p1_30Hz_waveforms.png

The plots are stacked waveforms, not energy heatmaps.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def parse_args():
    p = argparse.ArgumentParser(
        description="Export raw and 0.1-30 Hz bandpassed waveform windows around predicted footstep events."
    )
    p.add_argument("--input", "-i", required=True, help="Input DAS file, CSV with ch_* columns or TDMS.")
    p.add_argument("--detections", "-d", default="output/detections.csv", help="CSV2Steps detections CSV.")
    p.add_argument("--outdir", "-o", default="output/step_waveform_windows", help="Output parent folder.")
    p.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz.")
    p.add_argument("--start-event", type=int, default=20, help="First event index to export. Default is 20.")
    p.add_argument("--end-event", type=int, default=50, help="Last event index to export, inclusive. Default is 50.")
    p.add_argument(
        "--index-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether start/end event numbers are 0-based or 1-based. Default is 1.",
    )
    p.add_argument("--window-sec", type=float, default=8.0, help="Time window length in seconds.")
    p.add_argument("--channels", type=int, default=10, help="Number of adjacent channels to plot.")
    p.add_argument(
        "--time-column",
        default="time_s",
        help="Detection CSV time column in seconds. Default: time_s.",
    )
    p.add_argument(
        "--channel-column",
        default="channel_center",
        help="Detection CSV channel column. Default: channel_center.",
    )
    p.add_argument("--lowcut", type=float, default=0.1, help="Bandpass low cut in Hz.")
    p.add_argument("--highcut", type=float, default=30.0, help="Bandpass high cut in Hz.")
    p.add_argument("--filter-order", type=int, default=4, help="Butterworth filter order.")
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Robust-normalize each channel within each plot window before plotting.",
    )
    p.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    return p.parse_args()


def load_tdms_to_dataframe(path):
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError("Reading TDMS requires nptdms. Install it with: pip install nptdms") from exc

    tdms_file = TdmsFile.read(path)
    channels = []
    for group in tdms_file.groups():
        for channel in group.channels():
            channels.append(channel)

    if not channels:
        raise ValueError("No channels were found in the TDMS file.")

    data = [np.asarray(ch[:], dtype=np.float64) for ch in channels]
    min_len = min(len(x) for x in data)
    if min_len == 0:
        raise ValueError("At least one TDMS channel is empty.")

    if len({len(x) for x in data}) != 1:
        print(f"[Warn] TDMS channels have different lengths; truncating to {min_len} samples.")

    arr = np.stack([x[:min_len] for x in data], axis=1)
    return pd.DataFrame(arr, columns=[f"ch_{i}" for i in range(arr.shape[1])])


def load_input_dataframe(path):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tdms":
        return load_tdms_to_dataframe(path)
    raise ValueError("Input must be a .csv or .tdms file.")


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
    padlen = 3 * (2 * sos.shape[0] + 1)
    if X.shape[0] <= padlen:
        raise ValueError(
            f"Window has {X.shape[0]} samples, but filtfilt needs more than {padlen}. "
            "Use a longer input or lower filter order."
        )

    Xf = np.zeros_like(X, dtype=np.float64)
    for c in range(X.shape[1]):
        col = X[:, c].astype(np.float64)
        col = col - np.nanmean(col)
        Xf[:, c] = sosfiltfilt(sos, col)
    return Xf


def choose_channel_column(detections, requested):
    if requested in detections.columns:
        return requested

    fallbacks = ["channel_center", "channel_centroid", "channel_argmax", "channel"]
    for col in fallbacks:
        if col in detections.columns:
            print(f"[Warn] Channel column '{requested}' not found; using '{col}' instead.")
            return col

    raise ValueError(
        "Could not find a channel column in detections CSV. "
        "Try --channel-column with one of: " + ", ".join(map(str, detections.columns))
    )


def select_channel_indices(center_channel, n_channels_total, n_plot_channels):
    n_plot_channels = max(1, int(n_plot_channels))
    center = int(round(float(center_channel)))
    center = int(np.clip(center, 0, n_channels_total - 1))

    left = n_plot_channels // 2
    right = n_plot_channels - left - 1
    start = center - left
    end = center + right + 1

    if start < 0:
        end += -start
        start = 0
    if end > n_channels_total:
        start -= end - n_channels_total
        end = n_channels_total
    start = max(0, start)
    return np.arange(start, end, dtype=int), center


def robust_normalize_columns(X):
    med = np.nanmedian(X, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(X - med), axis=0, keepdims=True)
    return (X - med) / (1.4826 * mad + 1e-9)


def band_label(lowcut, highcut):
    def fmt(value):
        text = f"{value:g}"
        return text.replace(".", "p")

    return f"{fmt(lowcut)}_{fmt(highcut)}Hz"


def plot_waveforms(X, times, channel_names, event_time, event_channel, title, outpath, normalize, dpi):
    Y = robust_normalize_columns(X) if normalize else X.astype(np.float64)
    if Y.shape[0] == 0 or Y.shape[1] == 0:
        raise ValueError("Empty waveform window.")

    # One robust scale for the whole window keeps relative channel amplitudes visible.
    scale = np.nanpercentile(np.abs(Y), 98)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(Y))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    spacing = 2.8
    offsets = np.arange(Y.shape[1], dtype=float) * spacing

    fig_h = max(4.0, 0.72 * Y.shape[1] + 1.8)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    for idx, name in enumerate(channel_names):
        y = Y[:, idx] / scale
        ax.plot(times, y + offsets[idx], linewidth=0.8, color="#1f77b4")

    ax.axvline(event_time, color="#d62728", linewidth=1.2, alpha=0.9)
    if event_channel in channel_names:
        y_idx = channel_names.index(event_channel)
        ax.axhline(offsets[y_idx], color="#d62728", linewidth=0.8, alpha=0.55)

    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def write_event_metadata(outpath, metadata):
    pd.DataFrame([metadata]).to_csv(outpath, index=False)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Loading DAS data: {args.input}")
    df = load_input_dataframe(args.input)
    ch_cols = [c for c in df.columns if str(c).lower().startswith("ch_")]
    ch_cols = sorted(ch_cols, key=channel_sort_key)
    if not ch_cols:
        raise ValueError("No ch_* columns found in input data.")

    X = df[ch_cols].to_numpy(dtype=np.float64)
    n_samples, n_channels = X.shape
    total_sec = n_samples / float(args.fs)
    print(f"[Info] Loaded {n_samples} samples, {n_channels} channels, duration {total_sec:.2f} s.")

    detections = pd.read_csv(args.detections)
    if args.time_column not in detections.columns:
        raise ValueError(f"Time column '{args.time_column}' not found in {args.detections}.")
    channel_col = choose_channel_column(detections, args.channel_column)

    start_pos = args.start_event - args.index_base
    end_pos = args.end_event - args.index_base
    if start_pos < 0 or end_pos < start_pos:
        raise ValueError("Invalid event range. Check --start-event, --end-event and --index-base.")
    selected = detections.iloc[start_pos : end_pos + 1].copy()
    if selected.empty:
        raise ValueError("Selected event range is empty.")

    half_window = args.window_sec / 2.0

    event_specs = []
    needed_channel_indices = set()
    for det_pos, row in selected.iterrows():
        channel_value = float(row[channel_col])
        channel_idx, center_idx = select_channel_indices(channel_value, n_channels, args.channels)
        event_specs.append((det_pos, row, channel_value, channel_idx, center_idx))
        needed_channel_indices.update(int(i) for i in channel_idx)

    needed_channel_indices = np.array(sorted(needed_channel_indices), dtype=int)
    filtered_local_index = {int(global_i): local_i for local_i, global_i in enumerate(needed_channel_indices)}
    print(
        f"[Info] Filtering full signal for {len(needed_channel_indices)} needed channels "
        f"with {args.lowcut:g}-{args.highcut:g} Hz bandpass."
    )
    filtered_filename = f"bandpass_{band_label(args.lowcut, args.highcut)}_waveforms.png"
    X_filtered_needed = bandpass_filter_array(
        X[:, needed_channel_indices],
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
        order=args.filter_order,
    )

    exported = 0
    for det_pos, row, channel_value, channel_idx, center_idx in event_specs:
        event_number = det_pos + args.index_base
        event_time = float(row[args.time_column])

        start_sample = int(round((event_time - half_window) * args.fs))
        end_sample = int(round((event_time + half_window) * args.fs))
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)
        if end_sample <= start_sample:
            print(f"[Warn] Skipping event {event_number}: empty time window.")
            continue

        channel_names = [ch_cols[i] for i in channel_idx]
        event_channel_name = ch_cols[center_idx]

        X_win = X[start_sample:end_sample, :][:, channel_idx]
        local_idx = [filtered_local_index[int(i)] for i in channel_idx]
        X_filt = X_filtered_needed[start_sample:end_sample, :][:, local_idx]
        t_win = np.arange(start_sample, end_sample, dtype=np.float64) / args.fs
        event_dir = outdir / f"event_{event_number:03d}_t{event_time:.3f}s_ch{channel_value:.1f}"
        event_dir.mkdir(parents=True, exist_ok=True)

        base_title = (
            f"Event {event_number}: predicted t={event_time:.3f}s, "
            f"{channel_col}={channel_value:.2f}"
        )
        plot_waveforms(
            X_win,
            t_win,
            channel_names,
            event_time,
            event_channel_name,
            base_title + " | raw waveform",
            event_dir / "raw_waveforms.png",
            args.normalize,
            args.dpi,
        )

        plot_waveforms(
            X_filt,
            t_win,
            channel_names,
            event_time,
            event_channel_name,
            base_title + f" | bandpass {args.lowcut:g}-{args.highcut:g} Hz",
            event_dir / filtered_filename,
            args.normalize,
            args.dpi,
        )

        metadata = {
            "event_number": event_number,
            "detection_row": int(det_pos),
            "time_s": event_time,
            "channel_column": channel_col,
            "channel_value": channel_value,
            "center_channel_name": event_channel_name,
            "window_start_s": start_sample / args.fs,
            "window_end_s": end_sample / args.fs,
            "channels": ",".join(channel_names),
            "raw_plot": "raw_waveforms.png",
            "filtered_plot": filtered_filename,
            "filter_applied_before_windowing": True,
        }
        for col in detections.columns:
            if col not in metadata:
                metadata[f"detection_{col}"] = row[col]
        write_event_metadata(event_dir / "metadata.csv", metadata)
        exported += 1

    print(f"[Finish] Exported {exported} event folders to: {outdir}")


if __name__ == "__main__":
    main()
