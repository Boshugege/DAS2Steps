#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用途：
  读取 TDMS 多通道数据，带通 -> 包络 -> 二值化 -> 估计主轨迹速度方向，
  旋转补洞并连通域提取，输出旋转矩形覆盖的热图。
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.signal import butter, filtfilt
from scipy.ndimage import binary_closing, label, rotate


def parse_args():
    p = argparse.ArgumentParser(description="Plot filtered signal heatmap from TDMS.")
    p.add_argument("--input", "-i", required=True, help="输入文件路径（支持 .tdms / .csv）")
    p.add_argument("--outdir", "-o", default="output", help="结果输出目录（默认 ./output）")
    p.add_argument("--fs", type=float, required=True, help="采样率 （Hz）")
    p.add_argument("--dx", type=float, default=1.0, help="相邻通道间距（米），默认 1.0")
    p.add_argument("--lowcut", type=float, default=5, help="带通滤波低频截止（Hz），默认 5 Hz")
    p.add_argument("--highcut", type=float, default=10, help="带通滤波高频截止（Hz），默认 10 Hz")
    p.add_argument("--filter_order", type=int, default=4, help="Butterworth 滤波器阶数")
    p.add_argument("--env_win_sec", type=float, default=0.5, help="RMS 包络窗口长度（秒）")
    p.add_argument("--env_hop_sec", type=float, default=0.1, help="RMS 包络步长（秒）")
    p.add_argument("--mad_k", type=float, default=2.5, help="MAD 自适应阈值系数 k（低阈值）")
    p.add_argument("--close_time_sec", type=float, default=5, help="形态学 closing 时间长度（秒）")
    p.add_argument("--close_dist_m", type=float, default=5.0, help="形态学 closing 通道半径（米）")
    p.add_argument("--theta_win_sec", type=float, default=2.0, help="估计速度方向的滑窗长度（秒）")
    p.add_argument("--theta_hop_sec", type=float, default=0.5, help="速度方向滑窗步长（秒）")
    p.add_argument("--speed_min", type=float, default=0.3, help="速度扫描范围下限（m/s）")
    p.add_argument("--speed_max", type=float, default=2.5, help="速度扫描范围上限（m/s）")
    p.add_argument("--speed_step", type=float, default=0.05, help="速度扫描步长（m/s）")
    p.add_argument("--theta_jump_deg", type=float, default=20.0, help="方向跳变阈值（度），超过则切段")
    p.add_argument("--bbox_expand_m", type=float, default=3.0, help="旋转矩形宽度扩展（米）")
    p.add_argument("--bbox_expand_time_sec", type=float, default=5, help="旋转矩形时间方向扩展（秒）")
    p.add_argument("--min_cc_area", type=int, default=50, help="最小连通域像素数")
    return p.parse_args()


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0:
        low = 1e-6
    if high >= 1:
        high = 0.999999
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter_array(X, fs, lowcut=10.0, highcut=60.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    Xf = np.zeros_like(X)
    for c in range(X.shape[1]):
        col = X[:, c]
        col = col - np.mean(col)
        colf = filtfilt(b, a, col, axis=0, method="pad")
        Xf[:, c] = colf
    return Xf


def rms_envelope(X, fs, win_sec, hop_sec):
    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    if win <= 1 or hop <= 0:
        raise ValueError("env_win_sec 和 env_hop_sec 需要大于 0")
    n_time = X.shape[0]
    if n_time < win:
        return np.zeros((0, X.shape[1])), win, hop

    n_frames = 1 + (n_time - win) // hop
    starts = np.arange(n_frames) * hop
    ends = starts + win
    energy = np.cumsum(X * X, axis=0, dtype=np.float64)
    energy = np.vstack([np.zeros((1, energy.shape[1])), energy])
    sums = energy[ends] - energy[starts]
    rms = np.sqrt(sums / float(win))
    return rms, win, hop


def mad_threshold_mask(E, k):
    med = np.nanmedian(E, axis=0)
    mad = np.nanmedian(np.abs(E - med), axis=0)
    thr = med + k * 1.4826 * mad
    return E > thr


def _estimate_best_slope(mask, hop_sec, dx, speed_min, speed_max, speed_step, min_points=50):
    points = np.argwhere(mask)
    if points.shape[0] < min_points:
        return None
    t_idx = points[:, 0].astype(float)
    ch_idx = points[:, 1].astype(float)
    best_score = -1.0
    best_m = None
    speeds = np.arange(speed_min, speed_max + 1e-9, speed_step)
    for v in speeds:
        m = (v / dx) * hop_sec
        b = ch_idx - m * t_idx
        b_round = np.round(b).astype(int)
        counts = np.bincount(b_round - b_round.min())
        score = float(counts.max())
        if score > best_score:
            best_score = score
            best_m = m
    return best_m


def _theta_segments(theta_seq, frame_idx_seq, theta_jump_deg):
    segments = []
    if len(theta_seq) == 0:
        return segments
    current = [0]
    for i in range(1, len(theta_seq)):
        prev = theta_seq[i - 1]
        cur = theta_seq[i]
        split = False
        if prev * cur < 0:
            split = True
        if abs(cur - prev) > theta_jump_deg:
            split = True
        if split:
            segments.append(current)
            current = [i]
        else:
            current.append(i)
    if current:
        segments.append(current)
    return [(frame_idx_seq[s[0]], frame_idx_seq[s[-1]]) for s in segments]


def _oriented_bbox(points_t, points_ch, expand_m, expand_time_sec, dx):
    if len(points_t) < 3:
        return None
    pts = np.vstack([points_t, points_ch]).T
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    major = eigvecs[:, 0]
    minor = eigvecs[:, 1]
    proj_major = centered @ major
    proj_minor = centered @ minor
    min_major, max_major = proj_major.min(), proj_major.max()
    min_minor, max_minor = proj_minor.min(), proj_minor.max()
    expand_ch = expand_m / dx
    min_major -= expand_time_sec
    max_major += expand_time_sec
    min_minor -= expand_ch
    max_minor += expand_ch
    corners = []
    for a, b in [
        (min_major, min_minor),
        (max_major, min_minor),
        (max_major, max_minor),
        (min_major, max_minor),
    ]:
        corner = mean + a * major + b * minor
        corners.append((corner[0], corner[1]))
    return corners


def _load_tdms_to_dataframe(path):
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError("读取 TDMS 需要 nptdms，请先安装：pip install nptdms") from exc

    tdms_file = TdmsFile.read(path)
    channels = []
    for group in tdms_file.groups():
        for channel in group.channels():
            channels.append(channel)
    if len(channels) == 0:
        raise ValueError("TDMS 中未检测到任何通道。")

    data_list = []
    lengths = []
    for ch in channels:
        data = np.asarray(ch[:], dtype=np.float64)
        data_list.append(data)
        lengths.append(len(data))

    min_len = min(lengths)
    if min_len == 0:
        raise ValueError("TDMS 通道数据为空。")
    if len(set(lengths)) != 1:
        print(f"[Warn] TDMS 通道长度不一致，将截断到最短长度 {min_len}。")

    data_stack = np.stack([d[:min_len] for d in data_list], axis=1)
    return data_stack


def _load_csv_to_array(path):
    # CSV 格式：第一行为 ch_0~ch_n 列名，第二行开始为各通道信号
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            header = f.readline().strip()
    except OSError as exc:
        raise ValueError(f"无法读取 CSV 文件: {path}") from exc

    if not header:
        raise ValueError("CSV 首行为空，期望列名 ch_0~ch_n。")

    col_names = [h.strip() for h in header.split(",")]
    if len(col_names) == 0:
        raise ValueError("CSV 未检测到任何列。")
    if not all(name.startswith("ch_") for name in col_names):
        print("[Warn] CSV 首行列名不全是 ch_*，将按数值矩阵继续读取。")

    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    except Exception as exc:
        raise ValueError("CSV 数值读取失败，请确认从第二行开始均为数字。") from exc

    if data.size == 0:
        raise ValueError("CSV 数据为空。")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.shape[1] != len(col_names):
        raise ValueError(
            f"CSV 列数不匹配：表头 {len(col_names)} 列，数据 {data.shape[1]} 列。"
        )
    return data


def _load_input_to_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tdms":
        return _load_tdms_to_dataframe(path)
    if ext == ".csv":
        return _load_csv_to_array(path)
    raise ValueError("仅支持 .tdms 或 .csv 输入文件。")


def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    X_raw = _load_input_to_array(args.input).astype(np.float64)

    # 输出未处理信号热图
    n_time_raw, n_ch_raw = X_raw.shape
    vmax_raw = np.percentile(np.abs(X_raw), 99)
    fig_raw, ax_raw = plt.subplots(figsize=(14, 3.5))
    im_raw = ax_raw.imshow(
        X_raw.T,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax_raw,
        vmax=vmax_raw,
        extent=(0, n_time_raw / args.fs, n_ch_raw - 1, 0),
    )
    ax_raw.set_title("Raw signal (channels x time)")
    ax_raw.set_ylabel("Channel index")
    ax_raw.set_xlabel("Time (s)")
    plt.colorbar(im_raw, ax=ax_raw, orientation="vertical", label="amplitude")
    fig_raw.tight_layout()
    fig_raw.savefig(os.path.join(outdir, "raw_heatmap.png"), dpi=200)
    plt.close(fig_raw)

    # 带通滤波
    print(f"[Info] 对每通道做带通滤波 {args.lowcut}-{args.highcut} Hz")
    X_filt = bandpass_filter_array(
        X_raw, fs=args.fs, lowcut=args.lowcut, highcut=args.highcut, order=args.filter_order
    )

    # RMS 包络 + log
    env, env_win, env_hop = rms_envelope(X_filt, args.fs, args.env_win_sec, args.env_hop_sec)
    if env.shape[0] == 0:
        raise ValueError("数据长度不足以计算包络，请减小 env_win_sec 或检查输入。")
    env_log = np.log10(env + 1e-12)

    # MAD 自适应阈值（低阈值）
    mask = mad_threshold_mask(env_log, args.mad_k)

    # 估计主轨迹方向（滑动窗口）
    theta_win_frames = max(1, int(round(args.theta_win_sec / args.env_hop_sec)))
    theta_hop_frames = max(1, int(round(args.theta_hop_sec / args.env_hop_sec)))
    theta_seq = []
    theta_frames = []
    for start in range(0, mask.shape[0] - theta_win_frames + 1, theta_hop_frames):
        win_mask = mask[start : start + theta_win_frames]
        m = _estimate_best_slope(
            win_mask,
            args.env_hop_sec,
            args.dx,
            args.speed_min,
            args.speed_max,
            args.speed_step,
        )
        if m is None:
            continue
        theta = float(np.degrees(np.arctan(m)))
        theta_seq.append(theta)
        theta_frames.append(start + theta_win_frames // 2)

    if theta_seq:
        theta_segments = _theta_segments(theta_seq, theta_frames, args.theta_jump_deg)
    else:
        theta_segments = [(0, mask.shape[0] - 1)]

    close_frames = max(1, int(round(args.close_time_sec / args.env_hop_sec)))
    close_ch_radius = int(round(args.close_dist_m / args.dx))
    close_ch = max(1, close_ch_radius * 2 + 1)
    close_struct = np.ones((close_frames, close_ch), dtype=bool)

    mask_connected = np.zeros_like(mask, dtype=bool)
    for seg_start, seg_end in theta_segments:
        if seg_end < seg_start:
            continue
        if theta_seq:
            seg_thetas = [
                th
                for th, fr in zip(theta_seq, theta_frames)
                if seg_start <= fr <= seg_end
            ]
            if not seg_thetas:
                continue
            theta = float(np.median(seg_thetas))
        else:
            theta = 0.0
        seg_mask = np.zeros_like(mask, dtype=float)
        seg_mask[seg_start : seg_end + 1] = mask[seg_start : seg_end + 1].astype(float)
        rotated = rotate(seg_mask, angle=-theta, reshape=False, order=0, mode="constant", cval=0.0)
        rotated_bin = rotated > 0.5
        rotated_closed = binary_closing(rotated_bin, structure=close_struct)
        restored = rotate(rotated_closed.astype(float), angle=theta, reshape=False, order=0, mode="constant", cval=0.0)
        mask_connected |= restored > 0.5

    # 连通域 + PCA 旋转矩形
    labeled, n_labels = label(mask_connected)
    segments_info = []
    for idx in range(1, n_labels + 1):
        coords = np.argwhere(labeled == idx)
        if coords.shape[0] < args.min_cc_area:
            continue
        t_sec = coords[:, 0].astype(float) * args.env_hop_sec
        ch = coords[:, 1].astype(float)
        corners = _oriented_bbox(t_sec, ch, args.bbox_expand_m, args.bbox_expand_time_sec, args.dx)
        if corners is None:
            continue
        segments_info.append(corners)

    n_time, n_ch = X_filt.shape

    # 输出带通滤波信号热图（无蒙版）
    vmax = np.percentile(np.abs(X_filt), 99)
    fig_bp, ax_bp = plt.subplots(figsize=(14, 3.5))
    im_bp = ax_bp.imshow(
        X_filt.T,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(0, n_time / args.fs, n_ch - 1, 0),
    )
    ax_bp.set_title("Bandpass-only signal (channels x time)")
    ax_bp.set_ylabel("Channel index")
    ax_bp.set_xlabel("Time (s)")
    plt.colorbar(im_bp, ax=ax_bp, orientation="vertical", label="amplitude")
    fig_bp.tight_layout()
    fig_bp.savefig(os.path.join(outdir, "bandpass_heatmap.png"), dpi=200)
    plt.close(fig_bp)

    # 输出带通滤波信号热图（叠加覆盖）
    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(
        X_filt.T,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(0, n_time / args.fs, n_ch - 1, 0),
    )
    ax.set_title("Bandpass-only signal (channels x time)")
    ax.set_ylabel("Channel index")
    ax.set_xlabel("Time (s)")
    plt.colorbar(im, ax=ax, orientation="vertical", label="amplitude")
    for corners in segments_info:
        poly = Polygon(
            corners,
            closed=True,
            fill=True,
            facecolor="yellow",
            edgecolor="yellow",
            alpha=0.25,
            linewidth=1.2,
        )
        ax.add_patch(poly)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "heatmap_boxed.png"), dpi=200)
    plt.close(fig)

    # 输出包络热图
    fig_env, ax_env = plt.subplots(figsize=(10, 4))
    env_vmax = np.percentile(env_log, 99)
    env_vmin = np.percentile(env_log, 5)
    im_env = ax_env.imshow(
        env_log.T,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="viridis",
        vmin=env_vmin,
        vmax=env_vmax,
        extent=(0, env.shape[0] * args.env_hop_sec, n_ch - 1, 0),
    )
    ax_env.set_title("Log RMS envelope (channels x time)")
    ax_env.set_ylabel("Channel index")
    ax_env.set_xlabel("Time (s)")
    plt.colorbar(im_env, ax=ax_env, orientation="vertical", label="log10(RMS)")
    fig_env.tight_layout()
    fig_env.savefig(os.path.join(outdir, "envelope_heatmap.png"), dpi=200)
    plt.close(fig_env)

    # 输出二值 mask
    fig_mask, ax_mask = plt.subplots(figsize=(10, 4))
    im_mask = ax_mask.imshow(
        mask_connected.T.astype(float),
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="gray_r",
        vmin=0,
        vmax=1,
        extent=(0, env.shape[0] * args.env_hop_sec, n_ch - 1, 0),
    )
    ax_mask.set_title("Binary mask (cleaned, channels x time)")
    ax_mask.set_ylabel("Channel index")
    ax_mask.set_xlabel("Time (s)")
    plt.colorbar(im_mask, ax=ax_mask, orientation="vertical", label="mask")
    fig_mask.tight_layout()
    fig_mask.savefig(os.path.join(outdir, "binary_mask.png"), dpi=200)
    plt.close(fig_mask)

    # 输出 f-k 频谱图（时间-通道二维 FFT）
    fk = np.fft.fft2(X_filt, axes=(0, 1))
    fk = np.fft.fftshift(fk, axes=(0, 1))
    fk_mag = np.abs(fk)
    fk_db = 20 * np.log10(fk_mag + 1e-12)

    f_axis = np.fft.fftshift(np.fft.fftfreq(n_time, d=1.0 / args.fs))
    k_axis = np.fft.fftshift(np.fft.fftfreq(n_ch, d=args.dx))

    vmax_fk = np.percentile(fk_db, 99)
    vmin_fk = vmax_fk - 60
    fig_fk, ax_fk = plt.subplots(figsize=(6.5, 4.5))
    im_fk = ax_fk.imshow(
        fk_db.T,
        aspect="auto",
        origin="lower",
        extent=(f_axis[0], f_axis[-1], k_axis[0], k_axis[-1]),
        cmap="magma",
        vmin=vmin_fk,
        vmax=vmax_fk,
    )
    ax_fk.set_title("f-k spectrum (bandpass only)")
    ax_fk.set_xlabel("Frequency (Hz)")
    ax_fk.set_ylabel("Wavenumber (cycles/m)")
    plt.colorbar(im_fk, ax=ax_fk, orientation="vertical", label="Magnitude (dB)")
    fig_fk.tight_layout()
    fig_fk.savefig(os.path.join(outdir, "bandpass_only_fk.png"), dpi=200)
    plt.close(fig_fk)

    print("[Finish] 输出文件已保存到:", outdir)


if __name__ == "__main__":
    main()
